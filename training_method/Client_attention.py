"""Candidate-to-client attention without a grid-by-grid attention matrix."""
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package='APPlacement')
class SpatialClientAttention(layers.Layer):
    """Keep per-client identity and relative geometry at every action cell.

    Inputs: CNN [B,H,W,C], GNN [B,N,G], raw nodes [B,N,F], node mask [B,N].
    Only real client nodes are keys/values. Served clients remain visible so
    their association and RU allocation can still change after a placement.
    Candidate chunks bound the temporary pair tensor; no H*W by H*W matrix.
    """
    def __init__(self, units=16, chunk_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.units, self.chunk_size = int(units), int(chunk_size)
        if self.units < 1 or self.chunk_size < 1:
            raise ValueError('Attention units and chunk_size must be positive')
        self.query_projection = layers.Dense(self.units, name='query')
        self.key_projection = layers.Dense(self.units, name='key')
        self.value_projection = layers.Dense(self.units, name='value')
        self.relative_hidden = layers.Dense(8, activation='tanh', name='relative_hidden')
        self.relative_score = layers.Dense(1, use_bias=False, name='relative_score')
        self.output_projection = layers.Dense(self.units, activation='tanh', name='context')

    def build(self, shapes):
        cnn_channels, graph_channels, node_features = shapes[0][-1], shapes[1][-1], shapes[2][-1]
        if node_features is None or node_features < 15:
            raise ValueError('Client attention requires the 15-feature residual-demand state')
        node_width = graph_channels + node_features + cnn_channels
        child_shapes = (
            (self.query_projection, (None, None, cnn_channels + 2)),
            (self.key_projection, (None, None, node_width)),
            (self.value_projection, (None, None, node_width)),
            (self.relative_hidden, (None, None, None, 5)),
            (self.relative_score, (None, None, None, 8)),
            (self.output_projection, (None, None, self.units + 5)))
        for child, child_shape in child_shapes:
            # Direct build() calls need the child scope for unique HDF5 names.
            with tf.name_scope(child.name):
                child.build(child_shape)
        super().build(shapes)

    def call(self, inputs):
        cnn, graph_nodes, nodes, node_mask = inputs
        shape = tf.shape(cnn)
        batch, height, width = shape[0], shape[1], shape[2]
        dtype = cnn.dtype
        gx = tf.cast(tf.range(width), dtype) / tf.cast(tf.maximum(width - 1, 1), dtype)
        gy = tf.cast(tf.range(height), dtype) / tf.cast(tf.maximum(height - 1, 1), dtype)
        xx, yy = tf.meshgrid(gx, gy)
        coordinates = tf.reshape(tf.stack([xx, yy], axis=-1), [-1, 2])
        positions = tf.shape(coordinates)[0]
        grid_coordinates = tf.broadcast_to(coordinates[None], [batch, positions, 2])
        flat_cnn = tf.reshape(cnn, [batch, positions, tf.shape(cnn)[-1]])
        query = self.query_projection(tf.concat([flat_cnn, grid_coordinates], axis=-1))

        # Exact-cell CNN features also give each client local map context.
        xy = tf.clip_by_value(nodes[..., 2:4], 0.0, 1.0)
        cell_x = tf.cast(tf.round(xy[..., 0] * tf.cast(tf.maximum(width - 1, 0), dtype)), tf.int32)
        cell_y = tf.cast(tf.round(xy[..., 1] * tf.cast(tf.maximum(height - 1, 0), dtype)), tf.int32)
        batch_ids = tf.broadcast_to(tf.range(batch)[:, None], tf.shape(cell_x))
        local_cnn = tf.gather_nd(cnn, tf.stack([batch_ids, cell_y, cell_x], axis=-1))
        node_context = tf.concat([graph_nodes, nodes, local_cnn], axis=-1)
        valid = (node_mask > 0) & (nodes[..., 1] > 0.5)
        # Pack clients before constructing pairs. MAX_NODES may be 64 while
        # only 10 clients exist; padding and APs must not multiply pair memory.
        client_count = tf.maximum(tf.reduce_max(tf.reduce_sum(tf.cast(valid, tf.int32), axis=1)), 1)
        client_ids = tf.argsort(tf.cast(valid, tf.int32), axis=1, direction='DESCENDING', stable=True)
        client_ids = client_ids[:, :client_count]
        node_context = tf.gather(node_context, client_ids, batch_dims=1)
        xy = tf.gather(xy, client_ids, batch_dims=1)
        valid = tf.gather(valid, client_ids, batch_dims=1)
        shortfall = tf.gather(nodes[..., 13:15], client_ids, batch_dims=1)
        key = self.key_projection(node_context)
        value = self.value_projection(node_context)
        valid_float = tf.cast(valid, dtype)

        chunks = tf.TensorArray(dtype, size=0, dynamic_size=True, infer_shape=False,
                                element_shape=tf.TensorShape([None, None, self.units + 5]))

        def body(start, index, output):
            end = tf.minimum(start + self.chunk_size, positions)
            count = end - start
            relative = xy[:, None, :, :] - coordinates[None, start:end, None, :]
            distance_squared = tf.reduce_sum(tf.square(relative), axis=-1, keepdims=True)
            demand = tf.broadcast_to(shortfall[:, None], [batch, count, client_count, 2])
            geometry = tf.concat([relative, distance_squared, demand], axis=-1)
            scores = tf.matmul(query[:, start:end], key, transpose_b=True)
            scores /= tf.sqrt(tf.cast(self.units, dtype))
            scores += tf.squeeze(self.relative_score(self.relative_hidden(geometry)), axis=-1)
            scores = tf.where(valid[:, None, :], scores, tf.cast(-1e9, dtype))
            weights = tf.nn.softmax(scores, axis=-1) * valid_float[:, None, :]
            weights = tf.math.divide_no_nan(weights, tf.reduce_sum(weights, axis=-1, keepdims=True))
            context = tf.matmul(weights, value)
            # Keep relative geometry in the values too: even one client must
            # produce candidate-dependent context (softmax over one key is 1).
            relative_context = tf.reduce_sum(weights[..., None] * geometry, axis=2)
            context = tf.concat([context, relative_context], axis=-1)
            return end, index + 1, output.write(index, tf.transpose(context, [1, 0, 2]))

        _, _, chunks = tf.while_loop(lambda start, index, output: start < positions,
                                     body, (0, 0, chunks), parallel_iterations=1)
        context = tf.transpose(chunks.concat(), [1, 0, 2])
        context = self.output_projection(context)
        # Exactly zero when there are no clients, including projection biases.
        context *= tf.cast(tf.reduce_any(valid, axis=1), dtype)[:, None, None]
        result = tf.reshape(context, [batch, height, width, self.units])
        result.set_shape([cnn.shape[0], cnn.shape[1], cnn.shape[2], self.units])
        return result

    def compute_output_shape(self, shapes):
        return (*shapes[0][:3], self.units)

    def get_config(self):
        return dict(super().get_config(), units=self.units, chunk_size=self.chunk_size)
