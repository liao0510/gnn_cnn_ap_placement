"""Direct candidate-to-AP context and explicit AP proximity/density features."""
import math
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package='APPlacement')
class SpatialAPAttention(layers.Layer):
    """Attend only to real APs, retaining geometry even with just one AP.

    Inputs: CNN [B,H,W,C], GNN [B,N,G], raw nodes [B,N,15], node mask
    [B,N], normalized adjacency [B,N,N]. Client neighbors of each AP supply
    its exact association count. Raw AP features supply demand, delivered
    rate, satisfaction, mesh membership, RU/time shares and residual demand.

    Output: learned context (units channels), followed by five unprojected
    summaries: nearest squared normalized distance / 2, nearest small-scale
    Gaussian proximity, small/large Gaussian AP densities d/(1+d), and AP
    count n/(1+n). All channels are exactly zero when no AP is placed.

    Local Gaussian scales use grid cells, not normalized map coordinates;
    defaults 5/20 cells correspond to 0.5/2 m at the current 0.1 m grid.
    These are observations, not a spacing penalty or action restriction.
    Candidate chunks and packed APs avoid a grid-by-grid attention tensor.
    """
    SUMMARY_CHANNELS = 5
    GEOMETRY_CHANNELS = 14

    def __init__(self, units=16, chunk_size=1024,
                 distance_scales_cells=(5.0, 20.0), **kwargs):
        super().__init__(**kwargs)
        self.units, self.chunk_size = int(units), int(chunk_size)
        self.distance_scales_cells = tuple(float(s) for s in distance_scales_cells)
        if self.units < 1 or self.chunk_size < 1:
            raise ValueError('Attention units and chunk_size must be positive')
        if (len(self.distance_scales_cells) != 2 or
                any(not math.isfinite(s) or s <= 0 for s in self.distance_scales_cells)):
            raise ValueError('Provide two positive finite AP distance scales')
        self.query_projection = layers.Dense(self.units, name='query')
        self.key_projection = layers.Dense(self.units, name='key')
        self.value_projection = layers.Dense(self.units, name='value')
        self.relative_hidden = layers.Dense(16, activation='tanh', name='relative_hidden')
        self.relative_score = layers.Dense(1, use_bias=False, name='relative_score')
        self.output_projection = layers.Dense(self.units, activation='tanh', name='context')

    def build(self, shapes):
        cnn_channels, graph_channels, node_features = shapes[0][-1], shapes[1][-1], shapes[2][-1]
        if node_features is None or node_features < 15:
            raise ValueError('AP attention requires the 15-feature AP/client state')
        node_width = graph_channels + node_features + cnn_channels + 1
        child_shapes = (
            (self.query_projection, (None, None, cnn_channels + 2)),
            (self.key_projection, (None, None, node_width)),
            (self.value_projection, (None, None, node_width)),
            (self.relative_hidden, (None, None, None, self.GEOMETRY_CHANNELS)),
            (self.relative_score, (None, None, None, 16)),
            (self.output_projection, (None, None, self.units + self.GEOMETRY_CHANNELS)))
        for child, child_shape in child_shapes:
            with tf.name_scope(child.name):
                child.build(child_shape)
        super().build(shapes)

    def call(self, inputs):
        cnn, graph_nodes, nodes, node_mask, adjacency = inputs
        shape = tf.shape(cnn)
        batch, height, width = shape[0], shape[1], shape[2]
        dtype = cnn.dtype
        extent = tf.cast(tf.stack([tf.maximum(width - 1, 1), tf.maximum(height - 1, 1)]), dtype)
        gx = tf.cast(tf.range(width), dtype) / extent[0]
        gy = tf.cast(tf.range(height), dtype) / extent[1]
        xx, yy = tf.meshgrid(gx, gy)
        coordinates = tf.reshape(tf.stack([xx, yy], axis=-1), [-1, 2])
        positions = tf.shape(coordinates)[0]
        grid_xy = tf.broadcast_to(coordinates[None], [batch, positions, 2])
        flat_cnn = tf.reshape(cnn, [batch, positions, tf.shape(cnn)[-1]])
        query = self.query_projection(tf.concat([flat_cnn, grid_xy], axis=-1))

        xy = tf.clip_by_value(nodes[..., 2:4], 0.0, 1.0)
        cell_x = tf.cast(tf.round(xy[..., 0] * tf.cast(width - 1, dtype)), tf.int32)
        cell_y = tf.cast(tf.round(xy[..., 1] * tf.cast(height - 1, dtype)), tf.int32)
        batch_ids = tf.broadcast_to(tf.range(batch)[:, None], tf.shape(cell_x))
        local_cnn = tf.gather_nd(cnn, tf.stack([batch_ids, cell_y, cell_x], axis=-1))
        valid_clients = tf.cast((node_mask > 0) & (nodes[..., 1] > 0.5), dtype)
        client_count = tf.reduce_sum(tf.cast(adjacency > 0, dtype) * valid_clients[:, None, :], axis=-1)
        client_load = client_count / (1.0 + client_count)
        node_context = tf.concat([graph_nodes, nodes, local_cnn, client_load[..., None]], axis=-1)
        valid = (node_mask > 0) & (nodes[..., 0] > 0.5)
        total_aps = tf.reduce_sum(tf.cast(valid, dtype), axis=1)
        packed_count = tf.maximum(tf.reduce_max(tf.reduce_sum(tf.cast(valid, tf.int32), axis=1)), 1)
        ap_ids = tf.argsort(tf.cast(valid, tf.int32), axis=1, direction='DESCENDING', stable=True)[:, :packed_count]
        node_context = tf.gather(node_context, ap_ids, batch_dims=1)
        xy = tf.gather(xy, ap_ids, batch_dims=1)
        valid = tf.gather(valid, ap_ids, batch_dims=1)
        # Eight existing raw service fields plus exact association count.
        services = tf.concat([client_load[..., None], nodes[..., 7:15]], axis=-1)
        services = tf.gather(services, ap_ids, batch_dims=1)
        valid_float = tf.cast(valid, dtype)
        key = self.key_projection(node_context)
        value = self.value_projection(node_context)
        count_summary = total_aps / (1.0 + total_aps)
        scales = tf.constant(self.distance_scales_cells, dtype=dtype)

        chunks = tf.TensorArray(dtype, size=0, dynamic_size=True, infer_shape=False,
            element_shape=tf.TensorShape([None, None, self.units + self.SUMMARY_CHANNELS]))

        def body(start, index, output):
            end = tf.minimum(start + self.chunk_size, positions)
            count = end - start
            relative = xy[:, None, :, :] - coordinates[None, start:end, None, :]
            squared = tf.reduce_sum(tf.square(relative), axis=-1, keepdims=True)
            grid_squared = tf.reduce_sum(tf.square(relative * extent), axis=-1, keepdims=True)
            proximity = tf.exp(-0.5 * grid_squared / tf.square(scales))
            service = tf.broadcast_to(services[:, None], [batch, count, packed_count, 9])
            geometry = tf.concat([relative, squared, proximity, service], axis=-1)
            scores = tf.matmul(query[:, start:end], key, transpose_b=True)
            scores /= tf.sqrt(tf.cast(self.units, dtype))
            scores += tf.squeeze(self.relative_score(self.relative_hidden(geometry)), axis=-1)
            scores = tf.where(valid[:, None, :], scores, tf.cast(-1e9, dtype))
            weights = tf.nn.softmax(scores, axis=-1) * valid_float[:, None, :]
            weights = tf.math.divide_no_nan(weights, tf.reduce_sum(weights, axis=-1, keepdims=True))
            context = tf.matmul(weights, value)
            relative_context = tf.reduce_sum(weights[..., None] * geometry, axis=2)
            learned = self.output_projection(tf.concat([context, relative_context], axis=-1))

            # These observations bypass the learned projection: softmax cannot
            # hide a nearby AP, nor average several colocated APs into one.
            nearest = tf.reduce_min(tf.where(valid[:, None, :, None], squared,
                                            tf.cast(2.0, dtype)), axis=2) * 0.5
            masked_proximity = proximity * valid_float[:, None, :, None]
            closest = tf.reduce_max(masked_proximity[..., :1], axis=2)
            density = tf.reduce_sum(masked_proximity, axis=2)
            density = density / (1.0 + density)
            ap_count = tf.broadcast_to(count_summary[:, None, None], [batch, count, 1])
            summaries = tf.concat([nearest, closest, density, ap_count], axis=-1)
            context = tf.concat([learned, summaries], axis=-1)
            context *= tf.cast(total_aps > 0, dtype)[:, None, None]
            return end, index + 1, output.write(index, tf.transpose(context, [1, 0, 2]))

        _, _, chunks = tf.while_loop(lambda start, index, output: start < positions,
            body, (0, 0, chunks), parallel_iterations=1)
        context = tf.transpose(chunks.concat(), [1, 0, 2])
        result = tf.reshape(context, [batch, height, width, self.units + self.SUMMARY_CHANNELS])
        result.set_shape([cnn.shape[0], cnn.shape[1], cnn.shape[2], self.units + self.SUMMARY_CHANNELS])
        return result

    def compute_output_shape(self, shapes):
        return (*shapes[0][:3], self.units + self.SUMMARY_CHANNELS)

    def get_config(self):
        return dict(super().get_config(), units=self.units, chunk_size=self.chunk_size,
                    distance_scales_cells=self.distance_scales_cells)
