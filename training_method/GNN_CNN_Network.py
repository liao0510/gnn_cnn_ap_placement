import tensorflow as tf
from tensorflow.keras import layers, models


def build_network(cnn_channels=6, node_features=13):
    map_input = layers.Input((None, None, cnn_channels), name='map_input')
    node_input = layers.Input((None, node_features), name='node_input')
    adjacency_input = layers.Input((None, None), name='adjacency_input')
    node_mask_input = layers.Input((None,), name='node_mask_input')

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(map_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    cnn_feature = layers.Conv2D(128, 3, padding='same', activation='relu', dilation_rate=4)(x)

    # Simple GCN-style message passing. A is normalized by Network_state.
    g = layers.Dense(64, activation='relu')(node_input)
    g = layers.Lambda(lambda z: tf.matmul(z[0], z[1]))([adjacency_input, g])
    g = layers.Dense(64, activation='relu')(g)
    g = layers.Lambda(lambda z: tf.matmul(z[0], z[1]))([adjacency_input, g])
    g = layers.Dense(128, activation='relu')(g)

    mask = layers.Lambda(lambda m: tf.expand_dims(m, -1))(node_mask_input)
    g = layers.Multiply()([g, mask])
    graph_sum = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(g)
    node_count = layers.Lambda(lambda m: tf.maximum(tf.reduce_sum(m, axis=1, keepdims=True), 1.0))(node_mask_input)
    graph_feature = layers.Lambda(lambda z: z[0] / z[1])([graph_sum, node_count])
    graph_feature = layers.Dense(128, activation='relu')(graph_feature)

    def broadcast_graph(z):
        cnn, graph = z
        shape = tf.shape(cnn)
        graph = tf.reshape(graph, [-1, 1, 1, 128])
        return tf.tile(graph, [1, shape[1], shape[2], 1])

    graph_spatial = layers.Lambda(broadcast_graph)([cnn_feature, graph_feature])
    fused = layers.Concatenate()([cnn_feature, graph_spatial])
    fused = layers.Conv2D(128, 3, padding='same', activation='relu')(fused)

    logits = layers.Conv2D(1, 1, padding='same')(fused)
    def spatial_softmax(t):
        shape = tf.shape(t)
        flat = tf.reshape(t, [shape[0], -1])
        flat -= tf.reduce_max(flat, axis=-1, keepdims=True)
        p = tf.nn.softmax(flat)
        p = tf.clip_by_value(p, 1e-7, 1.0)
        p /= tf.reduce_sum(p, axis=-1, keepdims=True)
        return tf.reshape(p, shape)
    actor = layers.Lambda(spatial_softmax, name='actor')(logits)

    v = layers.GlobalAveragePooling2D()(fused)
    v = layers.Concatenate()([v, graph_feature])
    v = layers.Dense(128, activation='relu')(v)
    v = layers.Dense(64, activation='relu')(v)
    critic = layers.Dense(1, name='critic')(v)
    return models.Model([map_input, node_input, adjacency_input, node_mask_input], [actor, critic])

if __name__ == "__main__":
    model = build_network(7,13)
    model.summary()
