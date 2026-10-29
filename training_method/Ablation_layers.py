"""Serializable layers shared by the four-input ablation models."""
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package='APPlacement')
class MapOnly(layers.Layer):
    def call(self, inputs):
        # Keep the four-input API without reading graph tensor values.
        return tf.identity(inputs[0])


@tf.keras.utils.register_keras_serializable(package='APPlacement')
class NeighborAggregation(layers.Layer):
    def call(self, inputs):
        return tf.matmul(inputs[0], inputs[1])


@tf.keras.utils.register_keras_serializable(package='APPlacement')
class MaskedMean(layers.Layer):
    def call(self, inputs):
        nodes, mask = inputs
        return tf.reduce_sum(nodes * mask[..., None], axis=1) / tf.maximum(
            tf.reduce_sum(mask, axis=1, keepdims=True), 1.0)


@tf.keras.utils.register_keras_serializable(package='APPlacement')
class CoordinateQueries(layers.Layer):
    def call(self, inputs):
        graph, raster = inputs
        # Only the raster shape is used: no materials, RSSI or other raster values.
        shape = tf.shape(raster)
        batch, height, width = shape[0], shape[1], shape[2]
        x, y = tf.meshgrid(tf.range(width), tf.range(height))
        xy = tf.stack([tf.cast(x, graph.dtype) / tf.cast(tf.maximum(width-1, 1), graph.dtype),
                       tf.cast(y, graph.dtype) / tf.cast(tf.maximum(height-1, 1), graph.dtype)], -1)
        xy = tf.tile(xy[None], [batch, 1, 1, 1])
        features = tf.tile(graph[:, None, None, :], [1, height, width, 1])
        return tf.concat([features, xy], axis=-1)


@tf.keras.utils.register_keras_serializable(package='APPlacement')
class SpatialSoftmax(layers.Layer):
    def call(self, logits):
        shape = tf.shape(logits)
        flat = tf.reshape(logits, [shape[0], -1])
        flat -= tf.reduce_max(flat, axis=-1, keepdims=True)
        p = tf.clip_by_value(tf.nn.softmax(flat), 1e-7, 1.0)
        p /= tf.reduce_sum(p, axis=-1, keepdims=True)
        return tf.reshape(p, shape)
