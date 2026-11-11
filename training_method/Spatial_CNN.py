"""Multiscale spatial encoder with exact-grid skips and global context.

    Full -> 1/2 -> 1/4 -> 1/8 -> 1/16 resolution. Most wide convolutions
    run on small maps; raw input and a full-resolution skip preserve details.
"""
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package='APPlacement')
class ResizeLike(layers.Layer):
    """Resize features to an actual reference grid, including odd dimensions."""
    def call(self, inputs):
        source, reference = inputs
        return tf.image.resize(source, tf.shape(reference)[1:3], method='bilinear')

    def compute_output_shape(self, shapes):
        return (shapes[0][0], shapes[1][1], shapes[1][2], shapes[0][-1])


@tf.keras.utils.register_keras_serializable(package='APPlacement')
class AdaptiveAveragePool2D(layers.Layer):
    """Differentiable adaptive bins, even when a feature grid is smaller.

    Explicit means avoid ResizeArea's missing gradient on some TF 2.10 builds.
    Floor/ceil bin boundaries cover every input cell without cropping.
    """
    def __init__(self, bins, **kwargs):
        super().__init__(**kwargs)
        self.bins = int(bins)
        if self.bins < 1:
            raise ValueError('bins must be positive')

    def call(self, inputs):
        height, width = tf.shape(inputs)[1], tf.shape(inputs)[2]
        rows = []
        for y in range(self.bins):
            top = y * height // self.bins
            bottom = ((y + 1) * height + self.bins - 1) // self.bins
            cells = []
            for x in range(self.bins):
                left = x * width // self.bins
                right = ((x + 1) * width + self.bins - 1) // self.bins
                cells.append(tf.reduce_mean(inputs[:, top:bottom, left:right, :], axis=(1, 2)))
            rows.append(tf.stack(cells, axis=1))
        return tf.stack(rows, axis=1)

    def compute_output_shape(self, shape):
        return (shape[0], self.bins, self.bins, shape[-1])

    def get_config(self):
        return dict(super().get_config(), bins=self.bins)


@tf.keras.utils.register_keras_serializable(package='APPlacement')
class BroadcastGraph(layers.Layer):
    """Keep the existing global graph fusion, with a compact projection."""
    def call(self, inputs):
        reference, graph = inputs
        shape = tf.shape(reference)
        return tf.tile(graph[:, None, None, :], [1, shape[1], shape[2], 1])

    def compute_output_shape(self, shapes):
        return (shapes[0][0], shapes[0][1], shapes[0][2], shapes[1][-1])


def conv_block(features, channels, name):
    for index in (1, 2):
        features = layers.Conv2D(channels, 3, padding='same', activation='relu',
                                 kernel_initializer='he_normal', name=f'{name}_conv{index}')(features)
    return features


def build_spatial_encoder(map_input):
    """Return one feature vector per original grid cell; no action downsampling."""
    widths = (16, 24, 40, 64)
    skips = []
    features = map_input
    for level, channels in enumerate(widths):
        if level:
            features = layers.MaxPooling2D(2, padding='same', name=f'cnn_pool{level}')(features)
        features = conv_block(features, channels, f'cnn_encoder{level}')
        skips.append(features)

    features = layers.MaxPooling2D(2, padding='same', name='cnn_pool4')(features)
    for rate in (1, 2, 4, 8):
        features = layers.Conv2D(96, 3, padding='same', dilation_rate=rate,
            activation='relu', kernel_initializer='he_normal', name=f'cnn_context_d{rate}')(features)

    # Global and regional summaries give every candidate access to all cells,
    # even on maps larger than the dilated path's finite receptive field.
    context = [features]
    for bins in (1, 2, 4):
        pooled = AdaptiveAveragePool2D(bins, name=f'cnn_context_pool{bins}')(features)
        pooled = layers.Conv2D(16, 1, activation='relu', name=f'cnn_context_project{bins}')(pooled)
        context.append(ResizeLike(name=f'cnn_context_resize{bins}')([pooled, features]))
    features = layers.Concatenate(name='cnn_context_concat')(context)
    features = layers.Conv2D(96, 1, activation='relu', name='cnn_context_fusion')(features)

    for level in reversed(range(len(widths))):
        features = ResizeLike(name=f'cnn_decoder{level}_resize')([features, skips[level]])
        features = layers.Conv2D(widths[level], 1, activation='relu',
                                 name=f'cnn_decoder{level}_project')(features)
        detail = [features, skips[level]]
        if level == 0:
            # The raw channels keep a one-cell client / thin wall at its
            # original coordinate instead of relying on pooled reconstruction.
            detail.append(map_input)
        features = layers.Concatenate(name=f'cnn_decoder{level}_skip')(detail)
        features = conv_block(features, widths[level], f'cnn_decoder{level}')
    return layers.Activation('linear', name='cnn_features')(features)
