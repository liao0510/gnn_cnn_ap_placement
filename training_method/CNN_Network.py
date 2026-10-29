"""CNN-only encoder; graph inputs are interface placeholders."""
import tensorflow as tf
from tensorflow.keras import layers, models
from training_method.Ablation_layers import (
    MapOnly, NeighborAggregation, MaskedMean, CoordinateQueries, SpatialSoftmax)


def build_network(cnn_channels=6, node_features=13):
    map_input = layers.Input((None, None, cnn_channels), name='map_input')
    node_input = layers.Input((None, node_features), name='node_input')
    adjacency_input = layers.Input((None, None), name='adjacency_input')
    node_mask_input = layers.Input((None,), name='node_mask_input')
    inputs = [map_input, node_input, adjacency_input, node_mask_input]
    x = MapOnly(name='map_only')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', dilation_rate=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', dilation_rate=4)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    logits = layers.Conv2D(1, 1, padding='same')(x)
    actor = SpatialSoftmax(name='actor')(logits)
    v = layers.GlobalAveragePooling2D()(x)
    v = layers.Dense(128, activation='relu')(v)
    v = layers.Dense(64, activation='relu')(v)
    critic = layers.Dense(1, name='critic')(v)
    return models.Model(inputs, [actor, critic], name='CNN_only_actor_critic')


if __name__ == "__main__":
    build_network().summary()
