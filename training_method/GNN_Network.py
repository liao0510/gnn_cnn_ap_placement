"""GNN-only encoder; map values are ignored, shape defines action grid."""
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
    g = layers.Dense(64, activation='relu')(node_input)
    g = NeighborAggregation()([adjacency_input, g])
    g = layers.Dense(64, activation='relu')(g)
    g = NeighborAggregation()([adjacency_input, g])
    g = layers.Dense(128, activation='relu')(g)
    graph = MaskedMean()([g, node_mask_input])
    graph = layers.Dense(128, activation='relu')(graph)
    spatial = CoordinateQueries()([graph, map_input])
    spatial = layers.Dense(128, activation='relu')(spatial)
    logits = layers.Dense(1)(spatial)
    actor = SpatialSoftmax(name='actor')(logits)
    v = layers.GlobalAveragePooling2D()(spatial)
    v = layers.Concatenate()([v, graph])
    v = layers.Dense(128, activation='relu')(v)
    v = layers.Dense(64, activation='relu')(v)
    critic = layers.Dense(1, name='critic')(v)
    return models.Model(inputs, [actor, critic], name='GNN_only_actor_critic')


if __name__ == "__main__":
    build_network().summary()
