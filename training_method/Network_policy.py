"""Shared placement constraints and numerically stable categorical policy."""
import numpy as np
import tensorflow as tf


def placement_mask(walls, forbidden, occupied_positions=()):
    """Exclude walls, forbidden cells, and occupied (x, y) positions."""
    walls, forbidden = np.asarray(walls), np.asarray(forbidden)
    if walls.ndim != 2 or forbidden.shape != walls.shape:
        raise ValueError('Walls and forbidden must be matching 2-D maps')
    available = (walls == 0) & (forbidden == 0)
    for x, y in occupied_positions:
        available[y, x] = False
    if not np.any(available):
        raise ValueError('No unoccupied valid placement positions')
    return available


def build_policy_model(network):
    """Expose the existing actor logits and critic, sharing all network weights.

    Mask logits before softmax instead of using the actor's clipped, unmasked
    probabilities. The original network and its saved weight format are kept.
    """
    return tf.keras.Model(network.inputs,
                          [network.get_layer('actor').input, network.outputs[1]],
                          name='placement_policy')


def masked_log_probabilities(logits, action_masks):
    """Return [batch, height, width, 1] log probabilities; invalid cells are -inf."""
    logits = tf.convert_to_tensor(logits)
    action_masks = tf.convert_to_tensor(action_masks, dtype=tf.bool)
    tf.debugging.assert_rank(logits, 4)
    tf.debugging.assert_rank(action_masks, 3)
    tf.debugging.assert_equal(tf.shape(logits)[-1], 1)
    tf.debugging.assert_equal(tf.shape(logits)[:3], tf.shape(action_masks))
    tf.debugging.assert_all_finite(logits, 'Placement logits must be finite')
    shape = tf.shape(logits)
    flat = tf.reshape(logits, [shape[0], -1])
    valid = tf.reshape(action_masks, [shape[0], -1])
    tf.debugging.assert_equal(tf.reduce_any(valid, axis=1), True,
                              message='Each state needs a valid placement position')
    masked = tf.where(valid, flat, tf.cast(-np.inf, logits.dtype))
    return tf.reshape(tf.nn.log_softmax(masked, axis=-1), shape)
