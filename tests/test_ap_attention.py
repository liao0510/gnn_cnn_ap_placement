"""AP geometry, learned context, gradients and exact-cell exclusion."""
import unittest
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
from training_method.AP_attention import SpatialAPAttention
from training_method.Network_state import build_state, NODE_FEATURES
from training_method.Network_environment import run_network_environment
from training_method.Network_policy import placement_mask, masked_log_probabilities
from config_radio import ChannelConfig
from method.Cal_link_capacity import RU_DATA_SUBCARRIERS


class APAttentionTests(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(91)
        rng = np.random.default_rng(29)
        self.cnn = tf.constant(rng.normal(size=(2, 5, 7, 4)).astype(np.float32))
        self.graph = tf.constant(rng.normal(size=(2, 5, 6)).astype(np.float32))
        self.nodes = np.zeros((2, 5, NODE_FEATURES), np.float32)
        self.nodes[:, :2, 0] = 1
        self.nodes[:, 0, 2:4] = [1/6, 1/4]
        self.nodes[:, 1, 2:4] = [5/6, 3/4]
        self.nodes[:, 0, 7:15] = [.5, .3, .6, 1, .5, 1, .2, .4]
        self.nodes[:, 1, 7:15] = [.2, .2, 1, 1, 1, 1, 0, 0]
        self.nodes[:, 2, 1] = 1
        self.nodes[:, 2, 2:4] = [.5, .5]
        self.mask = np.array([[1, 1, 1, 0, 0]] * 2, np.float32)
        self.adjacency = np.broadcast_to(np.eye(5, dtype=np.float32), (2, 5, 5)).copy()
        self.adjacency[:, 0, 2] = self.adjacency[:, 2, 0] = .5
        self.layer = SpatialAPAttention(units=8, chunk_size=7, distance_scales_cells=(1., 3.))

    def inputs(self, nodes=None, mask=None):
        return [self.cnn, self.graph, self.nodes if nodes is None else nodes,
                self.mask if mask is None else mask, self.adjacency]

    def test_permutation_and_masked_padding_preserve_context(self):
        expected = self.layer(self.inputs())
        permutation = [3, 1, 4, 0, 2]
        actual = self.layer([self.cnn, tf.gather(self.graph, permutation, axis=1),
            self.nodes[:, permutation], self.mask[:, permutation],
            self.adjacency[:, permutation][:, :, permutation]])
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        changed = self.nodes.copy(); changed[:, 3:] = 1000
        graph = self.graph.numpy(); graph[:, 3:] = 1000
        adjacency = self.adjacency.copy(); adjacency[:, :, 3:] = 1000
        actual = self.layer([self.cnn, graph, changed, self.mask, adjacency])
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_exact_location_and_number_are_visible_without_learned_projection(self):
        single = self.nodes.copy(); single[:, 1, 0] = 0
        one = self.layer(self.inputs(single)).numpy()
        # Raw summaries bypass the learned output projection.
        summary = one[..., 8:]
        self.assertAlmostEqual(float(summary[0, 1, 1, 0]), 0.)
        self.assertAlmostEqual(float(summary[0, 1, 1, 1]), 1.)
        self.assertGreater(float(summary[0, 1, 1, 1]), float(summary[0, 4, 6, 1]))
        self.assertAlmostEqual(float(summary[0, 1, 1, 4]), .5)
        # Adding a colocated AP must increase density rather than averaging away.
        nearby = self.nodes.copy(); nearby[:, 1, 2:4] = nearby[:, 0, 2:4]
        two = self.layer(self.inputs(nearby)).numpy()
        self.assertGreater(float(two[0, 1, 1, 10]), float(one[0, 1, 1, 10]))
        self.assertAlmostEqual(float(two[0, 1, 1, 12]), 2/3, places=6)
        moved = single.copy(); moved[:, 0, 2] += 1/6
        after = self.layer(self.inputs(moved)).numpy()
        self.assertGreater(float(np.max(np.abs(after - one))), 1e-4)

    def test_local_scale_uses_cells_on_different_map_sizes(self):
        node = np.zeros((1, 1, NODE_FEATURES), np.float32); node[..., 0] = 1
        values = []
        for size in (5, 9):
            args = [np.zeros((1, size, size, 4), np.float32), np.zeros((1, 1, 6), np.float32),
                    node, np.ones((1, 1), np.float32), np.ones((1, 1, 1), np.float32)]
            values.append(float(self.layer(args)[0, 0, 2, 9]))
        np.testing.assert_allclose(values, [np.exp(-2.)] * 2, atol=1e-6)

    def test_empty_and_mixed_AP_counts_are_finite_and_zero_for_no_AP(self):
        empty = self.nodes.copy(); empty[..., 0] = 0
        result = self.layer(self.inputs(empty))
        np.testing.assert_array_equal(result, 0.)
        mixed = self.nodes.copy(); mixed[0, :, 0] = 0
        result = self.layer(self.inputs(mixed)).numpy()
        np.testing.assert_array_equal(result[0], 0.)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertGreater(float(np.abs(result[1]).sum()), 0.)

    def test_compiled_gradients_and_chunk_size_are_consistent(self):
        expected = self.layer(self.inputs())
        compiled = tf.function(self.layer)
        np.testing.assert_allclose(compiled(self.inputs()), expected, atol=1e-6)
        variable = tf.Variable(self.nodes)
        with tf.GradientTape(persistent=True) as tape:
            output = compiled(self.inputs(variable))
            loss = tf.reduce_sum(tf.square(output))
        coordinate_gradient = tape.gradient(loss, variable).numpy()
        gradients = tape.gradient(loss, self.layer.trainable_variables)
        self.assertTrue(np.all(np.isfinite(coordinate_gradient)))
        self.assertGreater(float(np.abs(coordinate_gradient[:, :2, 2:4]).sum()), 1e-6)
        self.assertTrue(all(g is not None and np.all(np.isfinite(g)) for g in gradients))
        self.assertTrue(any(float(tf.reduce_sum(tf.abs(g))) > 0 for g in gradients))
        unchunked = SpatialAPAttention(units=8, chunk_size=100, distance_scales_cells=(1., 3.))
        unchunked(self.inputs()); unchunked.set_weights(self.layer.get_weights())
        np.testing.assert_allclose(unchunked(self.inputs()), expected, atol=1e-6)

    def test_service_load_and_allocation_fields_affect_learned_context(self):
        before = self.layer(self.inputs()).numpy()
        loaded = self.nodes.copy(); loaded[:, 0, 7:15] = [1, .2, .2, 1, .1, .5, .8, .8]
        after = self.layer(self.inputs(loaded)).numpy()
        self.assertGreater(float(np.max(np.abs(after[..., :8] - before[..., :8]))), 1e-5)
        np.testing.assert_array_equal(after[..., 8:], before[..., 8:])


class APStateAndMaskTests(unittest.TestCase):
    def test_AP_allocation_share_fields_match_environment(self):
        walls = np.zeros((8, 8)); signal = np.full(walls.shape, -40.)
        aps, clients = [(1, 1)], [(3, 3), (5, 5)]
        net = run_network_environment(aps, clients, [100., 100.], [signal], ChannelConfig())
        state = build_state(walls, walls, aps, clients, [100., 100.], signal, net, 1, 5, 8)
        resource = net['ap_resources'][0]
        self.assertAlmostEqual(float(state['nodes'][0, 11]),
            RU_DATA_SUBCARRIERS[resource['ru_tones']] / net['full_channel_data_subcarriers'])
        self.assertEqual(float(state['nodes'][0, 12]), resource['client_time_fraction'])
        np.testing.assert_array_equal(state['map'][..., 2], np.eye(8)[1, :, None] * np.eye(8)[1, None, :])

    def test_peaked_logits_cannot_select_any_occupied_cell(self):
        walls = np.zeros((5, 7)); forbidden = walls.copy()
        walls[0, 0] = 10; forbidden[0, 1] = 1
        clients = [(1, 1)]
        aps = []
        logits = np.arange(35, dtype=np.float32).reshape(1, 5, 7, 1)
        for _ in range(5):
            mask = placement_mask(walls, forbidden, aps + clients)
            for x, y in aps + clients:self.assertFalse(mask[y, x])
            logs = masked_log_probabilities(logits, mask[None]).numpy()
            index = int(np.argmax(logs)); aps.append((index % 7, index // 7))
        self.assertEqual(len(set(aps + clients)), 6)


if __name__ == '__main__':
    unittest.main()
