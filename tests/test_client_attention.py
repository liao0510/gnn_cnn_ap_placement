"""Residual demand semantics, physical-scale summaries and client fusion."""
import unittest
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
import config
from config_radio import ChannelConfig
from training_method.Network_environment import run_network_environment
from training_method.Network_state import (build_state, demand_density_maps, NODE_FEATURES,
    REMAINING_DEMAND_CHANNEL, ORIGINAL_DEMAND_CHANNEL, DENSITY_CHANNELS,
    REMAINING_DEMAND_FEATURE, REMAINING_RATIO_FEATURE)
from training_method.Network_reward import calculate_reward
from training_method.Client_attention import SpatialClientAttention


class DemandStateTests(unittest.TestCase):
    def state(self, delivered, demands=(100., 100., 100.)):
        walls = np.zeros((21, 21))
        clients = [(4, 4), (10, 10), (16, 16)]
        signal = np.full(walls.shape, -40.)
        net = run_network_environment([(0, 0)], clients, list(demands), [signal], ChannelConfig())
        # Test state encoding against known measured allocation values.
        for row, value in zip(net['allocation_results'], delivered):
            row['throughput_mbps'] = value
            row['satisfaction_ratio'] = min(value / row['demand_mbps'], 1.) if row['demand_mbps'] else 1.
        net['ap_resources'][0]['throughput_mbps'] = sum(delivered)
        state = build_state(walls, walls, [(0, 0)], clients, list(demands), signal, net, 1, 5, 8)
        return state, clients, net['rate_reference_mbps']

    def test_partial_full_and_unserved_demand_preserve_original_requirements(self):
        state, clients, reference = self.state([0., 50., 100.])
        for i, ((x, y), remaining) in enumerate(zip(clients, (100., 50., 0.))):
            self.assertAlmostEqual(state['map'][y, x, REMAINING_DEMAND_CHANNEL], remaining/reference)
            self.assertAlmostEqual(state['map'][y, x, ORIGINAL_DEMAND_CHANNEL], 100./reference)
            self.assertAlmostEqual(state['nodes'][i+1, REMAINING_DEMAND_FEATURE], remaining/reference)
            self.assertAlmostEqual(state['nodes'][i+1, REMAINING_RATIO_FEATURE], remaining/100.)
            self.assertAlmostEqual(state['nodes'][i+1, 7], 100./reference)
            self.assertEqual(state['nodes'][i+1, 1], 1.)
        np.testing.assert_array_equal(state['mask'], [1, 1, 1, 1, 0, 0, 0, 0])

    def test_satisfied_clients_zero_demand_maps_and_can_become_unmet_again(self):
        satisfied, _, _ = self.state([100., 100., 100.])
        for channel in (REMAINING_DEMAND_CHANNEL, *DENSITY_CHANNELS):
            np.testing.assert_array_equal(satisfied['map'][..., channel], 0)
        rebound, clients, reference = self.state([40., 100., 100.])
        x, y = clients[0]
        self.assertAlmostEqual(rebound['map'][y, x, REMAINING_DEMAND_CHANNEL], 60./reference)
        self.assertGreater(rebound['map'][y, x, DENSITY_CHANNELS[0]], 0.)

    def test_zero_demand_is_finite_and_creates_no_remaining_demand(self):
        state, _, _ = self.state([0., 0., 0.], demands=(0., 0., 0.))
        self.assertTrue(all(np.all(np.isfinite(value)) for value in state.values()))
        np.testing.assert_array_equal(state['nodes'][:, 13:15], 0.)
        np.testing.assert_array_equal(state['map'][..., [3, 6, 7, 8]], 0.)

    def test_density_adds_demands_and_uses_physical_distance_and_fixed_scale(self):
        settings = config.snapshot()
        one = demand_density_maps((21, 21), [(0, 0)], [100.], .1, settings)
        two = demand_density_maps((21, 21), [(0, 0), (0, 0)], [100., 100.], .1, settings)
        coarse = demand_density_maps((11, 11), [(0, 0)], [100.], .2, settings)
        for a, b, c in zip(one, two, coarse):
            self.assertGreater(b[0, 0], a[0, 0])
            self.assertGreater(a[0, 0], a[0, 10])
            self.assertAlmostEqual(a[0, 10], c[0, 5])

    def test_one_point_zero_scales_use_signed_unclipped_fractional_gains(self):
        self.assertEqual(config.COVERAGE_GAIN_SCALE, 1.)
        self.assertEqual(config.THROUGHPUT_GAIN_SCALE, 1.)
        reward, parts = calculate_reward(.8, .1, [{'satisfaction_ratio': .9}], True,
                                         [{'satisfaction_ratio': .2}])
        self.assertAlmostEqual(parts['coverage'], .7)
        self.assertAlmostEqual(parts['throughput'], .7)
        self.assertAlmostEqual(reward, .2*.7 + .7*.7 + .1)
        _, drop = calculate_reward(.8, .8, [{'satisfaction_ratio': .4}], True,
                                   [{'satisfaction_ratio': .9}])
        self.assertAlmostEqual(drop['throughput'], -.5)


class ClientAttentionTests(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(32)
        rng = np.random.default_rng(14)
        self.cnn = tf.constant(rng.normal(size=(2, 5, 7, 4)).astype(np.float32))
        self.graph = tf.constant(rng.normal(size=(2, 4, 5)).astype(np.float32))
        self.nodes = np.zeros((2, 4, NODE_FEATURES), np.float32)
        self.nodes[:, :2, 1] = 1
        self.nodes[:, 0, 2:4] = [.2, .3]
        self.nodes[:, 1, 2:4] = [.8, .6]
        self.nodes[:, 0, 13:15] = [.2, .8]
        self.nodes[:, 1, 13:15] = [.1, .4]
        self.mask = np.array([[1, 1, 1, 0], [1, 1, 1, 0]], np.float32)
        self.layer = SpatialClientAttention(units=8, chunk_size=7)

    def test_client_permutation_and_masked_padding_do_not_change_output(self):
        expected = self.layer([self.cnn, self.graph, self.nodes, self.mask])
        permutation = [1, 3, 0, 2]
        actual = self.layer([self.cnn, tf.gather(self.graph, permutation, axis=1),
            self.nodes[:, permutation], self.mask[:, permutation]])
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        modified = self.nodes.copy()
        modified[:, 3] = 1000
        graph = self.graph.numpy(); graph[:, 3] = 1000
        actual = self.layer([self.cnn, graph, modified, self.mask])
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_single_client_coordinates_change_spatial_context_and_have_gradients(self):
        nodes = self.nodes.copy(); nodes[:, 1, 1] = 0
        before = self.layer([self.cnn, self.graph, nodes, self.mask])
        moved = nodes.copy(); moved[:, 0, 2:4] = [.7, .9]
        after = self.layer([self.cnn, self.graph, moved, self.mask])
        self.assertGreater(float(tf.reduce_max(tf.abs(after-before))), 1e-5)
        variable = tf.Variable(nodes)
        with tf.GradientTape() as tape:
            context = self.layer([self.cnn, self.graph, variable, self.mask])
            loss = tf.reduce_sum(context[:, 2, 3])
        grad = tape.gradient(loss, variable).numpy()
        self.assertTrue(np.all(np.isfinite(grad)))
        self.assertGreater(float(np.abs(grad[:, 0, 2:4]).sum()), 1e-6)
        self.assertGreater(float(np.abs(grad[:, 0, 13:15]).sum()), 1e-6)

    def test_no_clients_is_zero_and_tf_function_matches_eager_with_finite_gradients(self):
        empty = self.nodes.copy(); empty[..., 1] = 0
        result = self.layer([self.cnn, self.graph, empty, self.mask])
        np.testing.assert_array_equal(result, 0.)
        mixed = self.nodes.copy(); mixed[0, :, 1] = 0
        result = self.layer([self.cnn, self.graph, mixed, self.mask])
        np.testing.assert_array_equal(result[0], 0.)
        self.assertGreater(float(tf.reduce_sum(tf.abs(result[1]))), 0.)
        expected = self.layer([self.cnn, self.graph, self.nodes, self.mask])
        compiled = tf.function(self.layer)
        np.testing.assert_allclose(compiled([self.cnn, self.graph, self.nodes, self.mask]), expected, atol=1e-6)
        with tf.GradientTape() as tape:
            result = compiled([self.cnn, self.graph, self.nodes, self.mask])
            loss = tf.reduce_sum(tf.square(result))
        gradients = tape.gradient(loss, self.layer.trainable_variables)
        self.assertTrue(all(g is not None and np.all(np.isfinite(g)) for g in gradients))
        self.assertTrue(any(float(tf.reduce_sum(tf.abs(g))) > 0 for g in gradients))
        # A different chunk size must preserve the same mathematical operation.
        unchunked = SpatialClientAttention(units=8, chunk_size=100)
        unchunked([self.cnn, self.graph, self.nodes, self.mask])
        unchunked.set_weights(self.layer.get_weights())
        np.testing.assert_allclose(unchunked([self.cnn, self.graph, self.nodes, self.mask]), expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
