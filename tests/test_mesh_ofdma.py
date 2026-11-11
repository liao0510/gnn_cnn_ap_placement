"""Behavioral checks for SNR mesh, strongest access SINR and HE RU budgets."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import config
from config_radio import ChannelConfig
from method.Cal_SNR import build_ap_mesh, calculate_noise_dbm
from method.Cal_link_capacity import calculate_link_capacity, RU_LAYOUTS, RU_DATA_SUBCARRIERS, FULL_CHANNEL_RU
from method.Cal_throughput_allocation import allocate_throughput, txop_allocation
from training_method.Network_environment import run_network_environment
from training_method.Network_state import build_state, CNN_CHANNELS, NODE_FEATURES
from training_method.Network_reward import (mesh_rssi_map, coverage_score, calculate_reward,
                                            new_ap_connected_to_mesh)
from training_method.Experiment import latest_model, create_run


class MeshTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ChannelConfig()
        self.noise = calculate_noise_dbm(self.cfg)

    def maps(self, rows):
        return [np.asarray([row], dtype=float) + self.noise for row in rows]

    def test_triangle_preserves_cycle_and_exact_threshold(self):
        positions = [(0, 0), (1, 0), (2, 0)]
        mesh = build_ap_mesh(positions, self.maps([[50, 4, 4], [4, 50, 4], [4, 4, 50]]), self.cfg)
        self.assertEqual(len(mesh['mesh_edges']), 3)
        self.assertEqual(mesh['in_mesh'], [True] * 3)
        np.testing.assert_array_equal(mesh['mesh_adjacency'], np.ones((3, 3)) - np.eye(3))

    def test_bidirectional_snr_without_extra_rssi_gate(self):
        cfg = ChannelConfig(mesh_snr_threshold_db=-3)
        mesh = build_ap_mesh([(0, 0), (1, 0)], self.maps([[40, -2], [-2, 40]]), cfg)
        self.assertEqual(len(mesh['mesh_edges']), 1)
        mesh = build_ap_mesh([(0, 0), (1, 0)], self.maps([[40, 20], [3, 40]]), self.cfg)
        self.assertEqual(mesh['mesh_edges'], [])

    def test_largest_component_is_not_anchored_to_ap_zero(self):
        mesh = build_ap_mesh([(0, 0), (1, 0), (2, 0)],
                             self.maps([[50, -10, -10], [-10, 50, 10], [-10, 10, 50]]), self.cfg)
        self.assertEqual(mesh['in_mesh'], [False, True, True])
        self.assertAlmostEqual(mesh['mesh_connectivity'], 2/3)
        self.assertFalse(mesh['mesh_connected'])

    def test_strongest_sinr_and_low_sinr_unassociated(self):
        maps = self.maps([[50, 10, 15, 3], [10, 50, 25, 2]])
        net = run_network_environment([(0, 0), (1, 0)], [(2, 0), (3, 0)], [100, 100], maps, self.cfg)
        self.assertEqual(net['connected_AP'], [1, -1])
        self.assertAlmostEqual(net['connected_AP_SINR'][0], 25 - 10*np.log10(1 + 10**1.5))
        self.assertEqual(net['allocation_results'][1]['throughput_mbps'], 0)

    def test_empty_mesh_has_zero_coverage_and_valid_state(self):
        walls = np.zeros((2, 3))
        net = run_network_environment([], [(1, 0)], [100], [], self.cfg)
        total = mesh_rssi_map([], [], walls.shape)
        self.assertEqual(coverage_score(total), 0)
        self.assertEqual(net['connected_AP'], [-1])
        state = build_state(walls, walls, [], [(1, 0)], [100], total, net, 0, 3, 4)
        self.assertEqual(state['map'].shape, (2, 3, CNN_CHANNELS))
        self.assertEqual(state['nodes'].shape, (4, NODE_FEATURES))
        self.assertEqual(state['nodes'][0, 10], 0)
        self.assertTrue(all(np.all(np.isfinite(value)) for value in state.values()))

    def test_gnn_has_cycle_edges_and_identical_ap_types(self):
        positions = [(0, 0), (1, 0), (2, 0)]
        net = run_network_environment(positions, [], [], self.maps([[50]*3]*3), self.cfg)
        walls = np.zeros((1, 3))
        state = build_state(walls, walls, positions, [], [], np.full((1, 3), -40), net, 3, 3, 3)
        self.assertTrue(np.all(state['adjacency'] > 0))
        np.testing.assert_array_equal(state['nodes'][:, 0], [1, 1, 1])

    def test_disconnected_ap_strong_access_is_excluded(self):
        maps = self.maps([[50, 10, -10, 10], [10, 50, -10, 20], [-10, -10, 50, 50]])
        net = run_network_environment([(0, 0), (1, 0), (2, 0)], [(3, 0)], [100], maps, self.cfg)
        self.assertEqual(net['connected_AP'], [-1])
        self.assertFalse(net['client_links'][0]['links'][2]['link_valid'])
        self.assertIn(2, net['client_links'][0]['links'][1]['interfering_aps'])

    def test_satisfaction_drop_is_penalized(self):
        _, parts = calculate_reward(0.5, 0.5, [{'satisfaction_ratio': 0.5}], True,
                                     [{'satisfaction_ratio': 1.0}])
        self.assertLess(parts['throughput'], 0)

    def connection_reward(self, rows):
        positions = [(i, 0) for i in range(len(rows))]
        maps = self.maps(rows)
        previous = build_ap_mesh(positions[:-1], maps[:-1], self.cfg)
        current = build_ap_mesh(positions, maps, self.cfg)
        connected = new_ap_connected_to_mesh(current['mesh_adjacency'], previous['in_mesh'])
        reward, parts = calculate_reward(0.5, 0.5, [], connected, [])
        self.assertEqual(reward, config.CONNECTIVITY_REWARD_WEIGHT * parts['connectivity'])
        return parts['connectivity'], current

    def test_first_ap_starts_mesh_and_earns_full_connection_bonus(self):
        conn, mesh = self.connection_reward([[50]])
        self.assertEqual(conn, 1)
        self.assertEqual(mesh['mesh_connectivity'], 1)

    def test_later_disconnected_ap_has_no_connection_bonus(self):
        conn, _ = self.connection_reward([[50, -10], [-10, 50]])
        self.assertEqual(conn, 0)

    def test_five_successful_placements_average_one_including_seed(self):
        connections = [self.connection_reward([[50.] * count for _ in range(count)])[0]
                       for count in range(1, 6)]
        self.assertEqual(connections, [1.] * 5)
        self.assertEqual(float(np.mean(connections)), 1.)

    def test_joining_serving_mesh_gets_full_bonus_despite_isolated_ap(self):
        conn, mesh = self.connection_reward([[50, 10, -10, 10], [10, 50, -10, -10],
                                             [-10, -10, 50, -10], [10, -10, -10, 50]])
        self.assertEqual(conn, 1)
        self.assertEqual(mesh['mesh_connectivity'], 0.75)

    def test_joining_isolated_group_does_not_earn_bonus_when_mesh_switches(self):
        # Before placement AP0/AP1 is the serving mesh; AP2/AP3 is tied.
        # AP4 joins only AP2/AP3, making that group the new largest component.
        conn, mesh = self.connection_reward([[50, 10, -10, -10, -10],
                                             [10, 50, -10, -10, -10],
                                             [-10, -10, 50, 10, 10],
                                             [-10, -10, 10, 50, -10],
                                             [-10, -10, 10, -10, 50]])
        self.assertTrue(mesh['in_mesh'][-1])
        self.assertEqual(conn, 0)

    def test_bridging_serving_mesh_and_isolated_ap_gets_bonus(self):
        conn, mesh = self.connection_reward([[50, 10, -10, 10], [10, 50, -10, -10],
                                             [-10, -10, 50, 10], [10, -10, 10, 50]])
        self.assertEqual(conn, 1)
        self.assertTrue(mesh['mesh_connected'])

    def test_connection_reward_rejects_old_fractional_ratio(self):
        with self.assertRaises(ValueError):
            calculate_reward(0.5, 0.5, [], 0.75, [])


class OfdmaTests(unittest.TestCase):
    def allocate(self, n, demands=None, sinrs=None, bandwidth=80):
        return allocate_throughput([10000.] * n if demands is None else demands,
                                  [0] * n, [40.] * n if sinrs is None else sinrs,
                                  1, ChannelConfig(bandwidth_mhz=bandwidth), mac_efficiency=0.7)

    def test_single_client_full_ru(self):
        rows, resources = self.allocate(1)
        self.assertEqual(rows[0]['ru_tones'], 996)
        self.assertAlmostEqual(rows[0]['throughput_mbps'], 980 * 10 * (5/6) / 13.6 * 0.7)
        self.assertEqual(resources[0]['concurrent_clients'], 1)

    def test_two_clients_simultaneously_use_484_rus(self):
        rows, resources = self.allocate(2, sinrs=[40, 20])
        self.assertEqual([row['ru_tones'] for row in rows], [484, 484])
        self.assertEqual([row['time_fraction'] for row in rows], [1, 1])
        self.assertGreater(rows[0]['throughput_mbps'], rows[1]['throughput_mbps'])
        self.assertAlmostEqual(rows[0]['throughput_mbps'], 468 * 10 * (5/6) / 13.6 * 0.7)
        self.assertEqual(len(txop_allocation(resources[0], 0)), 2)

    def test_layout_boundaries_and_resource_conservation(self):
        for bandwidth, layouts in RU_LAYOUTS.items():
            for n in (1, 2, 3, 4, 5, 9, 17, 37, 38, 75):
                rows, resources = self.allocate(n, bandwidth=bandwidth)
                resource = resources[0]
                tones, concurrency = resource['ru_tones'], resource['concurrent_clients']
                self.assertLessEqual(concurrency, layouts[tones])
                self.assertLessEqual(concurrency * RU_DATA_SUBCARRIERS[tones],
                                     RU_DATA_SUBCARRIERS[FULL_CHANNEL_RU[bandwidth]])
                full = calculate_link_capacity(40, 0.7, bandwidth_mhz=bandwidth)['link_capacity_mbps']
                self.assertLessEqual(sum(row['throughput_mbps'] for row in rows), full + 1e-8)

    def test_round_robin_is_fair_and_has_no_slot_collisions(self):
        rows, resources = self.allocate(40)
        counts = [0] * 40
        for txop in range(40):
            schedule = txop_allocation(resources[0], txop)
            self.assertEqual(len(schedule), 37)
            self.assertEqual(len({row['client'] for row in schedule}), 37)
            self.assertEqual(len({row['ru_index'] for row in schedule}), 37)
            for row in schedule:
                counts[row['client']] += 1
        self.assertEqual(counts, [37] * 40)
        self.assertAlmostEqual(rows[0]['time_fraction'], 37/40)

    def test_demand_caps_and_zero_demand_uses_no_ru(self):
        rows, _ = self.allocate(3, demands=[10, 10000, 0])
        self.assertEqual(rows[0]['throughput_mbps'], 10)
        self.assertEqual(rows[0]['ru_tones'], 484)
        self.assertEqual(rows[2]['ru_tones'], 0)
        self.assertTrue(rows[2]['demand_met'])

    def test_independent_per_ap_ru_budgets(self):
        rows, resources = allocate_throughput([10000, 10000], [0, 1], [40, 40], 2, ChannelConfig())
        self.assertEqual([row['ru_tones'] for row in rows], [996, 996])
        self.assertEqual(len(resources), 2)

    def test_rate_threshold_and_invalid_inputs(self):
        self.assertEqual(calculate_link_capacity(3.99)['link_capacity_mbps'], 0)
        self.assertEqual(calculate_link_capacity(4)['mcs'], 0)
        with self.assertRaises(ValueError):
            calculate_link_capacity(40, ru_tones=996, bandwidth_mhz=20)
        with self.assertRaises(ValueError):
            self.allocate(1, demands=[-1])


class ExperimentTests(unittest.TestCase):
    def test_only_compatible_complete_models_are_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / 'old'
            old.mkdir()
            (old / 'config.json').write_text(json.dumps({'artifacts': {'model': 'old.keras'}}))
            (old / 'old.keras').touch()
            with self.assertRaises(FileNotFoundError):
                latest_model(root)
            run = create_run(root, 'mesh', 'model.keras', config.snapshot(), {})
            with self.assertRaises(FileNotFoundError):
                latest_model(root)
            (run / 'model.keras').touch()
            self.assertEqual(latest_model(root), run / 'model.keras')


if __name__ == '__main__':
    unittest.main()
