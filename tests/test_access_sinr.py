"""Independent power budgets, AP-addition regressions and SINR state semantics."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import config
import config_radio as radio
from config_radio import ChannelConfig
from method.Cal_SNR import build_ap_mesh, inspect_client_links, calculate_noise_dbm
from training_method.Network_environment import run_network_environment
from training_method.Network_state import build_state, NetworkNormalizer
from training_method.Network_reward import calculate_reward, coverage_score, mesh_rssi_map
from training_method.Experiment import validate_model_snapshot, latest_model


class AccessSINRTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ChannelConfig()

    @staticmethod
    def maps(values, width=8):
        return [np.full((1, width), dbm, dtype=float) for dbm in values]

    def inspect(self, values, cfg=None, members=None):
        return inspect_client_links([(7, 0)], [(i, 0) for i in range(len(values))],
                                    self.maps(values), [True]*len(values) if members is None else members,
                                    self.cfg if cfg is None else cfg)

    def net(self, values, clients=((7, 0),)):
        return run_network_environment([(i, 0) for i in range(len(values))], list(clients),
                                       [100.] * len(clients), self.maps(values), self.cfg)

    def test_one_AP_reduces_exactly_to_reference_snr_without_self_interference(self):
        rows, association, selected = self.inspect([-60.])
        link = rows[0]['links'][0]
        self.assertAlmostEqual(selected[0], 27.96910013008056)
        self.assertEqual(association, [0])
        self.assertEqual(link['interference_mw'], 0.)
        self.assertIsNone(link['interference_dbm'])
        self.assertEqual(link['interfering_aps'], [])
        json.dumps(rows, allow_nan=False)

    def test_three_candidates_match_hand_calculated_sinr(self):
        rows, association, selected = self.inspect([-60., -65., -70.],
            ChannelConfig(client_sinr_threshold_db=-20.))
        links = rows[0]['links']
        np.testing.assert_allclose([r['sinr_db'] for r in links],
            [3.790066426497088, -5.420224330193789, -11.198574046326764], atol=1e-10)
        self.assertEqual(association, [0])
        self.assertAlmostEqual(links[0]['interference_mw'], 10**(-6.5) + 1e-7)
        self.assertAlmostEqual(links[0]['interference_dbm'], -63.80668951933905)
        self.assertAlmostEqual(selected[0], links[0]['sinr_db'])
        self.assertTrue(all('snr_db' not in r for r in links))

    def test_high_snr_can_fail_access_sinr_threshold_without_breaking_mesh(self):
        net = self.net([-60., -65., -70.])
        self.assertEqual(net['connected_AP'], [-1])
        self.assertEqual(net['connected_AP_SINR'], [None])
        self.assertEqual(net['allocation_results'][0]['throughput_mbps'], 0.)
        self.assertIsNone(net['allocation_results'][0]['mcs'])
        self.assertEqual(len(net['mesh_edges']), 3)
        self.assertEqual(net['mesh_connectivity'], 1.)

    def test_adding_AP_recomputes_existing_sinr_mcs_throughput_and_residual(self):
        one, two, three = [self.net(values) for values in
                            ([-60.], [-60., -70.], [-60., -70., -65.])]
        self.assertEqual(one['connected_AP'], [0])
        self.assertEqual(two['connected_AP'], [0])
        self.assertAlmostEqual(two['connected_AP_SINR'][0], 9.931224935390683)
        self.assertEqual(one['allocation_results'][0]['mcs'], 7)
        self.assertEqual(two['allocation_results'][0]['mcs'], 1)
        self.assertEqual(one['allocation_results'][0]['throughput_mbps'], 100.)
        expected = .7 * 980 * 2 * .5 / 13.6
        self.assertAlmostEqual(two['allocation_results'][0]['throughput_mbps'], expected)
        self.assertEqual(three['connected_AP'], [-1])
        self.assertEqual(three['allocation_results'][0]['throughput_mbps'], 0.)
        _, parts = calculate_reward(1., 1., two['allocation_results'], True,
                                    one['allocation_results'])
        self.assertLess(parts['throughput_gain'], 0.)
        walls = np.zeros((1, 8))
        state = build_state(walls, walls, [(0, 0), (1, 0)], [(7, 0)], [100.],
                            np.full(walls.shape, -60.), two, 2, 5, 8)
        self.assertAlmostEqual(float(state['nodes'][2, 5]),
                               (two['connected_AP_SINR'][0] + 40.) / 80., places=6)
        self.assertAlmostEqual(float(state['nodes'][2, 14]), 1-expected/100, places=6)

    def test_new_strong_AP_can_become_serving_AP_after_reassociation(self):
        before = self.net([-70.])
        after = self.net([-70., -50.])
        self.assertEqual(before['connected_AP'], [0])
        self.assertEqual(after['connected_AP'], [1])
        self.assertEqual(after['client_links'][0]['links'][1]['interfering_aps'], [0])

    def test_nonserving_mesh_AP_and_AP_without_clients_still_interfere(self):
        rows, association, _ = self.inspect([-60., -70.], members=[True, False])
        self.assertEqual(association, [0])
        self.assertEqual(rows[0]['links'][0]['interfering_aps'], [1])
        self.assertAlmostEqual(rows[0]['links'][0]['sinr_db'], 9.931224935390683)
        net = self.net([-60., -70.])
        self.assertEqual(net['ap_resources'][1]['associated_clients'], [])
        self.assertGreater(net['allocation_results'][0]['interference_mw'], 0.)

    def test_interference_is_measured_at_client_and_is_receiver_specific(self):
        maps = self.maps([-40., -40.])
        maps[0][0, 6:8] = [-60., -80.]
        maps[1][0, 6:8] = [-80., -60.]
        net = run_network_environment([(0, 0), (1, 0)], [(6, 0), (7, 0)],
                                      [100., 100.], maps, self.cfg)
        self.assertEqual(net['connected_AP'], [0, 1])
        for row in net['allocation_results']:
            self.assertAlmostEqual(row['interference_mw'], 1e-8, delta=1e-18)
        self.assertEqual(len(net['mesh_edges']), 1)

    def test_mcs_and_equal_ru_capacity_use_sinr_without_double_division(self):
        net = self.net([-60., -70., -75., -80.], clients=((6, 0), (7, 0)))
        self.assertEqual(net['connected_AP'], [0, 0])
        for row in net['allocation_results']:
            self.assertAlmostEqual(row['sinr_db'], 8.439994040570298)
            self.assertEqual(row['mcs'], 1)
            self.assertEqual(row['ru_tones'], 484)
            self.assertEqual(row['time_fraction'], 1.)
            self.assertAlmostEqual(row['throughput_mbps'], .7*468/13.6)

    def test_other_AP_powers_are_summed_not_averaged(self):
        rows, _, _ = self.inspect([-50., -60., -60.])
        self.assertAlmostEqual(rows[0]['links'][0]['interference_mw'], 2e-6, delta=1e-17)
        self.assertAlmostEqual(rows[0]['links'][0]['interference_dbm'], -56.98970004336019)

    def test_weak_interference_is_not_lost_by_total_minus_desired_cancellation(self):
        rows, _, _ = self.inspect([-20., -250.])
        self.assertAlmostEqual(rows[0]['links'][0]['interference_mw'], 1e-25, delta=1e-35)
        self.assertAlmostEqual(rows[0]['links'][0]['interference_dbm'], -250.)

    def test_nan_link_is_ineligible_and_no_AP_has_finite_json(self):
        rows, association, _ = self.inspect([float('nan'), -60.])
        self.assertEqual(association, [1])
        self.assertIsNone(rows[0]['links'][0]['sinr_db'])
        empty = self.net([])
        self.assertEqual(empty['connected_AP'], [-1])
        self.assertEqual(empty['client_links'][0]['links'], [])
        json.dumps({'links': empty['client_links'], 'allocations': empty['allocation_results']},
                   allow_nan=False)

    def test_normalization_and_new_snapshot_keys(self):
        np.testing.assert_allclose(NetworkNormalizer.sinr([-80., -40., 0., 40., 80.]),
                                   [0., 0., .5, 1., 1.])
        self.assertEqual(radio.channel_config().client_sinr_threshold_db, 4.)
        self.assertEqual(radio.snapshot()['ACCESS_INTERFERENCE_MODEL'], 'full_band_all_aps_always_on')
        self.assertNotIn('CLIENT_SNR_THRESHOLD_DB', radio.snapshot())

    def test_old_SNR_weights_are_rejected_even_when_architecture_matches(self):
        old = dict(environment_version='ap_mesh_snr_ofdma_v1',
                   model_architecture=config.MODEL_ARCHITECTURE, artifacts={'model': 'model.keras'})
        with self.assertRaisesRegex(ValueError, 'SINR'):
            validate_model_snapshot(old)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/'old'; folder.mkdir()
            (folder/'config.json').write_text(json.dumps(old))
            (folder/'model.keras').touch()
            with self.assertRaisesRegex(FileNotFoundError, 'SINR'):
                latest_model(directory)


class RawPropagationTests(unittest.TestCase):
    def test_raw_weak_signal_preserved_and_legacy_mesh_and_coverage_unchanged(self):
        from method.Cal_average_gpu import Cal_db_gpu
        walls = np.zeros((5, 7), dtype=float)
        walls[:, 3] = 200.
        aps = [(0, 2), (6, 2)]
        simulator = Cal_db_gpu(walls=walls, row_chunk=2)
        raw = [simulator.calculate_signal_map(p)[0].get() for p in aps]
        self.assertLess(raw[0][2, 6], -100.)
        clipped = [np.clip(m, -100., -20.) for m in raw]
        cfg = ChannelConfig()
        expected = build_ap_mesh(aps, clipped, cfg)
        actual = run_network_environment(aps, [(1, 2)], [100.], raw, cfg)
        self.assertEqual(actual['mesh_edges'], expected['mesh_edges'])
        self.assertEqual(actual['in_mesh'], expected['in_mesh'])
        self.assertEqual(coverage_score(mesh_rssi_map(raw, actual['in_mesh']), -60., walls==0),
                         coverage_score(mesh_rssi_map(clipped, expected['in_mesh']), -60., walls==0))


if __name__ == '__main__':
    unittest.main()
