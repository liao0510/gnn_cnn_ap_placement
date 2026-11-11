"""Paired comparisons, known statistics, real CUDA/model round-trip and plots."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
import config
import config_radio as radio
import GNN_CNN_DRL_predict_batch as batch
import GNN_CNN_DRL_predict as single
from training_method.Experiment import create_run, save_model
from training_method.GNN_CNN_Network import build_network
from training_method.Prediction_analysis import ANALYSIS_METRICS, final_metrics, summarize_trials


class AnalysisTests(unittest.TestCase):
    def test_paired_difference_ci_preserves_pairing(self):
        trials = []
        for value in (0.1, 0.3, 0.5):
            trials.append({'gnn_cnn': {'metrics': {key: value + 0.1 for key, _, _ in ANALYSIS_METRICS}},
                           'random': {'metrics': {key: value for key, _, _ in ANALYSIS_METRICS}}})
        summary = summarize_trials(trials, bootstrap_samples=200)
        stats = summary['paired_difference_gnn_minus_random']['coverage']
        self.assertAlmostEqual(stats['mean'], 0.1)
        np.testing.assert_allclose(stats['mean_ci95'], [0.1, 0.1])
        self.assertEqual(stats['win_rate'], 1)
        self.assertEqual(stats['tie_rate'], 0)
        singleton = summarize_trials(trials[:1], bootstrap_samples=100)
        self.assertIsNone(singleton['methods']['random']['coverage']['mean_ci95'])
        self.assertIsNone(singleton['methods']['random']['coverage']['std'])

    def test_metrics_include_failed_clients_and_exclude_zero_demand_from_fairness(self):
        network = {'allocation_results': [
            {'throughput_mbps': 50, 'satisfaction_ratio': 0.5, 'demand_mbps': 100},
            {'throughput_mbps': 0, 'satisfaction_ratio': 0, 'demand_mbps': 100},
            {'throughput_mbps': 0, 'satisfaction_ratio': 1, 'demand_mbps': 0}],
            'network_summary': {'mean_satisfaction': 0.5, 'demand_met_ratio': 1/3,
                                'total_throughput_mbps': 50, 'associated_clients': 1},
            'mesh_connectivity': 2/3, 'mesh_connected': False}
        metrics = final_metrics(0.4, network)
        self.assertEqual(metrics['client_throughput_p05_mbps'], 0)
        self.assertEqual(metrics['satisfaction_fairness'], 0.5)
        self.assertEqual(metrics['associated_client_ratio'], 1/3)
        self.assertEqual(metrics['mesh_connectivity'], 2/3)
        self.assertEqual(metrics['mesh_connected'], 0)
        network['allocation_results'] = []
        self.assertIsNone(final_metrics(0.4, network)['mean_satisfaction'])
        self.assertIsNone(final_metrics(0.4, network)['satisfaction_fairness'])


class BatchWorkflowTests(unittest.TestCase):
    def test_real_batch_pairing_reproducibility_map_switching_and_single_predict_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = root / 'maps'
            maps.mkdir()
            walls = np.zeros((8, 9), dtype=int)
            walls[0, :] = 1
            forbidden = np.zeros_like(walls)
            forbidden[1, 1] = 1
            for name in ('a', 'b'):
                (maps / f'{name}.json').write_text(json.dumps(
                    {'wall_map': walls.tolist(), 'forbidden': forbidden.tolist()}))
            settings = dict(config.snapshot(), RSSI_RAY_STEPS=8, RSSI_ROW_CHUNK=4)
            run = create_run(root / 'models', 'batch_test', 'model.keras', settings, radio.snapshot())
            tf.random.set_seed(7)
            save_model(build_network(), run / 'model.keras')
            updates = dict(NUM_PREDICTIONS=3, RANDOM_MAP_EACH_RUN=False,
                MAP_PATH=str(maps / 'a.json'), MAP_DIRECTORY=str(maps), MODEL_PATH=str(run / 'model.keras'),
                AP_COUNT=3, CLIENT_COUNT=2, CLIENT_DEMAND_MBPS=100., CLIENT_DEMANDS=None,
                RANDOM_SEED=42, BOOTSTRAP_SAMPLES=100, OUTPUT_DIRECTORY=str(root / 'results'), SHOW_PLOTS=False)
            with patch.multiple(batch, **updates), patch.object(radio, 'ACCESS_BANDWIDTH_MHZ', 20):
                result = batch.run_batch_predictions()
                with patch.object(batch, 'BOOTSTRAP_SAMPLES', 200):
                    repeated = batch.run_batch_predictions()
                with patch.multiple(batch, RANDOM_MAP_EACH_RUN=True, NUM_PREDICTIONS=2):
                    switched = batch.run_batch_predictions()
            self.assertNotEqual(result['output_dir'], repeated['output_dir'])
            self.assertNotEqual(switched['trials'][0]['map_path'], switched['trials'][1]['map_path'])
            self.assertEqual(len({json.dumps(t['client_positions']) for t in result['trials']}), 3)
            for trial, again in zip(result['trials'], repeated['trials']):
                self.assertEqual(trial['client_positions'], again['client_positions'])
                for method in ('gnn_cnn', 'random'):
                    self.assertEqual(trial[method], again[method])
                    positions = trial['client_positions'] + trial[method]['ap_positions']
                    self.assertEqual(len(set(map(tuple, positions))), 5)
                    self.assertTrue(all(walls[y, x] == forbidden[y, x] == 0 for x, y in positions))
                    self.assertEqual(len(trial[method]['allocation_results']), 2)
            output = Path(result['output_dir'])
            header = (output / 'client_metrics.csv').read_text(encoding='utf-8-sig').splitlines()[0].split(',')
            self.assertIn('sinr_db', header)
            self.assertNotIn('snr_db', header)
            self.assertEqual(len((output / 'trials.jsonl').read_text().splitlines()), 3)
            for name in ('trial_metrics.csv', 'client_metrics.csv', 'summary.csv', 'summary.json',
                         'evaluation_config.json'):
                self.assertGreater((output / name).stat().st_size, 0)
            for name in ('per_trial_metrics', 'metric_distributions', 'summary_metrics',
                         'paired_differences', 'client_distributions'):
                self.assertTrue((output / (name + '.png')).read_bytes().startswith(b'\x89PNG'))
                self.assertTrue((output / (name + '.pdf')).read_bytes().startswith(b'%PDF'))
            saved_config = json.loads((output / 'evaluation_config.json').read_text(encoding='utf-8'))
            self.assertEqual(saved_config['radio']['ACCESS_BANDWIDTH_MHZ'], 80)
            # Same weights/map/clients must produce exactly the original greedy deployment.
            with patch.multiple(config, PREDICT_AP_COUNT=3, PREDICT_CLIENT_COUNT=2,
                PREDICT_CLIENT_POSITIONS=result['trials'][0]['client_positions'],
                PREDICT_CLIENT_DEMAND_MBPS=100., PREDICT_CLIENT_DEMANDS=None,
                PREDICT_MAP_PATH=str(maps / 'a.json'), PREDICT_OUTPUT_DIR=str(root / 'single'),
                SHOW_PREDICTION_PLOTS=False, PRINT_PREDICTION_MODEL_SUMMARY=False,
                CLIENT_DENSITY_SIGMAS_METERS=(-1., -1.), CLIENT_DENSITY_REFERENCE_MBPS=-1.), \
                    patch.object(radio, 'GRID_SIZE_METERS', -1.):
                # Even invalid live feature/radio settings cannot replace the
                # saved physical scales used to train this checkpoint.
                report = single.run_prediction(run / 'model.keras')
            self.assertEqual(report['ap_positions'], result['trials'][0]['gnn_cnn']['ap_positions'])
            self.assertEqual(final_metrics(report['coverage'], report['network']),
                             result['trials'][0]['gnn_cnn']['metrics'])

    def test_invalid_settings_fail_before_loading_model(self):
        with patch.object(batch, 'NUM_PREDICTIONS', 0):
            with self.assertRaises(ValueError):
                batch.run_batch_predictions()


if __name__ == '__main__':
    unittest.main()
