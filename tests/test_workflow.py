"""Real TensorFlow update, CuPy propagation, saved-model prediction and plots."""
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf

# Keep the small actor/critic smoke test independent of cuDNN availability.
# The propagation simulator still uses the installed CuPy/CUDA implementation.
tf.config.set_visible_devices([], 'GPU')
import config
import config_radio as radio
import GNN_CNN_DRL_train as train
import GNN_CNN_DRL_predict as predict
from training_method.Experiment import latest_model
from training_method.Network_state import CNN_CHANNELS, NODE_FEATURES


class WorkflowTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'gnn_drl_wmn' / 'gnn_drl_wmn').is_dir(),
                         'Optional alternative PPO project is not present in this workspace')
    def test_alternative_gnn_ppo_uses_same_mesh_ofdma_model(self):
        folder = Path(__file__).resolve().parents[1] / 'gnn_drl_wmn' / 'gnn_drl_wmn'
        with patch.object(sys, 'path', [str(folder)] + sys.path):
            from wmn_gnn_model import Config
            from training import training
            from predict import predict as predict_gnn
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = root / 'maps'
            maps.mkdir()
            (maps / 'map.json').write_text(json.dumps({'wall_map': np.zeros((8, 8)).tolist()}))
            cfg = Config(map_path=str(maps), num_aps=3, num_clients=2, candidate_count=8,
                         hidden_dim=8, episodes=1, update_every_episodes=1, ppo_epochs=1,
                         channel_row_chunk=4, channel_ray_steps=8,
                         save_weights=str(root / 'model.weights.h5'),
                         history_path=str(root / 'history.json'), prediction_path=str(root / 'prediction.json'))
            _, history = training(cfg)
            result = predict_gnn(cfg)
            self.assertEqual(len(history['reward']), 1)
            self.assertEqual(len(result['ap_positions']), 3)
            self.assertEqual(len(result['mesh_edges']), 3)
            self.assertTrue(all(row['ru_tones'] > 0 for row in result['network']['allocation_results']))

    def test_ablation_encoders_accept_new_state_shapes(self):
        from training_method.CNN_Network import build_network as cnn
        from training_method.GNN_Network import build_network as gnn
        inputs = [np.zeros((1, 4, 4, CNN_CHANNELS), np.float32),
                  np.zeros((1, 5, NODE_FEATURES), np.float32),
                  np.eye(5, dtype=np.float32)[None], np.ones((1, 5), np.float32)]
        for builder in (cnn, gnn):
            actor, critic = builder()(inputs, training=False)
            self.assertEqual(tuple(actor.shape), (1, 4, 4, 1))
            self.assertEqual(tuple(critic.shape), (1, 1))
            self.assertAlmostEqual(float(tf.reduce_sum(actor)), 1.0, places=5)

    def test_real_training_prediction_and_saved_radio_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = root / 'maps'
            maps.mkdir()
            wall = np.zeros((10, 10), dtype=int)
            wall[0, :] = 1
            map_path = maps / 'map.json'
            map_path.write_text(json.dumps({'wall_map': wall.tolist(),
                                            'forbidden': np.zeros_like(wall).tolist()}))
            updates = dict(N_CLIENT=2, AP_AMOUNT=3, MAX_NODES=5, RANDOM_SEED=42,
                           ROLLOUT_EPISODES=1, TRAIN_MICROBATCH_SIZE=2,
                           PPO_EPOCHS=2, PPO_MINIBATCH_SIZE=2,
                           VALIDATION_INTERVAL=1, VALIDATION_EPISODES=2,
                           CHECKPOINT_INTERVAL=1, SHOW_TRAINING_PLOTS=False, PRINT_MODEL_SUMMARY=False,
                           RSSI_RAY_STEPS=8, RSSI_ROW_CHUNK=4,
                           PREDICT_MAP_PATH=str(map_path), PREDICT_AP_COUNT=3,
                           PREDICT_CLIENT_COUNT=2, PREDICT_CLIENT_POSITIONS=[(1, 1), (8, 8)],
                           PREDICT_OUTPUT_DIR=str(root / 'prediction'), SHOW_PREDICTION_PLOTS=False,
                           PRINT_PREDICTION_MODEL_SUMMARY=False)
            with patch.multiple(config, **updates):
                history = train.run_training(map_dir=maps, episodes=2, model_dir=root / 'models')
                model = latest_model(root / 'models')
                self.assertEqual(len(history['episode']), 6)
                np.testing.assert_array_equal(history['connectivity'], [1] * 6)
                np.testing.assert_array_equal(history['mesh_connectivity'], [1] * 6)
                self.assertTrue(all(np.all(np.isfinite(value)) for value in history.values()))
                self.assertTrue((model.parent / 'checkpoints' / 'episode_000001.keras').is_file())
                self.assertTrue((model.parent / 'training_history.png').is_file())
                # Prediction must use the training snapshot, not edited live radio settings.
                with patch.object(radio, 'ACCESS_BANDWIDTH_MHZ', 20):
                    report = predict.run_prediction(model)
                self.assertEqual(report['radio_config']['ACCESS_BANDWIDTH_MHZ'], 80)
                self.assertEqual(len(report['ap_positions']), 3)
                self.assertEqual(len(report['network']['in_mesh']), 3)
                self.assertEqual(report['initial_coverage'], 0)
                self.assertEqual(len(report['history']), 3)
                self.assertEqual([row['reward_parts']['connectivity'] for row in report['history']],
                                 [1, 1, 1])
                self.assertEqual([row['new_ap_connected'] for row in report['history']],
                                 [True, True, True])
                self.assertEqual([row['mesh_connectivity'] for row in report['history']], [1] * 3)
                self.assertEqual(len(report['network']['mesh_edges']), 3)
                self.assertEqual(len(set(map(tuple, report['ap_positions'] + report['client_positions']))), 5)
                self.assertTrue((root / 'prediction' / 'network_maps.png').is_file())
                self.assertTrue((root / 'prediction' / 'client_details_1.png').is_file())
                self.assertTrue((root / 'prediction' / 'mesh_links_1.png').is_file())
                encoded = json.dumps(report, allow_nan=False).lower()
                for obsolete in ('gateway', 'backhaul', 'parents'):
                    self.assertNotIn(obsolete, encoded)
                self.assertIn('sinr_db', encoded)
                self.assertTrue(report['assumptions']['all_deployed_aps_always_transmit'])
                self.assertEqual(report['assumptions']['access_link_metric'], 'SINR')
                self.assertEqual(report['assumptions']['mesh_link_metric'], 'SNR')
                for row in report['network']['allocation_results']:
                    self.assertIn('sinr_db', row)
                    self.assertNotIn('snr_db', row)
                    self.assertLessEqual(row['throughput_mbps'], row['demand_mbps'])
                    self.assertLessEqual(row['time_fraction'], 1)


if __name__ == '__main__':
    unittest.main()
