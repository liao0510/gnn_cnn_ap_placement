"""Global context, exact grid dimensions, gradients and checkpoint compatibility."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
import config
from training_method.GNN_CNN_Network import build_network
from training_method.Network_policy import build_policy_model
from training_method.Experiment import create_run, latest_model, validate_model_snapshot
from training_method.Network_state import CNN_CHANNELS, NODE_FEATURES


def inputs(height, width):
    rng = np.random.default_rng(9)
    nodes = rng.random((1, 5, NODE_FEATURES), dtype=np.float32)
    nodes[..., :2] = 0
    nodes[:, 0, 0] = 1
    nodes[:, 1:, 1] = 1
    return [rng.random((1, height, width, CNN_CHANNELS), dtype=np.float32),
            nodes,
            np.eye(5, dtype=np.float32)[None], np.ones((1, 5), dtype=np.float32)]


class SpatialCnnTests(unittest.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_original_action_grid_for_odd_rectangular_and_tiny_maps(self):
        model = build_network()
        for height, width in ((1, 1), (7, 9), (33, 47), (100, 101)):
            actor, critic = model(inputs(height, width), training=False)
            self.assertEqual(tuple(actor.shape), (1, height, width, 1))
            self.assertEqual(tuple(critic.shape), (1, 1))
            self.assertTrue(np.all(np.isfinite(actor)))
            self.assertTrue(np.all(np.isfinite(critic)))
            self.assertAlmostEqual(float(tf.reduce_sum(actor)), 1.0, places=5)

    def test_far_client_and_precise_local_detail_have_gradient_paths(self):
        model = build_network()
        encoder = tf.keras.Model(model.inputs[0], model.get_layer('cnn_features').output)
        # Positive averaging kernels establish active paths independently of
        # random ReLU gating; this verifies connectivity, not learned quality.
        for layer in encoder.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                kernel, bias = layer.get_weights()
                kernel.fill(1.0 / np.prod(kernel.shape[:3]))
                bias.fill(0.01)
                layer.set_weights([kernel, bias])
        raster = tf.Variable(np.full((1, 65, 81, CNN_CHANNELS), 0.1, np.float32))
        with tf.GradientTape() as tape:
            feature = encoder(raster, training=False)[0, 2, 2, 0]
        gradient = tape.gradient(feature, raster).numpy()
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertGreater(gradient[0, 2, 2, 0], 0)   # wall at this exact area
        self.assertGreater(gradient[0, 61, 77, 3], 0) # client far outside old 17x17 window

    def test_single_cell_client_positions_remain_distinguishable(self):
        model = build_network()
        encoder = tf.keras.Model(model.inputs[0], model.get_layer('cnn_features').output)
        tf.random.set_seed(19)
        a = np.zeros((1, 33, 35, CNN_CHANNELS), np.float32)
        b = a.copy()
        a[0, 16, 16, 3] = 0.25
        b[0, 16, 17, 3] = 0.25
        features_a, features_b = encoder(a, training=False), encoder(b, training=False)
        self.assertGreater(float(tf.reduce_max(tf.abs(features_a - features_b))), 1e-6)

    def test_checkpoint_and_full_model_roundtrip(self):
        model = build_network()
        sample = inputs(13, 19)
        expected = build_policy_model(model)(sample, training=False)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'model.keras')
            model.save(path)
            fresh = build_network()
            fresh.load_weights(path)
            restored = tf.keras.models.load_model(path, compile=False)
            for network in (fresh, restored):
                actual = build_policy_model(network)(sample, training=False)
                for target, value in zip(expected, actual):
                    np.testing.assert_allclose(value, target, atol=1e-6)


class ArchitectureVersionTests(unittest.TestCase):
    def test_old_and_other_architectures_are_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, architecture in (('legacy', None), ('v2', 'gnn_cnn_multiscale_v2'),
                                       ('v3', 'gnn_cnn_client_attention_v3'),
                                       ('other', 'different_network')):
                folder = root / name
                folder.mkdir()
                metadata = {'environment_version': config.ENVIRONMENT_VERSION,
                            'artifacts': {'model': 'model.keras'}}
                if architecture is not None:
                    metadata['model_architecture'] = architecture
                (folder / 'config.json').write_text(json.dumps(metadata))
                (folder / 'model.keras').touch()
                with self.assertRaisesRegex(ValueError, 'retrain'):
                    validate_model_snapshot(metadata)
            with self.assertRaisesRegex(FileNotFoundError, config.MODEL_ARCHITECTURE):
                latest_model(root)
            current = create_run(root, 'new', 'model.keras', config.snapshot(), {})
            (current / 'model.keras').touch()
            self.assertEqual(latest_model(root), current / 'model.keras')
            # A newer legacy checkpoint must not displace the compatible model.
            (root / 'legacy' / 'model.keras').touch()
            self.assertEqual(latest_model(root), current / 'model.keras')


if __name__ == '__main__':
    unittest.main()
