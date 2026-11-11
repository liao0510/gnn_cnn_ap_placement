"""Rollout math, fixed validation and best-checkpoint persistence."""
import contextlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
import config
import config_radio as radio
import GNN_CNN_DRL_train as train
from training_method.Network_validation import prepare_validation, evaluate_validation
from training_method.Experiment import create_run, latest_model, model_snapshot
from training_method.Network_state import CNN_CHANNELS, NODE_FEATURES


def small_network(*args):
    """Same policy interface, cheap enough to compare exact gradient semantics."""
    image = tf.keras.Input((None, None, CNN_CHANNELS))
    nodes = tf.keras.Input((None, NODE_FEATURES))
    adjacency = tf.keras.Input((None, None))
    mask = tf.keras.Input((None,))
    spatial = tf.keras.layers.Conv2D(3, 1, activation='tanh')(image)
    logits = tf.keras.layers.Conv2D(1, 1)(spatial)
    actor = tf.keras.layers.Lambda(lambda x: tf.reshape(
        tf.nn.softmax(tf.reshape(x, [tf.shape(x)[0], -1])), tf.shape(x)), name='actor')(logits)
    graph = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2])[:, None])(nodes)
    adj = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2])[:, None])(adjacency)
    masked = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)[:, None])(mask)
    value = tf.keras.layers.Dense(1)(tf.keras.layers.Concatenate()([
        tf.keras.layers.GlobalAveragePooling2D()(spatial), graph, adj, masked]))
    return tf.keras.Model([image, nodes, adjacency, mask], [actor, value])


def trajectories(shapes=((4, 5), (4, 5))):
    rng = np.random.default_rng(5)
    result = []
    for j, (height, width) in enumerate(shapes):
        episode = []
        for i in range(2):
            state = {'map': rng.normal(size=(height, width, CNN_CHANNELS)).astype(np.float32),
                     'nodes': rng.normal(size=(3, NODE_FEATURES)).astype(np.float32),
                     'adjacency': np.eye(3, dtype=np.float32), 'mask': np.ones(3, np.float32)}
            available = np.ones((height, width), bool)
            available[0, 0] = False
            episode.append((state, (i + 1, 1), float(j + i + 1), available))
        result.append(episode)
    return result


class RolloutTests(unittest.TestCase):
    def test_deployment_summary_uses_episode_reward_sums_and_final_metrics(self):
        def episode(episode_id, rewards, coverages, satisfactions):
            trajectory = [(None, None, reward, None) for reward in rewards]
            records = [(cov, {}, {'mean_satisfaction': sat}, 1.)
                       for cov, sat in zip(coverages, satisfactions)]
            return episode_id, trajectory, records

        first = episode(1, [1., 2.], [.2, .8], [.1, .4])
        second = episode(2, [-1., 2., 4.], [.1, .5, .9], [.2, .6, 1.])
        result = train.summarize_training_rollout([first, second])
        self.assertEqual(result['mean_episode_reward'], 4.)  # mean(3, 5), not mean per action.
        self.assertAlmostEqual(result['coverage_ratio'], .85)
        self.assertAlmostEqual(result['throughput_satisfaction'], .7)
        partial = train.summarize_training_rollout([second])
        self.assertEqual(partial['mean_episode_reward'], 5.)
        self.assertEqual(partial['coverage_ratio'], .9)
        with self.assertRaises(ValueError):
            train.summarize_training_rollout([])

    def trainer(self, size=2):
        with patch.object(train, 'build_network', small_network), patch.multiple(
                config, PRINT_MODEL_SUMMARY=False, TRAIN_MICROBATCH_SIZE=size,
                PPO_EPOCHS=1, PPO_MINIBATCH_SIZE=40, PPO_SHUFFLE_MINIBATCHES=False,
                PPO_ENTROPY_COEF=0., GAE_LAMBDA=1.):
            return train.APTrainer(gamma=.5)

    def test_microbatches_match_single_batch_update_and_returns_reset(self):
        full, micro = self.trainer(10), self.trainer(3)
        micro.model.set_weights(full.model.get_weights())
        # SGD exposes the actual accumulated gradient without Adam epsilon effects.
        full.optimizer = tf.keras.optimizers.SGD(.01)
        micro.optimizer = tf.keras.optimizers.SGD(.01)
        rollout = trajectories()
        full_rows = full.train_rollout(rollout)
        micro_rows = micro.train_rollout(rollout)
        self.assertEqual(int(micro.optimizer.iterations.numpy()), 1)
        self.assertEqual(micro.last_update_metrics['microbatches'], 2)
        flat = [row for episode in micro_rows for row in episode]
        np.testing.assert_allclose([row['return_target'] for row in flat], [2, 2, 3.5, 3])
        normalized = [row['actor_advantage'] for row in flat]
        self.assertAlmostEqual(float(np.mean(normalized)), 0., places=6)
        self.assertAlmostEqual(float(np.std(normalized)), 1., places=6)
        for a, b in zip(full.model.get_weights(), micro.model.get_weights()):
            np.testing.assert_allclose(a, b, atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose([r['loss'] for ep in full_rows for r in ep],
                                   [r['loss'] for r in flat], atol=2e-6)

    def test_different_map_shapes_and_optional_normalization(self):
        trainer = self.trainer(4)
        trainer.normalize_advantages = False
        rows = trainer.train_rollout(trajectories(((4, 5), (6, 7))))
        self.assertEqual(trainer.last_update_metrics['microbatches'], 2)
        self.assertEqual(int(trainer.optimizer.iterations.numpy()), 1)
        for row in [row for episode in rows for row in episode]:
            self.assertEqual(row['advantage'], row['actor_advantage'])
            self.assertTrue(all(np.isfinite(value) for value in row.values()))

    def test_invalid_mask_cannot_update_weights(self):
        trainer = self.trainer(1)
        rollout = trajectories()
        # Even after a valid first microbatch, a later invalid action must not apply gradients.
        state, _, reward, mask = rollout[1][0]
        rollout[1][0] = (state, (0, 0), reward, mask)
        weights = trainer.model.get_weights()
        with self.assertRaisesRegex(ValueError, 'excluded'):
            trainer.train_rollout(rollout)
        self.assertEqual(int(trainer.optimizer.iterations.numpy()), 0)
        for before, after in zip(weights, trainer.model.get_weights()):
            np.testing.assert_array_equal(before, after)


def make_maps(root, count):
    root.mkdir()
    for index in range(count):
        wall = np.zeros((8, 9), dtype=int)
        wall[0, :] = 10
        forbidden = np.zeros_like(wall)
        forbidden[-1, -1] = 1
        (root / f'map_{index}.json').write_text(json.dumps(
            {'wall_map': wall.tolist(), 'forbidden': forbidden.tolist()}))
    return sorted(root.glob('*.json'))


class ValidationTests(unittest.TestCase):
    def settings(self):
        return dict(config.snapshot(), N_CLIENT=2, AP_AMOUNT=2, MAX_NODES=4,
                    VALIDATION_EPISODES=4, RSSI_RAY_STEPS=8, RSSI_ROW_CHUNK=4)

    def test_held_out_split_is_reproducible_and_does_not_consume_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            files = make_maps(Path(directory) / 'maps', 10)
            before_np, before_python = np.random.get_state(), random.getstate()
            training, manifest = prepare_validation(files, self.settings())
            self.assertEqual(len(training), 8)
            self.assertEqual(len(manifest['validation_maps']), 2)
            self.assertFalse(set(map(str, training)) & set(manifest['validation_maps']))
            self.assertEqual(prepare_validation(files, self.settings()), (training, manifest))
            np.testing.assert_array_equal(before_np[1], np.random.get_state()[1])
            self.assertEqual(before_np[2:], np.random.get_state()[2:])
            self.assertEqual(before_python, random.getstate())
            for scenario in manifest['scenarios']:
                self.assertEqual(len(set(map(tuple, scenario['clients']))), 2)
                self.assertTrue(all(y != 0 and (x, y) != (8, 7) for x, y in scenario['clients']))

    def test_explicit_validation_overlap_is_rejected_and_single_map_is_labeled(self):
        with tempfile.TemporaryDirectory() as directory:
            files = make_maps(Path(directory) / 'maps', 1)
            settings = self.settings()
            _, manifest = prepare_validation(files, settings)
            self.assertEqual(manifest['scope'], 'same_map_fixed_clients_only')
            settings['VALIDATION_MAP_DIR'] = str(files[0].parent)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                prepare_validation(files, settings)
            independent = make_maps(Path(directory) / 'validation', 1)
            settings['VALIDATION_MAP_DIR'] = str(independent[0].parent)
            training, manifest = prepare_validation(files, settings)
            self.assertEqual(training, files)
            self.assertEqual(manifest['scope'], 'held_out_maps')

    def test_real_validation_repeats_without_mutating_model_or_training_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            files = make_maps(Path(directory) / 'maps', 1)
            settings = self.settings()
            _, manifest = prepare_validation(files, settings)
            trainer = RolloutTests().trainer()
            weights = trainer.model.get_weights()
            before_np, before_python = np.random.get_state(), random.getstate()
            first = evaluate_validation(trainer.policy_model, manifest, settings, radio.snapshot())
            second = evaluate_validation(trainer.policy_model, manifest, settings, radio.snapshot())
            self.assertEqual(first, second)
            metrics = first[0]
            self.assertAlmostEqual(metrics['score'], .2 * metrics['coverage'] +
                                   .7 * metrics['mean_satisfaction'] + .1 * metrics['mesh_connectivity'])
            for before, after in zip(weights, trainer.model.get_weights()):
                np.testing.assert_array_equal(before, after)
            np.testing.assert_array_equal(before_np[1], np.random.get_state()[1])
            self.assertEqual(before_np[2:], np.random.get_state()[2:])
            self.assertEqual(before_python, random.getstate())

    def test_partial_rollout_best_checkpoint_and_final_checkpoint_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = make_maps(root / 'maps', 1)
            scores, saved_weights = iter([.4, .8, .2]), []

            def controlled_validation(policy, *args):
                # Improvement replaces the saved best; the worse final model cannot.
                score = next(scores)
                saved_weights.append(policy.get_weights())
                metrics = dict(coverage=score, mean_satisfaction=score, mesh_connectivity=score,
                               demand_met_ratio=score, total_throughput_mbps=score * 100,
                               mesh_connected=0., score=score)
                return metrics, []

            updates = dict(N_CLIENT=2, AP_AMOUNT=2, MAX_NODES=4, RANDOM_SEED=42,
                           ROLLOUT_EPISODES=2, TRAIN_MICROBATCH_SIZE=3,
                           PPO_EPOCHS=2, PPO_MINIBATCH_SIZE=3,
                           CHECKPOINT_INTERVAL=3, VALIDATION_INTERVAL=2, VALIDATION_EPISODES=2,
                           PRINT_MODEL_SUMMARY=False, SHOW_TRAINING_PLOTS=False,
                           RSSI_RAY_STEPS=8, RSSI_ROW_CHUNK=4)
            output = io.StringIO()
            with patch.multiple(config, **updates), patch.object(train, 'evaluate_validation', controlled_validation), \
                    patch.object(train, 'build_network', small_network), contextlib.redirect_stdout(output):
                history = train.run_training(files[0].parent, episodes=5, model_dir=root / 'models')
            best_path = latest_model(root / 'models')
            self.assertEqual(best_path.name, 'best_model.keras')
            run = best_path.parent
            updates = json.loads((run / 'update_history.json').read_text())
            self.assertEqual([row['episodes'] for row in updates], [2, 2, 1])
            self.assertEqual([row['transitions'] for row in updates], [4, 4, 2])
            self.assertEqual([row['optimizer_updates'] for row in updates], [4, 4, 2])
            self.assertTrue(all('explained_variance' in row for row in updates))
            minibatches = json.loads((run / 'ppo_minibatch_history.json').read_text())
            self.assertTrue(all('explained_variance' in row for row in minibatches))
            self.assertTrue(any(row['explained_variance'] is None for row in minibatches))
            for update in updates:
                episode_ids = range(update['first_episode'], update['last_episode'] + 1)
                final_indices = [max(i for i, value in enumerate(history['episode']) if value == ep)
                                 for ep in episode_ids]
                expected_reward = np.mean([sum(reward for ep_id, reward in zip(history['episode'], history['rewards'])
                                               if ep_id == ep) for ep in episode_ids])
                self.assertAlmostEqual(update['mean_episode_reward'], expected_reward)
                self.assertAlmostEqual(update['coverage_ratio'], np.mean([history['coverage'][i] for i in final_indices]))
                self.assertAlmostEqual(update['throughput_satisfaction'],
                                       np.mean([history['mean_satisfaction'][i] for i in final_indices]))
                for row in (row for row in minibatches if row['rollout'] == update['rollout']):
                    for key in ('mean_episode_reward', 'coverage_ratio', 'throughput_satisfaction'):
                        self.assertEqual(row[key], update[key])
                    self.assertEqual(row['policy_loss'], row['actor_loss'])
                    self.assertEqual(row['value_loss'], row['critic_loss'])
                    self.assertAlmostEqual(row['loss'], row['policy_loss'] + .5 * row['value_loss']
                                           - config.PPO_ENTROPY_COEF * row['entropy'], places=5)
                self.assertAlmostEqual(update['mean_loss'], update['policy_loss'] + .5 * update['value_loss']
                                       - config.PPO_ENTROPY_COEF * update['entropy'], places=5)
            np.testing.assert_array_equal(history['optimizer_update'], [4] * 4 + [8] * 4 + [10] * 2)
            validation = json.loads((run / 'validation_history.json').read_text())
            self.assertEqual([row['episode'] for row in validation], [2, 4, 5])
            self.assertEqual(model_snapshot(best_path)['best_validation']['episode'], 4)
            self.assertIn('Best validation episode: 4', output.getvalue())
            self.assertIn(' EV=', output.getvalue())
            self.assertIn(str(best_path), output.getvalue())
            self.assertTrue((run / 'validation_history.png').is_file())
            self.assertTrue((run / 'checkpoints' / 'episode_000004.keras').is_file())
            loaded = small_network()
            loaded.load_weights(str(best_path))
            for actual, expected in zip(loaded.get_weights(), saved_weights[1]):
                np.testing.assert_array_equal(actual, expected)
            loaded.load_weights(str(run / 'model.keras'))
            for actual, expected in zip(loaded.get_weights(), saved_weights[-1]):
                np.testing.assert_array_equal(actual, expected)
            self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(saved_weights[1], saved_weights[-1])))

    def test_latest_run_prefers_best_without_selecting_incomplete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = create_run(root, 'old', 'model.keras', config.snapshot(), {})
            new = create_run(root, 'new', 'model.keras', config.snapshot(), {})
            interrupted = create_run(root, 'unfinished', 'model.keras', config.snapshot(), {})
            for index, run in enumerate((old, new, interrupted)):
                (run / 'best_model.keras').touch()
                metadata = json.loads((run / 'config.json').read_text())
                metadata['artifacts']['best_model'] = 'best_model.keras'
                (run / 'config.json').write_text(json.dumps(metadata))
                if run != interrupted:
                    final = run / 'model.keras'
                    final.touch()
                    os.utime(final, (100 + index, 100 + index))
            os.utime(old / 'best_model.keras', (999, 999))
            self.assertEqual(latest_model(root), new / 'best_model.keras')


if __name__ == '__main__':
    unittest.main()
