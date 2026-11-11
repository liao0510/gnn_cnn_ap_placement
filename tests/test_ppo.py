"""PPO objective gradients, fixed behavior policy, GAE and optimizer boundaries."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
import config
import GNN_CNN_DRL_train as train
from training_method.Network_policy import masked_log_probabilities
from training_method.PPO import (PPOTransition, episode_gae, clipped_actor_loss,
                                masked_entropy, validate_ppo_settings, explained_variance)
from test_training_rollout import small_network, trajectories


def frozen_rollout(trainer, episodes):
    frozen = []
    for episode in episodes:
        captured = []
        for index, (state, action, reward, mask) in enumerate(episode):
            logits, value = trainer.policy_model([state[key][None] for key in ('map', 'nodes', 'adjacency', 'mask')],
                                                 training=False)
            logs = masked_log_probabilities(logits, mask[None]).numpy()
            x, y = action
            captured.append(PPOTransition(state, action, reward, mask, float(logs[0, y, x, 0]),
                                          float(value[0, 0]), index == len(episode) - 1))
        frozen.append(captured)
    return frozen


class PPOMathTests(unittest.TestCase):
    def test_explained_variance_perfect_constant_and_wrong_predictions(self):
        self.assertEqual(explained_variance([1., 2., 3.], [1., 2., 3.]), 1.)
        self.assertEqual(explained_variance([1., 2., 3.], [0., 0., 0.]), 0.)
        self.assertEqual(explained_variance([1., 2., 3.], [3., 2., 1.]), -3.)
        # EV measures variance rather than fixed bias (unlike R-squared).
        self.assertEqual(explained_variance([1., 2., 3.], [6., 7., 8.]), 1.)

    def test_explained_variance_undefined_targets_are_json_null(self):
        for target, value in (([2., 2.], [1., 3.]), ([1.], [2.])):
            ev = explained_variance(target, value)
            self.assertIsNone(ev)
            self.assertEqual(json.dumps({'explained_variance': ev}, allow_nan=False),
                             '{"explained_variance": null}')
        with self.assertRaises(ValueError):
            explained_variance([1., 2.], [float('nan'), 1.])

    def test_gae_terminal_and_lambda_returns(self):
        # delta=[1 + .9*.4 - .5, 2 - .4] = [.86, 1.6]
        adv, targets = episode_gae([1., 2.], [.5, .4], [False, True], .9, .8)
        np.testing.assert_allclose(adv, [2.012, 1.6], rtol=1e-6)
        np.testing.assert_allclose(targets, [2.512, 2.], rtol=1e-6)
        _, mc = episode_gae([1., 2.], [.5, .4], [False, True], .9, 1.)
        np.testing.assert_allclose(mc, [2.8, 2.], rtol=1e-6)
        with self.assertRaisesRegex(ValueError, 'terminal'):
            episode_gae([1., 2.], [.5, .4], [False, False], .9, .95)

    def test_clipping_has_correct_sign_and_saturates_only_improving_direction(self):
        logs = tf.Variable(np.log([1.5, .5, .5, 1.5]).astype(np.float32))
        advantages = tf.constant([1., -1., 1., -1.])
        with tf.GradientTape() as tape:
            losses = clipped_actor_loss(logs, advantages, .2)
            total = tf.reduce_sum(losses)
        np.testing.assert_allclose(losses.numpy(), [-1.2, .8, -.5, 1.5], atol=1e-6)
        np.testing.assert_allclose(tape.gradient(total, logs).numpy(), [0., 0., -.5, 1.5], atol=1e-6)

    def test_masked_entropy_has_no_invalid_cell_nan_or_gradient(self):
        logits = tf.Variable(np.zeros((1, 2, 2, 1), np.float32))
        masks = tf.constant([[[True, False], [False, True]]])
        with tf.GradientTape() as tape:
            logs = masked_log_probabilities(logits, masks)
            entropy = tf.reduce_sum(masked_entropy(logs, masks))
        grad = tape.gradient(entropy, logits).numpy()
        self.assertAlmostEqual(float(entropy), np.log(2.), places=6)
        self.assertTrue(np.all(np.isfinite(grad)))
        self.assertEqual(float(grad[0, 0, 1, 0]), 0.)

    def test_invalid_settings_are_rejected_before_creating_a_run(self):
        for key, value in (('PPO_EPOCHS', 0), ('PPO_MINIBATCH_SIZE', 0), ('PPO_CLIP_EPSILON', 1.),
                           ('GAE_LAMBDA', 1.1), ('PPO_TARGET_KL', -1.), ('PPO_ENTROPY_COEF', float('nan')),
                           ('PPO_VALUE_CLIP_EPSILON', 0.), ('PPO_ADAM_EPSILON', 0.)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_ppo_settings(dict(config.snapshot(), **{key: value}))


class PPOUpdateTests(unittest.TestCase):
    def trainer(self, **settings):
        changes = dict(PRINT_MODEL_SUMMARY=False, PPO_MINIBATCH_SIZE=2, PPO_EPOCHS=3,
                       TRAIN_MICROBATCH_SIZE=1, PPO_SHUFFLE_MINIBATCHES=False)
        changes.update(settings)
        with patch.multiple(config, **changes), patch.object(train, 'build_network', small_network):
            return train.APTrainer(lr=.005)

    def test_multi_epoch_updates_keep_old_policy_and_gae_fixed(self):
        trainer = self.trainer()
        rollout = frozen_rollout(trainer, trajectories())
        original_logs = np.asarray([step.old_log_probability for ep in rollout for step in ep])
        original_values = np.asarray([step.old_value for ep in rollout for step in ep])
        rows = [row for episode in trainer.train_rollout(rollout) for row in episode]
        self.assertEqual(int(trainer.optimizer.iterations.numpy()), 6)  # ceil(4/2)*3
        self.assertEqual(trainer.last_update_metrics['epochs_completed'], 3)
        self.assertEqual(trainer.last_update_metrics['microbatches'], 12)
        np.testing.assert_array_equal([row['old_log_probability'] for row in rows], original_logs)
        np.testing.assert_array_equal([row['value'] for row in rows], original_values)
        # At collection/current equality, the first mini-batch KL must be zero.
        self.assertAlmostEqual(trainer.last_minibatch_metrics[0]['approx_kl'], 0., places=6)
        self.assertGreater(trainer.last_update_metrics['approx_kl'], 0.)
        self.assertTrue(any(abs(row['probability_ratio'] - 1.) > 1e-5 for row in rows))
        for episode, episode_rows in zip(rollout, [rows[:2], rows[2:]]):
            adv, targets = episode_gae([s.reward for s in episode], [s.old_value for s in episode],
                                       [s.done for s in episode], config.GAMMA, config.GAE_LAMBDA)
            np.testing.assert_allclose([r['return_target'] for r in episode_rows], targets)
            np.testing.assert_allclose([r['advantage'] for r in episode_rows], adv)

    def test_ev_aggregates_all_microbatches_and_uses_current_critic(self):
        trainer = self.trainer(PPO_MINIBATCH_SIZE=3, PPO_EPOCHS=2, TRAIN_MICROBATCH_SIZE=1)
        rollout = frozen_rollout(trainer, trajectories())
        captured, original = [], trainer._loss_terms

        def capture(steps, ids, old_logs, old_values, advantages, targets):
            terms = original(steps, ids, old_logs, old_values, advantages, targets)
            captured.append((np.asarray(ids).copy(), terms['predicted_value'].numpy().copy(), targets.copy()))
            return terms

        with patch.object(trainer, '_loss_terms', capture):
            trainer.train_rollout(rollout)
        offset = 0
        for row in trainer.last_minibatch_metrics:
            chunks = captured[offset:offset + row['transitions']]
            ids = np.concatenate([chunk[0] for chunk in chunks])
            values = np.concatenate([chunk[1] for chunk in chunks])
            expected = explained_variance(chunks[0][2][ids], values)
            if expected is None:
                self.assertIsNone(row['explained_variance'])
            else:
                self.assertAlmostEqual(row['explained_variance'], expected)
            offset += row['transitions']
        # Every individual microbatch contains one sample (undefined EV), but
        # the three-sample mini-batch must have a valid EV computed jointly.
        self.assertIsNotNone(trainer.last_minibatch_metrics[0]['explained_variance'])
        self.assertIsNone(trainer.last_minibatch_metrics[1]['explained_variance'])
        final_predictions = np.concatenate([chunk[1] for chunk in captured[offset:]])
        self.assertAlmostEqual(trainer.last_update_metrics['explained_variance'],
                               explained_variance(captured[-1][2], final_predictions))
        self.assertTrue(np.any(np.abs(final_predictions - np.asarray([
            step.old_value for episode in rollout for step in episode])) > 1e-6))
        json.dumps(trainer.last_minibatch_metrics, allow_nan=False)

    def test_160_transitions_have_20_updates_with_requested_defaults(self):
        trainer = self.trainer(PPO_MINIBATCH_SIZE=40, PPO_EPOCHS=5, TRAIN_MICROBATCH_SIZE=8)
        short_episode = trajectories()[0]
        rollout = frozen_rollout(trainer, [short_episode * 2 + short_episode[:1] for _ in range(32)])
        trainer.train_rollout(rollout)
        self.assertEqual(trainer.last_update_metrics['optimizer_updates'], 20)
        self.assertEqual(trainer.last_update_metrics['epochs_completed'], 5)
        self.assertEqual(len(trainer.last_minibatch_metrics), 20)
        self.assertTrue(all(row['transitions'] == 40 for row in trainer.last_minibatch_metrics))

    def test_kl_early_stop_discards_candidate_update_and_preserves_old_statistics(self):
        trainer = self.trainer(PPO_TARGET_KL=1e-9, PPO_EPOCHS=10)
        rollout = frozen_rollout(trainer, trajectories())
        trainer.train_rollout(rollout)
        self.assertTrue(trainer.last_update_metrics['early_stopped'])
        self.assertLess(trainer.last_update_metrics['optimizer_updates'], 20)
        self.assertGreaterEqual(trainer.last_update_metrics['optimizer_updates'], 1)

    def test_optional_value_clipping_uses_fixed_old_values(self):
        trainer = self.trainer(PPO_VALUE_CLIP_EPSILON=.1)
        rollout = frozen_rollout(trainer, trajectories())
        steps = [s for ep in rollout for s in ep]
        old_logs = np.asarray([s.old_log_probability for s in steps], np.float32)
        old_values = np.asarray([s.old_value for s in steps], np.float32)
        # Move only the critic bias while keeping its collection values frozen.
        critic = trainer.model.outputs[1]._keras_history[0]
        critic.bias.assign_add(tf.ones_like(critic.bias))
        targets = old_values + 1.
        terms = trainer._loss_terms(steps, np.arange(4), old_logs, old_values, np.ones(4, np.float32), targets)
        np.testing.assert_allclose(terms['critic_loss'].numpy(), [.81] * 4, atol=2e-6)

    def test_shuffled_epochs_visit_every_transition_once_per_epoch(self):
        trainer = self.trainer(PPO_SHUFFLE_MINIBATCHES=True, PPO_MINIBATCH_SIZE=3)
        rollout = frozen_rollout(trainer, trajectories())
        visited, original = [], trainer._loss_terms

        def capture(steps, ids, *args):
            visited.extend(int(i) for i in ids)
            return original(steps, ids, *args)

        with patch.object(trainer, '_loss_terms', capture):
            trainer.train_rollout(rollout)
        # Last four visits are the final diagnostic evaluation, not an update.
        for start in range(0, 12, 4):
            self.assertEqual(sorted(visited[start:start + 4]), [0, 1, 2, 3])
        self.assertEqual(len(visited), 16)


if __name__ == '__main__':
    unittest.main()
