"""Masked categorical PPO-Clip with terminal GAE and memory-bounded updates.

Reference: Schulman et al., https://arxiv.org/abs/1707.06347 (Eq. 7, Algorithm 1).
Rollouts contain complete episodes. Old policy statistics and GAE targets stay
fixed across every optimization epoch; only the current policy is reevaluated.
"""
from typing import NamedTuple
import numpy as np
import tensorflow as tf
from training_method.Network_policy import masked_log_probabilities


class PPOTransition(NamedTuple):
    state: dict
    action: tuple
    reward: float
    action_mask: np.ndarray
    old_log_probability: float
    old_value: float
    done: bool


def validate_ppo_settings(settings):
    for key in ('ROLLOUT_EPISODES', 'PPO_MINIBATCH_SIZE', 'PPO_EPOCHS', 'TRAIN_MICROBATCH_SIZE'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('NORMALIZE_ADVANTAGES', 'PPO_SHUFFLE_MINIBATCHES'):
        if type(settings[key]) is not bool:
            raise ValueError(f'{key} must be True or False')
    for key in ('GAMMA', 'GAE_LAMBDA'):
        if not np.isfinite(settings[key]) or not 0 <= settings[key] <= 1:
            raise ValueError(f'{key} must be finite and between 0 and 1')
    if not np.isfinite(settings['PPO_CLIP_EPSILON']) or not 0 < settings['PPO_CLIP_EPSILON'] < 1:
        raise ValueError('PPO_CLIP_EPSILON must be between 0 and 1')
    for key in ('LR', 'GRADIENT_CLIP_NORM', 'PPO_ADAM_EPSILON'):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('PPO_ENTROPY_COEF', 'CRITIC_LOSS_WEIGHT'):
        if not np.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('PPO_TARGET_KL', 'PPO_VALUE_CLIP_EPSILON'):
        if settings[key] is not None and (not np.isfinite(settings[key]) or settings[key] <= 0):
            raise ValueError(f'{key} must be None or finite and positive')
    if settings['TRAINING_ALGORITHM'] != 'ppo_clip_v1':
        raise ValueError('This trainer implements TRAINING_ALGORITHM=ppo_clip_v1')


def episode_gae(rewards, values, dones, gamma, gae_lambda):
    """GAE for one complete episode; terminal bootstrap is exactly zero."""
    rewards, values = np.asarray(rewards, np.float32), np.asarray(values, np.float32)
    dones = np.asarray(dones, bool)
    if not len(rewards) or rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError('GAE requires matching, nonempty reward/value/done arrays')
    if not dones[-1] or np.any(dones[:-1]):
        raise ValueError('Each rollout episode must end with exactly one terminal transition')
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(values)):
        raise ValueError('Rewards and old values must be finite')
    advantages, accumulated = np.empty_like(rewards), 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = 0.0 if dones[index] else float(values[index + 1])
        delta = float(rewards[index]) + gamma * next_value - float(values[index])
        accumulated = delta + gamma * gae_lambda * (not dones[index]) * accumulated
        advantages[index] = accumulated
    return advantages, advantages + values


def clipped_actor_loss(log_ratio, advantages, clip_epsilon):
    ratio = tf.exp(log_ratio)
    clipped = tf.clip_by_value(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return -tf.minimum(ratio * advantages, clipped * advantages)


def masked_entropy(log_probs, masks):
    """Avoid 0 * -inf at forbidden cells, including during backpropagation."""
    safe_logs = tf.where(masks[..., None], log_probs, tf.zeros_like(log_probs))
    return -tf.reduce_sum(tf.exp(log_probs) * safe_logs, axis=[1, 2, 3])


def explained_variance(targets, predictions):
    """1 - Var(target - value) / Var(target), computed over the whole batch.

    Use fixed, unnormalized GAE lambda-return targets and raw critic values.
    A constant target (including a one-sample batch) has undefined EV: return
    None so strict JSON stores null and the terminal can display N/A.
    """
    targets, predictions = np.asarray(targets, np.float64), np.asarray(predictions, np.float64)
    if targets.ndim != 1 or targets.shape != predictions.shape or not targets.size:
        raise ValueError('Explained variance requires matching, nonempty 1-D arrays')
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(predictions)):
        raise ValueError('Explained variance inputs must be finite')
    variance = float(np.var(targets))
    return None if variance == 0.0 else float(1.0 - np.var(targets - predictions) / variance)


class PPOUpdater:
    def __init__(self, model, policy_model, optimizer, settings):
        validate_ppo_settings(settings)
        self.model, self.policy_model, self.optimizer = model, policy_model, optimizer
        self.gamma, self.gae_lambda = settings['GAMMA'], settings['GAE_LAMBDA']
        self.critic_loss_weight = settings['CRITIC_LOSS_WEIGHT']
        self.gradient_clip_norm = settings['GRADIENT_CLIP_NORM']
        self.microbatch_size = settings['TRAIN_MICROBATCH_SIZE']
        self.normalize_advantages = settings['NORMALIZE_ADVANTAGES']
        self.minibatch_size, self.ppo_epochs = settings['PPO_MINIBATCH_SIZE'], settings['PPO_EPOCHS']
        self.clip_epsilon, self.entropy_coef = settings['PPO_CLIP_EPSILON'], settings['PPO_ENTROPY_COEF']
        self.value_clip_epsilon, self.target_kl = settings['PPO_VALUE_CLIP_EPSILON'], settings['PPO_TARGET_KL']
        self.shuffle_minibatches = settings['PPO_SHUFFLE_MINIBATCHES']
        self.rollout_count = 0
        self.last_update_metrics = None
        self.last_minibatch_metrics = []

    @staticmethod
    def _inputs(steps, ids):
        return [tf.convert_to_tensor(np.stack([steps[i][0][key] for i in ids]), tf.float32)
                for key in ('map', 'nodes', 'adjacency', 'mask')]

    def _microbatches(self, steps, ids):
        """Keep each shuffled mini-batch intact; split unequal map shapes safely."""
        start = 0
        while start < len(ids):
            end = start + 1
            shape = tuple(steps[ids[start]][0][key].shape for key in ('map', 'nodes', 'adjacency', 'mask'))
            while end < min(len(ids), start + self.microbatch_size):
                if tuple(steps[ids[end]][0][key].shape for key in ('map', 'nodes', 'adjacency', 'mask')) != shape:
                    break
                end += 1
            yield ids[start:end]
            start = end

    def _old_statistics(self, steps):
        """Validate the whole rollout before any update.

        Production transitions save statistics at action sampling. Four-field
        tuples remain supported for old diagnostic callers only: their old
        statistics are frozen once at entry, before the first PPO update.
        """
        log_probs, values, missing = np.empty(len(steps), np.float32), np.empty(len(steps), np.float32), []
        for index, step in enumerate(steps):
            if len(step) not in (4, 7):
                raise ValueError('Expected PPOTransition or legacy four-field transition')
            state, (x, y), reward, mask = step[:4]
            mask = np.asarray(mask, bool)
            if (mask.shape != state['map'].shape[:2] or not 0 <= y < mask.shape[0]
                    or not 0 <= x < mask.shape[1] or not mask[y, x]):
                raise ValueError('Trajectory contains an action excluded by its placement mask')
            if not np.isfinite(reward):
                raise ValueError('Reward must be finite')
            if len(step) == 7:
                log_probs[index], values[index] = step[4:6]
            else:
                missing.append(index)
        for ids in self._microbatches(steps, missing):
            logits, predictions = self.policy_model(self._inputs(steps, ids), training=False)
            masks = np.stack([steps[i][3] for i in ids])
            logs = masked_log_probabilities(logits, masks).numpy()
            for offset, i in enumerate(ids):
                x, y = steps[i][1]
                log_probs[i], values[i] = logs[offset, y, x, 0], float(predictions[offset, 0])
        if not np.all(np.isfinite(log_probs)) or not np.all(np.isfinite(values)):
            raise ValueError('Old log probabilities and values must be finite')
        if np.any(log_probs > 1e-6):
            raise ValueError('Old action log probabilities cannot be positive')
        return log_probs, values

    def _loss_terms(self, steps, ids, old_logs, old_values, advantages, targets):
        masks = tf.convert_to_tensor(np.stack([steps[i][3] for i in ids]), tf.bool)
        logits, values = self.policy_model(self._inputs(steps, ids), training=False)
        logs = masked_log_probabilities(logits, masks)
        indices = tf.constant([[offset, steps[i][1][1], steps[i][1][0], 0]
                               for offset, i in enumerate(ids)], tf.int32)
        selected = tf.gather_nd(logs, indices)
        log_ratio = selected - tf.convert_to_tensor(old_logs[ids])
        ratio = tf.exp(log_ratio)
        actor = clipped_actor_loss(log_ratio, tf.convert_to_tensor(advantages[ids]), self.clip_epsilon)
        predicted_values = tf.reshape(values, [-1])
        critic = tf.square(tf.convert_to_tensor(targets[ids]) - predicted_values)
        if self.value_clip_epsilon is not None:
            old = tf.convert_to_tensor(old_values[ids])
            clipped = old + tf.clip_by_value(predicted_values - old, -self.value_clip_epsilon, self.value_clip_epsilon)
            critic = tf.maximum(critic, tf.square(tf.convert_to_tensor(targets[ids]) - clipped))
        entropy = masked_entropy(logs, masks)
        total = actor + self.critic_loss_weight * critic - self.entropy_coef * entropy
        return {'loss': total, 'actor_loss': actor, 'critic_loss': critic, 'entropy': entropy,
                'approx_kl': (ratio - 1.0) - log_ratio,
                'clip_fraction': tf.cast(tf.abs(ratio - 1.0) > self.clip_epsilon, tf.float32),
                'probability_ratio': ratio, 'predicted_value': predicted_values}

    def train_episode(self, trajectory):
        return self.train_rollout([trajectory])[0]

    def train_rollout(self, trajectories):
        if not trajectories or any(not episode for episode in trajectories):
            raise ValueError('Cannot train on an empty rollout/episode')
        steps = [step for episode in trajectories for step in episode]
        count, offset = len(steps), 0
        old_logs, old_values = self._old_statistics(steps)
        advantages, targets = [], []
        for episode in trajectories:
            length = len(episode)
            dones = [bool(step[6]) if len(step) == 7 else index == length - 1
                     for index, step in enumerate(episode)]
            adv, ret = episode_gae([step[2] for step in episode], old_values[offset:offset + length],
                                   dones, self.gamma, self.gae_lambda)
            advantages.extend(adv)
            targets.extend(ret)
            offset += length
        advantages, targets = np.asarray(advantages, np.float32), np.asarray(targets, np.float32)
        actor_advantages = advantages.copy()
        if self.normalize_advantages and count > 1:
            actor_advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
        variables = self.model.trainable_variables
        initial_iteration = int(self.optimizer.iterations.numpy())
        self.last_minibatch_metrics = []
        early_stopped, epochs_completed, microbatch_count, stop_kl = False, 0, 0, None
        for epoch in range(self.ppo_epochs):
            order = np.random.permutation(count) if self.shuffle_minibatches else np.arange(count)
            for start in range(0, count, self.minibatch_size):
                ids = order[start:start + self.minibatch_size]
                gradients = [tf.zeros_like(variable) for variable in variables]
                means = {key: 0.0 for key in ('loss', 'actor_loss', 'critic_loss', 'entropy', 'approx_kl', 'clip_fraction')}
                minibatch_predictions = []
                for micro_ids in self._microbatches(steps, ids):
                    with tf.GradientTape() as tape:
                        terms = self._loss_terms(steps, micro_ids, old_logs, old_values, actor_advantages, targets)
                        # Exactly the mean gradient of this PPO mini-batch, even
                        # when its final microbatch or mini-batch is short.
                        loss = tf.reduce_sum(terms['loss']) / len(ids)
                        tf.debugging.assert_all_finite(loss, 'PPO loss must be finite')
                    grads = tape.gradient(loss, variables)
                    gradients = [total + (grad if grad is not None else tf.zeros_like(total))
                                 for total, grad in zip(gradients, grads)]
                    for key in means:
                        means[key] += float(tf.reduce_sum(terms[key]).numpy()) / len(ids)
                    minibatch_predictions.extend(terms['predicted_value'].numpy().tolist())
                    microbatch_count += 1
                if self.target_kl is not None and means['approx_kl'] > 1.5 * self.target_kl:
                    early_stopped, stop_kl = True, means['approx_kl']
                    break  # Discard this mini-batch's gradients, keep previous updates.
                for grad in gradients:
                    tf.debugging.assert_all_finite(grad, 'PPO gradient must be finite')
                clipped, norm = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)
                self.optimizer.apply_gradients(zip(clipped, variables))
                self.last_minibatch_metrics.append(dict(means,
                    policy_loss=means['actor_loss'], value_loss=means['critic_loss'],
                    explained_variance=explained_variance(targets[ids], minibatch_predictions),
                    optimizer_update=int(self.optimizer.iterations.numpy()), epoch=epoch + 1,
                    minibatch=start // self.minibatch_size + 1, transitions=len(ids),
                    gradient_norm_before_clip=float(norm.numpy()),
                    gradient_clipped=bool(float(norm.numpy()) > self.gradient_clip_norm)))
            if early_stopped:
                break
            epochs_completed += 1
        # Histories describe the final updated policy evaluated against FIXED old
        # rollout statistics. Report once per placement, in original episode order.
        final = {key: np.empty(count, np.float32) for key in (
            'loss', 'actor_loss', 'critic_loss', 'entropy', 'approx_kl', 'clip_fraction', 'probability_ratio')}
        final_predictions = np.empty(count, np.float32)
        for ids in self._microbatches(steps, np.arange(count)):
            terms = self._loss_terms(steps, ids, old_logs, old_values, actor_advantages, targets)
            for key in final:
                tf.debugging.assert_all_finite(terms[key], f'Final PPO {key} must be finite')
                final[key][ids] = terms[key].numpy()
            final_predictions[ids] = terms['predicted_value'].numpy()
        self.rollout_count += 1
        norms = [row['gradient_norm_before_clip'] for row in self.last_minibatch_metrics]
        self.last_update_metrics = {
            'rollout': self.rollout_count, 'optimizer_update': int(self.optimizer.iterations.numpy()),
            'optimizer_updates': int(self.optimizer.iterations.numpy()) - initial_iteration,
            'episodes': len(trajectories), 'transitions': count, 'microbatches': microbatch_count,
            'epochs_completed': epochs_completed, 'early_stopped': early_stopped, 'stop_approx_kl': stop_kl,
            'gradient_norm_before_clip': float(np.mean(norms)) if norms else 0.0,
            'gradient_norm_max': float(max(norms)) if norms else 0.0,
            'gradient_clipped': any(row['gradient_clipped'] for row in self.last_minibatch_metrics),
            'advantage_mean': float(advantages.mean()), 'advantage_std': float(advantages.std()),
            'mean_loss': float(final['loss'].mean()),
            'policy_loss': float(final['actor_loss'].mean()), 'value_loss': float(final['critic_loss'].mean()),
            'entropy': float(final['entropy'].mean()),
            'approx_kl': float(final['approx_kl'].mean()), 'clip_fraction': float(final['clip_fraction'].mean()),
            'explained_variance': explained_variance(targets, final_predictions)}
        final.update(value=old_values, old_log_probability=old_logs, advantage=advantages,
                     return_target=targets, actor_advantage=actor_advantages)
        rows = [{key: float(values[i]) for key, values in final.items()} for i in range(count)]
        result, offset = [], 0
        for episode in trajectories:
            result.append(rows[offset:offset + len(episode)])
            offset += len(episode)
        return result
