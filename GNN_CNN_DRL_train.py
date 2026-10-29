import os, json, random
import config
import config_radio as radio
import numpy as np
import tensorflow as tf
import method.Cal_average_gpu as Cal_rssi
from method.Cal_SNR import as_numpy
from training_method.GNN_CNN_Network import build_network
from training_method.Network_policy import build_policy_model, masked_log_probabilities, placement_mask
from training_method.Network_environment import run_network_environment
from training_method.Network_state import build_state, CNN_CHANNELS, NODE_FEATURES
from training_method.Network_reward import (calculate_reward, coverage_score, mesh_rssi_map,
                                            new_ap_connected_to_mesh)
from method.Graph_visualize import plot_training_history
from training_method.Experiment import create_run, project_path, save_model


class APTrainer:
    def __init__(self, lr=None, gamma=None):
        lr=config.LR if lr is None else lr
        self.model=build_network(CNN_CHANNELS,NODE_FEATURES)
        self.policy_model=build_policy_model(self.model)
        if config.PRINT_MODEL_SUMMARY:
            self.model.summary(expand_nested=True)
        self.optimizer=tf.keras.optimizers.Adam(lr)
        self.gamma=config.GAMMA if gamma is None else gamma
        self.critic_loss_weight=config.CRITIC_LOSS_WEIGHT
        self.gradient_clip_norm=config.GRADIENT_CLIP_NORM
        self.reward_settings=config.reward_settings()

    def cal_reward(self, coverage, prev_coverage, allocation_results, new_ap_connected,
                   prev_allocation_results):
        return calculate_reward(coverage, prev_coverage, allocation_results, new_ap_connected,
                                prev_allocation_results,self.reward_settings)

    def select_action(self, state, action_mask):
        """Sample from the same masked policy used by the actor loss."""
        inputs=[state[k][None,...] for k in ('map','nodes','adjacency','mask')]
        logits,_=self.policy_model(inputs,training=False)
        log_probs=masked_log_probabilities(logits,action_mask[None,...])
        flat=tf.reshape(log_probs,[1,-1])
        idx=int(tf.random.categorical(flat,1)[0,0].numpy())
        width=action_mask.shape[1]
        return (idx%width,idx//width)

    def train_episode(self, trajectory):
        """Average all placement losses and apply one update per episode.

        trajectory contains (state_before_action, (x, y), reward, action_mask)
        tuples. Each mask is saved before placement, matching action sampling.
        Targets are observed discounted returns, with no value bootstrapping.
        """
        if not trajectory:
            raise ValueError('Cannot train on an empty episode')
        returns = np.empty(len(trajectory), dtype=np.float32)
        accumulated = 0.0
        for i in range(len(trajectory) - 1, -1, -1):
            accumulated = float(trajectory[i][2]) + self.gamma * accumulated
            returns[i] = accumulated
        inputs = [tf.convert_to_tensor(np.stack([step[0][key] for step in trajectory]),
                                       tf.float32)
                  for key in ('map', 'nodes', 'adjacency', 'mask')]
        indices = tf.constant([[i, action[1], action[0], 0]
                               for i, (_, action, _, _) in enumerate(trajectory)], tf.int32)
        action_masks = tf.convert_to_tensor(np.stack([step[3] for step in trajectory]),
                                             tf.bool)
        if not bool(tf.reduce_all(tf.gather_nd(action_masks, indices[:, :3])).numpy()):
            raise ValueError('Trajectory contains an action excluded by its placement mask')
        targets = tf.convert_to_tensor(returns, tf.float32)
        with tf.GradientTape() as tape:
            # Keep BatchNormalization in inference mode in BOTH rollout and
            # update. Gradients still train its gamma/beta and other weights;
            # using episode batch statistics here would change the policy.
            logits, values = self.policy_model(inputs, training=False)
            log_probs = masked_log_probabilities(logits, action_masks)
            values = tf.reshape(values, [-1])
            advantages = targets - values
            selected_log_probs = tf.gather_nd(log_probs, indices)
            actor_losses = -selected_log_probs * tf.stop_gradient(advantages)
            critic_losses = tf.square(advantages)
            total_losses = actor_losses + self.critic_loss_weight * critic_losses
            total = tf.reduce_mean(total_losses)
            tf.debugging.assert_all_finite(total, 'Placement loss must be finite')
        grads = tape.gradient(total, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.gradient_clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # Keep per-placement history aligned; its mean equals the update loss.
        metrics = {'loss': total_losses.numpy(), 'actor_loss': actor_losses.numpy(),
                   'critic_loss': critic_losses.numpy(), 'value': values.numpy(),
                   'advantage': advantages.numpy(), 'return_target': returns}
        return [{key: float(values[i]) for key, values in metrics.items()}
                for i in range(len(trajectory))]


def random_valid_position(walls,forbidden,occupied_positions=()):
    ids=np.flatnonzero(placement_mask(walls,forbidden,occupied_positions))
    idx=np.random.choice(ids)
    W=np.asarray(walls).shape[1]
    return (int(idx%W),int(idx//W))
from pathlib import Path
def run_training(map_dir=None, episodes=None, model_path=None, model_dir=None):
    """Train the AP-only mesh/OFDMA environment into a fresh experiment folder."""
    settings, radio_settings = config.snapshot(), radio.snapshot()
    map_dir = project_path(settings['MAP_DIR'] if map_dir is None else map_dir)
    episodes = settings['EPISODES'] if episodes is None else episodes
    for key, minimum in (('AP_AMOUNT', 1), ('N_CLIENT', 0), ('MAX_NODES', 1)):
        if not isinstance(settings[key], int) or settings[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if not isinstance(episodes, int) or episodes < 1:
        raise ValueError('EPISODES must be a positive integer')
    if settings['MAX_NODES'] < settings['N_CLIENT'] + settings['AP_AMOUNT']:
        raise ValueError('MAX_NODES must fit APs and clients')
    if not np.isfinite(settings['CLIENT_DEMAND']) or settings['CLIENT_DEMAND'] < 0:
        raise ValueError('CLIENT_DEMAND must be finite and nonnegative')
    if settings['CHECKPOINT_INTERVAL'] < 0:
        raise ValueError('CHECKPOINT_INTERVAL must be nonnegative')
    if model_path is not None and model_dir is not None:
        raise ValueError('Specify model_dir or a model_path template, not both')
    model_filename = settings['MODEL_FILENAME']
    if model_path is not None:
        template = project_path(model_path)
        model_dir, model_filename = template.parent, template.name
    model_dir = project_path(settings['MODEL_DIR'] if model_dir is None else model_dir)
    files = sorted(map_dir.glob('*.json'))
    if not files:
        raise ValueError(f'No floorplan JSON files in {map_dir}')
    channel_cfg = radio.channel_config(radio_settings)
    settings.update(MAP_DIR=str(map_dir), EPISODES=episodes,
                    MODEL_DIR=str(model_dir), MODEL_FILENAME=model_filename)
    run_dir = create_run(model_dir, settings['RUN_PREFIX'], model_filename, settings, radio_settings)
    print(f'Training run: {run_dir}')
    if settings['RANDOM_SEED'] is not None:
        random.seed(settings['RANDOM_SEED'])
        np.random.seed(settings['RANDOM_SEED'])
        tf.random.set_seed(settings['RANDOM_SEED'])
    trainer = APTrainer(lr=settings['LR'], gamma=settings['GAMMA'])
    keys = ('episode', 'total_loss', 'actor_loss', 'critic_loss', 'rewards', 'coverage',
            'coverage_gain', 'coverage_reward', 'throughput_reward', 'connectivity', 'mesh_connectivity',
            'mean_satisfaction', 'demand_met_ratio', 'total_throughput_mbps',
            'value', 'advantage', 'return_target')
    history = {key: [] for key in keys}
    for episode in range(episodes):
        path = random.choice(files)
        with path.open(encoding='utf-8') as stream:
            floorplan = json.load(stream)
        walls = np.asarray(floorplan['wall_map'])
        forbidden = np.asarray(floorplan.get('forbidden', np.zeros_like(walls)))
        if walls.ndim != 2 or forbidden.shape != walls.shape:
            raise ValueError('wall_map and forbidden must be matching 2-D maps')
        if np.count_nonzero((walls == 0) & (forbidden == 0)) < settings['N_CLIENT'] + settings['AP_AMOUNT']:
            raise ValueError('Not enough distinct free positions for APs and clients')
        coverage_mask = walls == 0
        clients = []
        for _ in range(settings['N_CLIENT']):
            clients.append(random_valid_position(walls, forbidden, clients))
        demands = [settings['CLIENT_DEMAND']] * len(clients)
        aps, maps = [], []
        simulator = Cal_rssi.Cal_db_gpu(walls=walls,
                    frequency_ghz=radio_settings['ACCESS_FREQUENCY_GHZ'],
                    tx_power=radio_settings['TX_POWER_DBM'],
                    grid_size=radio_settings['GRID_SIZE_METERS'],
                    ray_steps=settings['RSSI_RAY_STEPS'], row_chunk=settings['RSSI_ROW_CHUNK'])

        def evaluate():
            return run_network_environment(aps, clients, demands, maps, channel_cfg,
                                           mac_efficiency=radio_settings['MAC_EFFICIENCY'])

        net = evaluate()
        total_rssi = mesh_rssi_map(maps, net['in_mesh'], walls.shape)
        coverage = coverage_score(total_rssi, radio_settings['COVERAGE_THRESHOLD_DBM'], coverage_mask)
        state = build_state(walls, forbidden, aps, clients, demands, total_rssi,
                            net, 0, settings['AP_AMOUNT'], settings['MAX_NODES'])
        trajectory, records = [], []
        
        result_path = str(Path(path).parts[-2] + "\\" + Path(path).name)
        print(f"map:{result_path}")
        for step in range(settings['AP_AMOUNT']):
            action_mask = placement_mask(walls, forbidden, aps + clients)
            position = trainer.select_action(state, action_mask)
            previous_coverage, previous_allocations = coverage, net['allocation_results']
            previous_in_mesh = net['in_mesh']
            aps.append(position)
            signal, _ = simulator.calculate_signal_map(position)
            maps.append(as_numpy(signal))
            net = evaluate()
            new_ap_connected = new_ap_connected_to_mesh(net['mesh_adjacency'], previous_in_mesh)
            total_rssi = mesh_rssi_map(maps, net['in_mesh'], walls.shape)
            coverage = coverage_score(total_rssi, radio_settings['COVERAGE_THRESHOLD_DBM'], coverage_mask)
            reward, parts = trainer.cal_reward(coverage, previous_coverage, net['allocation_results'],
                                               new_ap_connected, previous_allocations)
            next_state = build_state(walls, forbidden, aps, clients, demands, total_rssi,
                                     net, step + 1, settings['AP_AMOUNT'], settings['MAX_NODES'])
            trajectory.append((state, position, reward, action_mask))
            records.append((coverage, parts, dict(net['network_summary']), net['mesh_connectivity']))
            state = next_state
            
            print(f'EP {episode + 1} AP {step + 1}/{settings["AP_AMOUNT"]} pos={position} '
                  f'edges={len(net["mesh_edges"])} mesh={net["mesh_connectivity"]:.1%} '
                  f'coverage={coverage:.1%} satisfaction={net["network_summary"]["mean_satisfaction"]:.1%} '
                  f'conn={parts["connectivity"]:.0f} reward={reward:.4f}')
        losses = trainer.train_episode(trajectory)
        for loss, transition, record in zip(losses, trajectory, records):
            cov, parts, summary, mesh_connectivity = record
            metrics = dict(episode=episode + 1, total_loss=loss['loss'],
                           actor_loss=loss['actor_loss'], critic_loss=loss['critic_loss'],
                           rewards=transition[2], coverage=cov, coverage_gain=parts['coverage_gain'],
                           coverage_reward=parts['coverage'], throughput_reward=parts['throughput'],
                           connectivity=parts['connectivity'], mesh_connectivity=mesh_connectivity,
                           value=loss['value'],
                           advantage=loss['advantage'], return_target=loss['return_target'],
                           mean_satisfaction=summary['mean_satisfaction'],
                           demand_met_ratio=summary['demand_met_ratio'],
                           total_throughput_mbps=summary['total_throughput_mbps'])
            for key, value in metrics.items():
                history[key].append(value)
        print(f'EP {episode + 1} mean_loss={np.mean([row["loss"] for row in losses]):.4f}')
        interval = settings['CHECKPOINT_INTERVAL']
        if interval and (episode + 1) % interval == 0 and episode + 1 < episodes:
            save_model(trainer.model, run_dir / 'checkpoints' / f'episode_{episode + 1:06d}.keras')
    np.savez(run_dir / 'training_history.npz', **{key: np.asarray(value) for key, value in history.items()})
    save_model(trainer.model, run_dir / model_filename)
    plot_training_history(history, save_path=str(run_dir / 'training_history.png'),
                          show=settings['SHOW_TRAINING_PLOTS'])
    return history


if __name__ == '__main__':
    run_training()
