import csv
import json
import random
from pathlib import Path
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
from method.Graph_visualize import plot_training_history, plot_validation_history
from training_method.Experiment import create_run, project_path, save_model, save_best_model
from training_method.Network_validation import prepare_validation, evaluate_validation, write_json
from training_method.PPO import PPOUpdater, PPOTransition, validate_ppo_settings


class APTrainer(PPOUpdater):
    def __init__(self, lr=None, gamma=None):
        lr=config.LR if lr is None else lr
        # Share GPU memory with the CuPy propagation model.
        for gpu in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # A caller may already have initialized the device.
        self.model=build_network(CNN_CHANNELS,NODE_FEATURES)
        self.policy_model=build_policy_model(self.model)
        if config.PRINT_MODEL_SUMMARY:
            self.model.summary(expand_nested=True)
        settings = config.snapshot()
        settings.update(LR=lr, GAMMA=config.GAMMA if gamma is None else gamma)
        validate_ppo_settings(settings)
        optimizer = tf.keras.optimizers.Adam(lr, epsilon=settings['PPO_ADAM_EPSILON'])
        super().__init__(self.model, self.policy_model, optimizer, settings)
        self.reward_settings=config.reward_settings()

    def cal_reward(self, coverage, prev_coverage, allocation_results, new_ap_connected,
                   prev_allocation_results):
        return calculate_reward(coverage, prev_coverage, allocation_results, new_ap_connected,
                                prev_allocation_results,self.reward_settings)

    def sample_action(self, state, action_mask):
        """Capture the masked behavior-policy probability/value BEFORE placement."""
        inputs=[state[k][None,...] for k in ('map','nodes','adjacency','mask')]
        logits,values=self.policy_model(inputs,training=False)
        log_probs=masked_log_probabilities(logits,action_mask[None,...])
        flat=tf.reshape(log_probs,[1,-1])
        idx=int(tf.random.categorical(flat,1)[0,0].numpy())
        width=action_mask.shape[1]
        return ((idx%width,idx//width), float(flat[0,idx].numpy()), float(values[0,0].numpy()))

    def select_action(self, state, action_mask):
        """Compatibility helper for callers that only need a sampled position."""
        return self.sample_action(state, action_mask)[0]



def random_valid_position(walls,forbidden,occupied_positions=()):
    ids=np.flatnonzero(placement_mask(walls,forbidden,occupied_positions))
    idx=np.random.choice(ids)
    W=np.asarray(walls).shape[1]
    return (int(idx%W),int(idx//W))
def summarize_training_rollout(episodes):
    """Training deployment metrics, one equal-weight observation per episode.

    Reward is the undiscounted episode SUM. Coverage/satisfaction use only the
    final AP deployment; they describe collection, before PPO optimization.
    """
    if not episodes:
        raise ValueError('Cannot summarize an empty training rollout')
    return {
        'mean_episode_reward': float(np.mean([
            sum(float(step[2]) for step in trajectory) for _, trajectory, _ in episodes])),
        'coverage_ratio': float(np.mean([records[-1][0] for _, _, records in episodes])),
        'throughput_satisfaction': float(np.mean([
            records[-1][2]['mean_satisfaction'] for _, _, records in episodes])),
    }


def run_training(map_dir=None, episodes=None, model_path=None, model_dir=None):
    """Train the AP-only mesh/OFDMA environment into a fresh experiment folder."""
    settings, radio_settings = config.snapshot(), radio.snapshot()
    validate_ppo_settings(settings)
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
    for key in ('ROLLOUT_EPISODES', 'TRAIN_MICROBATCH_SIZE', 'VALIDATION_INTERVAL', 'VALIDATION_EPISODES'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(settings['VALIDATION_RANDOM_SEED']) is not int or settings['VALIDATION_RANDOM_SEED'] < 0:
        raise ValueError('VALIDATION_RANDOM_SEED must be a nonnegative integer')
    if not isinstance(settings['NORMALIZE_ADVANTAGES'], bool):
        raise ValueError('NORMALIZE_ADVANTAGES must be True or False')
    if not 0 < settings['VALIDATION_MAP_FRACTION'] < 1:
        raise ValueError('VALIDATION_MAP_FRACTION must be between 0 and 1')
    best_filename = settings['BEST_MODEL_FILENAME']
    if (Path(best_filename).name != best_filename or not best_filename.endswith('.keras')):
        raise ValueError('BEST_MODEL_FILENAME must be a .keras filename without directories')
    if settings['CHECKPOINT_INTERVAL'] < 0:
        raise ValueError('CHECKPOINT_INTERVAL must be nonnegative')
    if model_path is not None and model_dir is not None:
        raise ValueError('Specify model_dir or a model_path template, not both')
    model_filename = settings['MODEL_FILENAME']
    if model_path is not None:
        template = project_path(model_path)
        model_dir, model_filename = template.parent, template.name
    model_dir = project_path(settings['MODEL_DIR'] if model_dir is None else model_dir)
    if best_filename == model_filename:
        raise ValueError('BEST_MODEL_FILENAME must differ from MODEL_FILENAME')
    files = sorted(map_dir.glob('*.json'))
    if not files:
        raise ValueError(f'No floorplan JSON files in {map_dir}')
    files, validation_manifest = prepare_validation(files, settings)
    channel_cfg = radio.channel_config(radio_settings)
    settings.update(MAP_DIR=str(map_dir), EPISODES=episodes,
                    MODEL_DIR=str(model_dir), MODEL_FILENAME=model_filename)
    run_dir = create_run(model_dir, settings['RUN_PREFIX'], model_filename, settings, radio_settings)
    write_json(run_dir / 'validation_scenarios.json', validation_manifest)
    print(f'Training run: {run_dir}')
    print(f'PPO-Clip: rollout={settings["ROLLOUT_EPISODES"]} episodes, '
          f'mini-batch={settings["PPO_MINIBATCH_SIZE"]} transitions, epochs={settings["PPO_EPOCHS"]}, '
          f'microbatch={settings["TRAIN_MICROBATCH_SIZE"]} transitions')
    print(f'Fixed validation: {settings["VALIDATION_EPISODES"]} scenarios, '
          f'{len(validation_manifest["validation_maps"])} maps; '
          f'{len(files)} training maps; scope={validation_manifest["scope"]}')
    if validation_manifest['scope'] == 'same_map_fixed_clients_only':
        print('Only one map: validation checks fixed clients on that map, not unseen-map generalization.')
    if settings['RANDOM_SEED'] is not None:
        random.seed(settings['RANDOM_SEED'])
        np.random.seed(settings['RANDOM_SEED'])
        tf.random.set_seed(settings['RANDOM_SEED'])
    trainer = APTrainer(lr=settings['LR'], gamma=settings['GAMMA'])
    keys = ('episode', 'total_loss', 'actor_loss', 'critic_loss', 'rewards', 'coverage',
            'coverage_gain', 'coverage_reward', 'throughput_reward', 'connectivity', 'mesh_connectivity',
            'mean_satisfaction', 'demand_met_ratio', 'total_throughput_mbps',
            'value', 'advantage', 'return_target', 'actor_advantage', 'optimizer_update',
            'old_log_probability', 'entropy', 'approx_kl', 'clip_fraction', 'probability_ratio', 'rollout')
    history = {key: [] for key in keys}
    pending, update_history, minibatch_history, validation_history = [], [], [], []
    best_validation = None
    next_validation = settings['VALIDATION_INTERVAL']
    next_checkpoint = settings['CHECKPOINT_INTERVAL']
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
            position, old_log_probability, old_value = trainer.sample_action(state, action_mask)
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
            trajectory.append(PPOTransition(state, position, reward, action_mask,
                old_log_probability, old_value, step + 1 == settings['AP_AMOUNT']))
            records.append((coverage, parts, dict(net['network_summary']), net['mesh_connectivity']))
            state = next_state
            
            print(f'EP {episode + 1} AP {step + 1}/{settings["AP_AMOUNT"]} pos={position} '
                  f'edges={len(net["mesh_edges"])} mesh={net["mesh_connectivity"]:.1%} '
                  f'coverage={coverage:.1%} satisfaction={net["network_summary"]["mean_satisfaction"]:.1%} '
                  f'conn={parts["connectivity"]:.0f} reward={reward:.4f}')
        pending.append((episode + 1, trajectory, records))
        if len(pending) < settings['ROLLOUT_EPISODES'] and episode + 1 < episodes:
            continue
        rollout_losses = trainer.train_rollout([item[1] for item in pending])
        deployment_metrics = summarize_training_rollout(pending)
        update = dict(trainer.last_update_metrics, first_episode=pending[0][0], last_episode=episode + 1,
                      **deployment_metrics)
        update_history.append(update)
        minibatch_history.extend(dict(row, rollout=update['rollout'], first_episode=pending[0][0],
                                      last_episode=episode + 1, **deployment_metrics)
                                 for row in trainer.last_minibatch_metrics)
        for (episode_id, trajectory, records), losses in zip(pending, rollout_losses):
            for loss, transition, record in zip(losses, trajectory, records):
                cov, parts, summary, mesh_connectivity = record
                metrics = dict(episode=episode_id, total_loss=loss['loss'],
                               actor_loss=loss['actor_loss'], critic_loss=loss['critic_loss'],
                               rewards=transition[2], coverage=cov, coverage_gain=parts['coverage_gain'],
                               coverage_reward=parts['coverage'], throughput_reward=parts['throughput'],
                               connectivity=parts['connectivity'], mesh_connectivity=mesh_connectivity,
                               value=loss['value'], advantage=loss['advantage'],
                               return_target=loss['return_target'], actor_advantage=loss['actor_advantage'],
                               optimizer_update=update['optimizer_update'], rollout=update['rollout'],
                               old_log_probability=loss['old_log_probability'], entropy=loss['entropy'],
                               approx_kl=loss['approx_kl'], clip_fraction=loss['clip_fraction'],
                               probability_ratio=loss['probability_ratio'],
                               mean_satisfaction=summary['mean_satisfaction'],
                               demand_met_ratio=summary['demand_met_ratio'],
                               total_throughput_mbps=summary['total_throughput_mbps'])
                for key, value in metrics.items():
                    history[key].append(value)
        pending.clear()
        ev = update['explained_variance']
        ev_text = 'N/A' if ev is None else f'{ev:.4f}'
        print(f'PPO ROLLOUT {update["rollout"]} EP {update["first_episode"]}-{episode + 1} '
              f'transitions={update["transitions"]} epochs={update["epochs_completed"]}/{settings["PPO_EPOCHS"]} '
              f'updates={update["optimizer_updates"]} total_updates={update["optimizer_update"]} '
              f'loss={update["mean_loss"]:.4f} KL={update["approx_kl"]:.6f} '
              f'clip={update["clip_fraction"]:.1%} entropy={update["entropy"]:.4f} EV={ev_text}'
              + (' [KL EARLY STOP]' if update['early_stopped'] else ''))
        write_json(run_dir / 'ppo_minibatch_history.json', minibatch_history)
        write_json(run_dir / 'update_history.json', update_history)
        np.savez(run_dir / 'training_history.npz', **{key: np.asarray(value) for key, value in history.items()})
        interval = settings['CHECKPOINT_INTERVAL']
        if interval and episode + 1 >= next_checkpoint:
            if episode + 1 < episodes:
                save_model(trainer.model, run_dir / 'checkpoints' / f'episode_{episode + 1:06d}.keras')
            next_checkpoint = ((episode + 1) // interval + 1) * interval
        if episode + 1 >= next_validation or episode + 1 == episodes:
            metrics, trials = evaluate_validation(trainer.policy_model, validation_manifest, settings, radio_settings)
            validation = dict(episode=episode + 1, optimizer_update=update['optimizer_update'], **metrics)
            validation_history.append(validation)
            write_json(run_dir / 'validation_history.json', validation_history)
            with (run_dir / 'validation_history.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(validation))
                writer.writeheader()
                writer.writerows(validation_history)
            improved = best_validation is None or validation['score'] > best_validation['score']
            if improved:
                save_best_model(trainer.model, run_dir, best_filename, validation)
                best_validation = validation
                write_json(run_dir / 'best_validation_trials.json', trials)
            satisfaction = metrics['mean_satisfaction']
            satisfaction_text = 'N/A' if satisfaction is None else f'{satisfaction:.1%}'
            print(f'VALIDATION EP {episode + 1}: score={metrics["score"]:.6f} '
                  f'coverage={metrics["coverage"]:.1%} '
                  f'satisfaction={satisfaction_text} '
                  f'mesh={metrics["mesh_connectivity"]:.1%}' + (' [BEST]' if improved else ''))
            next_validation = ((episode + 1) // settings['VALIDATION_INTERVAL'] + 1) * settings['VALIDATION_INTERVAL']
    np.savez(run_dir / 'training_history.npz', **{key: np.asarray(value) for key, value in history.items()})
    save_model(trainer.model, run_dir / model_filename)
    satisfaction = best_validation['mean_satisfaction']
    satisfaction_text = 'N/A' if satisfaction is None else f'{satisfaction:.2%}'
    print(f'Training complete (PPO-Clip): {episodes} episodes, {len(update_history)} rollouts, '
          f'{int(trainer.optimizer.iterations.numpy())} optimizer updates.')
    print(f'Best validation episode: {best_validation["episode"]} '
          f'(among {len(validation_history)} evaluated checkpoints)')
    print(f'Best objective score: {best_validation["score"]:.6f} | '
          f'coverage={best_validation["coverage"]:.2%} | satisfaction={satisfaction_text} | '
          f'mesh={best_validation["mesh_connectivity"]:.2%}')
    print(f'Best model: {run_dir / best_filename}')
    print(f'Final model: {run_dir / model_filename}')
    plot_validation_history(validation_history, best_validation['episode'],
                            save_path=str(run_dir / 'validation_history.png'), show=False)
    plot_training_history(history, save_path=str(run_dir / 'training_history.png'),
                          show=settings['SHOW_TRAINING_PLOTS'])
    return history


if __name__ == '__main__':
    LR_list = [0.0003, 0.0001, 0.00005, 0.00003, 0.00001]
    for i in range(len(LR_list)):
        config.LR = LR_list[i]
        run_training()
    
