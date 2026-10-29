"""Greedy placement with the same AP mesh and OFDMA model used for training."""
import json
import sys
import numpy as np
import tensorflow as tf
import config
import config_radio as radio
from method.Cal_average_gpu import Cal_db_gpu
from method.Cal_SNR import as_numpy
from method.Graph_visualize import plot_prediction_results
from training_method.GNN_CNN_Network import build_network
from training_method.Network_policy import build_policy_model, masked_log_probabilities, placement_mask
from training_method.Network_environment import run_network_environment
from training_method.Network_state import build_state, CNN_CHANNELS, NODE_FEATURES
from training_method.Network_reward import (mesh_rssi_map, coverage_score, calculate_reward,
                                            new_ap_connected_to_mesh)
from training_method.Experiment import latest_model, model_snapshot, project_path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray) or hasattr(value, 'get'):
        return json_safe(as_numpy(value).tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_prediction(model_path=None):
    settings = config.snapshot()
    ap_count, client_count = settings['PREDICT_AP_COUNT'], settings['PREDICT_CLIENT_COUNT']
    if any(not isinstance(n, int) or n < 0 for n in (ap_count, client_count)):
        raise ValueError('AP and client counts must be nonnegative integers')
    if settings['MAX_NODES'] < ap_count + client_count:
        raise ValueError('MAX_NODES must fit all APs and clients')
    requested = settings['PREDICT_MODEL_PATH'] if model_path is None else model_path
    model_path = latest_model(settings['MODEL_DIR']) if requested is None else project_path(requested)
    print(f"model_path:{model_path}")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    saved = model_snapshot(model_path)
    if saved is None or saved.get('environment_version') != config.ENVIRONMENT_VERSION:
        raise ValueError('Model environment is incompatible; train a new AP mesh/OFDMA model')
    radio_settings = saved['radio']
    training_settings = saved['training']
    reward_settings = {key: training_settings[key] for key in config.reward_settings()}
    cfg = radio.channel_config(radio_settings)
    map_path = project_path(settings['PREDICT_MAP_PATH'])
    with map_path.open(encoding='utf-8') as stream:
        floorplan = json.load(stream)
    walls = np.asarray(floorplan['wall_map'])
    forbidden = np.asarray(floorplan.get('forbidden', np.zeros_like(walls)))
    if walls.ndim != 2 or forbidden.shape != walls.shape:
        raise ValueError('wall_map and forbidden must be matching 2-D maps')
    valid = (walls == 0) & (forbidden == 0)
    height, width = walls.shape
    if np.count_nonzero(valid) < ap_count + client_count:
        raise ValueError('Not enough distinct free positions for all devices')
    rng = np.random.default_rng(settings['PREDICT_RANDOM_SEED'])
    if settings['PREDICT_RANDOM_SEED'] is not None:
        tf.random.set_seed(settings['PREDICT_RANDOM_SEED'])
    positions = settings['PREDICT_CLIENT_POSITIONS']
    if positions is None:
        ids = rng.choice(np.flatnonzero(valid), size=client_count, replace=False)
        clients = [(int(index % width), int(index // width)) for index in ids]
    else:
        clients = []
        for position in positions:
            if len(position) != 2 or any(int(value) != value for value in position):
                raise ValueError('Positions must use integer (x, y) coordinates')
            x, y = map(int, position)
            if not (0 <= x < width and 0 <= y < height and valid[y, x]):
                raise ValueError(f'Invalid client position: {(x, y)}')
            clients.append((x, y))
        if len(clients) != client_count or len(set(clients)) != client_count:
            raise ValueError('Provide CLIENT_COUNT distinct client positions')
    demands = ([float(settings['PREDICT_CLIENT_DEMAND_MBPS'])] * client_count
               if settings['PREDICT_CLIENT_DEMANDS'] is None
               else [float(value) for value in settings['PREDICT_CLIENT_DEMANDS']])
    if len(demands) != client_count or any(not np.isfinite(value) or value < 0 for value in demands):
        raise ValueError('Provide one finite nonnegative demand per client')
    model = build_network(CNN_CHANNELS, NODE_FEATURES)
    model.load_weights(str(model_path))
    policy = build_policy_model(model)
    if settings['PRINT_PREDICTION_MODEL_SUMMARY']:
        model.summary(expand_nested=True)
    simulator = Cal_db_gpu(walls=walls, frequency_ghz=radio_settings['ACCESS_FREQUENCY_GHZ'],
                           tx_power=radio_settings['TX_POWER_DBM'],
                           grid_size=radio_settings['GRID_SIZE_METERS'],
                           ray_steps=training_settings['RSSI_RAY_STEPS'],
                           row_chunk=training_settings['RSSI_ROW_CHUNK'])
    aps, maps, history = [], [], []

    def evaluate():
        return run_network_environment(aps, clients, demands, maps, cfg,
                                       mac_efficiency=radio_settings['MAC_EFFICIENCY'])

    net = evaluate()
    total = mesh_rssi_map(maps, net['in_mesh'], walls.shape)
    threshold = radio_settings['COVERAGE_THRESHOLD_DBM']
    coverage = coverage_score(total, threshold, walls == 0)
    initial_coverage = coverage
    for step in range(ap_count):
        state = build_state(walls, forbidden, aps, clients, demands, total,
                            net, step, ap_count, settings['MAX_NODES'])
        inputs = [state[key][None] for key in ('map', 'nodes', 'adjacency', 'mask')]
        available = placement_mask(walls, forbidden, aps + clients)
        logits, value = policy(inputs, training=False)
        log_probs = masked_log_probabilities(logits, available[None]).numpy()[0, :, :, 0]
        index = int(np.argmax(log_probs))
        position = (index % width, index // width)
        previous_allocations, previous_coverage = net['allocation_results'], coverage
        previous_in_mesh = net['in_mesh']
        aps.append(position)
        signal, _ = simulator.calculate_signal_map(position)
        maps.append(as_numpy(signal))
        net = evaluate()
        new_ap_connected = new_ap_connected_to_mesh(net['mesh_adjacency'], previous_in_mesh)
        total = mesh_rssi_map(maps, net['in_mesh'], walls.shape)
        coverage = coverage_score(total, threshold, walls == 0)
        reward, parts = calculate_reward(coverage, previous_coverage, net['allocation_results'],
                                         new_ap_connected, previous_allocations, reward_settings)
        history.append({'step': step + 1, 'position': position, 'coverage': coverage,
                        'mesh_connectivity': net['mesh_connectivity'],
                        'new_ap_connected': new_ap_connected,
                        'mesh_edge_count': len(net['mesh_edges']), 'reward': reward,
                        'reward_parts': parts, 'state_value': float(value.numpy().squeeze()),
                        'action_probability': float(np.exp(log_probs[position[1], position[0]])),
                        'network_summary': dict(net['network_summary'])})
        print(f'AP {step + 1}/{ap_count}: {position}, edges={len(net["mesh_edges"])}, '
              f'coverage={coverage:.1%}, conn={parts["connectivity"]:.0f}, reward={reward:.4f}')
    report = json_safe({'environment_version': config.ENVIRONMENT_VERSION,
                       'model_path': str(model_path), 'map_path': str(map_path),
                       'seed': settings['PREDICT_RANDOM_SEED'], 'training_run_id': saved['run_id'],
                       'ap_positions': aps, 'client_positions': clients, 'client_demands': demands,
                       'initial_coverage': initial_coverage, 'coverage': coverage,
                       'coverage_threshold_dbm': threshold, 'radio_config': radio_settings,
                       'assumptions': {'serving_mesh': 'largest_connected_component',
                                       'connectivity_reward': 'new_ap_connected_to_previous_serving_mesh',
                                       'noise_limited': True, 'constant_power_spectral_density': True,
                                       'ofdma_scheduler': radio_settings['OFDMA_SCHEDULER'],
                                       'equal_ru_sizes': True, 'demand_reclaims_idle_rus': False},
                       'history': history, 'network': net})
    output = project_path(settings['PREDICT_OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'prediction_results.json').open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    np.savez_compressed(output / 'prediction_maps.npz', walls=walls, forbidden=forbidden,
                        mesh_rssi=total, mesh_adjacency=net['mesh_adjacency'])
    plot_prediction_results(walls, forbidden, total, report, output_dir=output,
                            show=settings['SHOW_PREDICTION_PLOTS'])
    print(f'Results saved to: {output}')
    return report


if __name__ == '__main__':
    run_prediction()
