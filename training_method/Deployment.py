"""Shared greedy/random deployment evaluation for prediction and validation."""
import json
import numpy as np
from method.Cal_SNR import as_numpy
from training_method.Network_environment import run_network_environment
from training_method.Network_policy import masked_log_probabilities, placement_mask
from training_method.Network_reward import mesh_rssi_map, coverage_score
from training_method.Network_state import build_state
from training_method.Prediction_analysis import final_metrics


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


def read_floorplan(path, required_positions):
    with path.open(encoding='utf-8') as stream:
        floorplan = json.load(stream)
    walls = np.asarray(floorplan['wall_map'])
    forbidden = np.asarray(floorplan.get('forbidden', np.zeros_like(walls)))
    if (walls.ndim != 2 or not walls.size or forbidden.shape != walls.shape
            or not np.all(np.isfinite(walls)) or not np.all(np.isfinite(forbidden))):
        raise ValueError(f'Invalid wall_map / forbidden in {path}')
    if np.count_nonzero((walls == 0) & (forbidden == 0)) < required_positions:
        raise ValueError(f'Not enough distinct free positions in {path}')
    return walls, forbidden


def evaluate_deployment(walls, forbidden, clients, demands, ap_count, simulator,
                        cfg, radio_settings, max_nodes, policy=None, rng=None):
    """Same argmax policy and physical environment as the single predictor.

    With policy=None, place APs uniformly without replacement using the same
    wall/forbidden/occupied mask. Neither method forces mesh connectivity.
    """
    if policy is None and rng is None:
        raise ValueError('Random baseline requires its own RNG')
    aps, maps = [], []
    net = run_network_environment(aps, clients, demands, maps, cfg,
                                  mac_efficiency=radio_settings['MAC_EFFICIENCY'])
    total = mesh_rssi_map(maps, net['in_mesh'], walls.shape)
    width = walls.shape[1]
    for step in range(ap_count):
        available = placement_mask(walls, forbidden, aps + clients)
        if policy is None:
            index = int(rng.choice(np.flatnonzero(available)))
        else:
            state = build_state(walls, forbidden, aps, clients, demands, total,
                                net, step, ap_count, max_nodes)
            inputs = [state[key][None] for key in ('map', 'nodes', 'adjacency', 'mask')]
            logits, _ = policy(inputs, training=False)
            log_probs = masked_log_probabilities(logits, available[None]).numpy()
            index = int(np.argmax(log_probs))
        position = (index % width, index // width)
        aps.append(position)
        signal, _ = simulator.calculate_signal_map(position)
        maps.append(as_numpy(signal))
        net = run_network_environment(aps, clients, demands, maps, cfg,
                                      mac_efficiency=radio_settings['MAC_EFFICIENCY'])
        total = mesh_rssi_map(maps, net['in_mesh'], walls.shape)
    coverage = coverage_score(total, radio_settings['COVERAGE_THRESHOLD_DBM'], walls == 0)
    return json_safe({'ap_positions': aps, 'metrics': final_metrics(coverage, net),
                      'mesh_edges': net['mesh_edges'], 'mesh_components': net['mesh_components'],
                      'in_mesh': net['in_mesh'], 'allocation_results': net['allocation_results']})


