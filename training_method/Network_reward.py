"""Coverage, signed satisfaction gains, and new-AP mesh connection rewards."""
import numpy as np
import config
import config_radio as radio
from method.Cal_SNR import as_numpy


def new_ap_connected_to_mesh(mesh_adjacency, previous_in_mesh):
    """Whether the appended AP links to the serving mesh BEFORE placement.

    The first AP earns no connection bonus. Linking only to an isolated group
    also earns none, even if that group becomes the new largest component.
    """
    adjacency = np.asarray(mesh_adjacency)
    members = np.asarray(previous_in_mesh, dtype=bool)
    count = len(members) + 1
    if members.ndim != 1 or adjacency.shape != (count, count):
        raise ValueError('Provide the new adjacency and the previous AP mesh membership')
    return bool(np.any((adjacency[-1, :-1] > 0) & members))


def calculate_reward(coverage, prev_coverage, allocation_results, new_ap_connected,
                     prev_allocation_results, reward_settings=None):
    if not isinstance(new_ap_connected, (bool, np.bool_)):
        raise ValueError('new_ap_connected must be a boolean, not a mesh connectivity ratio')
    settings = config.reward_settings() if reward_settings is None else reward_settings
    if settings['COVERAGE_GAIN_SCALE'] <= 0 or settings['THROUGHPUT_GAIN_SCALE'] <= 0:
        raise ValueError('Reward gain scales must be positive')
    coverage_gain = float(coverage - prev_coverage)
    satisfaction = np.mean([r['satisfaction_ratio'] for r in allocation_results]) if allocation_results else 0.0
    previous = np.mean([r['satisfaction_ratio'] for r in prev_allocation_results]) if prev_allocation_results else 0.0
    throughput_gain = float(satisfaction - previous)
    cov = float(np.clip(coverage_gain / settings['COVERAGE_GAIN_SCALE'], -1, 1))
    thr = float(np.clip(throughput_gain / settings['THROUGHPUT_GAIN_SCALE'], -1, 1))
    conn = float(new_ap_connected)
    reward = (settings['COVERAGE_REWARD_WEIGHT'] * cov
              + settings['THROUGHPUT_REWARD_WEIGHT'] * thr
              + settings['CONNECTIVITY_REWARD_WEIGHT'] * conn)
    return float(reward), {'coverage': cov, 'coverage_gain': coverage_gain,
                           'throughput': thr, 'throughput_gain': throughput_gain,
                           'connectivity': conn}


def mesh_rssi_map(rssi_maps, in_mesh, shape=None):
    if len(rssi_maps) != len(in_mesh):
        raise ValueError('RSSI maps and mesh membership must align')
    if shape is None:
        if not rssi_maps:
            raise ValueError('Provide a map shape before any AP is placed')
        shape = rssi_maps[0].shape
    total = np.full(shape, -np.inf, dtype=np.float32)
    for signal, member in zip(rssi_maps, in_mesh):
        if member:
            total = np.fmax(total, as_numpy(signal))
    return total


def coverage_score(total_rssi, threshold=None, valid_mask=None):
    threshold = radio.COVERAGE_THRESHOLD_DBM if threshold is None else threshold
    values = as_numpy(total_rssi)
    valid = np.ones(values.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != values.shape:
        raise ValueError('Coverage mask must match RSSI map shape')
    return float(np.mean(np.isfinite(values[valid]) & (values[valid] >= threshold))) if np.any(valid) else 0.0
