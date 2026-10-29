"""SNR links, complete threshold mesh, and strongest-SNR association."""
import numpy as np
from config_radio import ChannelConfig


def as_numpy(value):
    return np.asarray(value.get() if hasattr(value, 'get') else value)


def calculate_noise_dbm(cfg):
    return -174.0 + 10.0 * np.log10(cfg.bandwidth_mhz * 1e6) + cfg.noise_figure_db


def build_ap_mesh(ap_positions, rssi_maps, cfg):
    """Keep every bidirectionally valid edge, including cycles.

    The largest connected component is the serving mesh. Ties use the lowest AP
    ID, without assigning a special AP role. A single AP is a one-node mesh.
    """
    count = len(ap_positions)
    if len(rssi_maps) != count:
        raise ValueError('Provide one RSSI map per AP')
    adjacency = np.zeros((count, count), dtype=np.float32)
    edges = []
    noise = calculate_noise_dbm(cfg)
    for i in range(count):
        for j in range(i + 1, count):
            xj, yj = ap_positions[j]
            xi, yi = ap_positions[i]
            forward = float(rssi_maps[i][yj, xj]) - noise
            reverse = float(rssi_maps[j][yi, xi]) - noise
            if (np.isfinite(forward) and np.isfinite(reverse)
                    and min(forward, reverse) >= cfg.mesh_snr_threshold_db):
                adjacency[i, j] = adjacency[j, i] = 1.0
                edges.append({'ap_a': i, 'ap_b': j, 'snr_a_to_b_db': forward,
                              'snr_b_to_a_db': reverse, 'snr_db': min(forward, reverse)})
    components, unseen = [], set(range(count))
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        component, stack = [], [first]
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = sorted(set(np.flatnonzero(adjacency[node])) & unseen)
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
        components.append(sorted(component))
    largest = min(components, key=lambda group: (-len(group), group[0])) if components else []
    members = set(largest)
    return {'mesh_edges': edges, 'mesh_adjacency': adjacency,
            'mesh_components': components, 'in_mesh': [i in members for i in range(count)],
            'mesh_connected': bool(count and len(components) == 1),
            'mesh_connectivity': len(largest) / count if count else 0.0}


def inspect_client_links(client_positions, ap_positions, rssi_maps, in_mesh, cfg):
    if not len(ap_positions) == len(rssi_maps) == len(in_mesh):
        raise ValueError('AP positions, RSSI maps and mesh membership must align')
    noise = calculate_noise_dbm(cfg)
    results, associated, selected_snr = [], [], []
    for client_id, (x, y) in enumerate(client_positions):
        links, best_ap, best_snr = [], -1, -np.inf
        for ap_id, signal in enumerate(rssi_maps):
            rssi = float(signal[y, x])
            snr = rssi - noise
            valid = bool(in_mesh[ap_id] and np.isfinite(snr)
                         and snr >= cfg.client_snr_threshold_db)
            links.append({'ap': ap_id, 'rssi_dbm': rssi, 'snr_db': snr,
                          'in_mesh': bool(in_mesh[ap_id]), 'link_valid': valid})
            if valid and snr > best_snr:
                best_ap, best_snr = ap_id, snr
        results.append({'client': client_id, 'position': (x, y), 'links': links})
        associated.append(best_ap)
        selected_snr.append(None if best_ap < 0 else best_snr)
    return results, associated, selected_snr
