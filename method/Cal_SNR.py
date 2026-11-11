"""AP-AP reference SNR mesh and strongest-SINR AP-client association."""
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
    """Reevaluate every candidate after each placement using all deployed APs.

    Each candidate is the desired transmitter; every OTHER AP is an always-on,
    full-band co-channel interferer, including APs outside the serving mesh and
    APs with no clients. Mesh membership controls association eligibility only.
    Input powers are per-AP received dBm at the CLIENT, before display clipping.
    Sum other powers directly in float64 (never subtract desired from a rounded
    total, which loses weak interference next to a very strong desired signal).
    """
    if not len(ap_positions) == len(rssi_maps) == len(in_mesh):
        raise ValueError('AP positions, RSSI maps and mesh membership must align')
    noise = calculate_noise_dbm(cfg)
    noise_mw = 10.0 ** (noise / 10.0)
    results, associated, selected_sinr = [], [], []
    for client_id, (x, y) in enumerate(client_positions):
        received_dbm = np.asarray([signal[y, x] for signal in rssi_maps], dtype=np.float64)
        finite = np.isfinite(received_dbm)
        powers_mw = np.zeros(len(ap_positions), dtype=np.float64)
        powers_mw[finite] = np.power(10.0, received_dbm[finite] / 10.0)
        if not np.all(np.isfinite(powers_mw)):
            raise ValueError('Received access powers must have finite linear representations')
        links, best_ap, best_sinr = [], -1, -np.inf
        for ap_id, rssi in enumerate(received_dbm):
            other = np.arange(len(ap_positions)) != ap_id
            interference = float(powers_mw[other].sum())
            denominator_dbm = float(10.0 * np.log10(noise_mw + interference))
            sinr = float(rssi - denominator_dbm) if finite[ap_id] else None
            valid = bool(in_mesh[ap_id] and sinr is not None
                         and sinr >= cfg.client_sinr_threshold_db)
            links.append({'ap': ap_id, 'rssi_dbm': float(rssi), 'sinr_db': sinr,
                          'noise_dbm': float(noise), 'interference_mw': interference,
                          'interference_dbm': (float(10.0 * np.log10(interference))
                                               if interference > 0 else None),
                          'noise_interference_dbm': denominator_dbm,
                          'interfering_aps': np.flatnonzero(other & (powers_mw > 0)).tolist(),
                          'in_mesh': bool(in_mesh[ap_id]), 'link_valid': valid})
            if valid and sinr > best_sinr:
                best_ap, best_sinr = ap_id, sinr
        results.append({'client': client_id, 'position': (x, y), 'links': links})
        associated.append(best_ap)
        selected_sinr.append(None if best_ap < 0 else best_sinr)
    return results, associated, selected_sinr
