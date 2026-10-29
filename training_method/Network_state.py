"""Six raster channels and 13 AP/client features; all mesh edges are visible."""
import numpy as np
import config
from method.Cal_link_capacity import RU_DATA_SUBCARRIERS

CNN_CHANNELS = 6
NODE_FEATURES = 13


class NetworkNormalizer:
    RSSI_MIN, RSSI_MAX = -100.0, -20.0
    SNR_MIN, SNR_MAX = -10.0, 40.0

    def __init__(self, reference_mbps):
        self.reference_mbps = float(reference_mbps)
        if not np.isfinite(self.reference_mbps) or self.reference_mbps <= 0:
            raise ValueError('Rate reference must be positive and finite')

    @staticmethod
    def clip01(value):
        return np.clip(value, 0.0, 1.0).astype(np.float32)

    @classmethod
    def rssi(cls, value):
        return cls.clip01((np.asarray(value) - cls.RSSI_MIN) / (cls.RSSI_MAX - cls.RSSI_MIN))

    @classmethod
    def snr(cls, value):
        return cls.clip01((np.asarray(value) - cls.SNR_MIN) / (cls.SNR_MAX - cls.SNR_MIN))

    def throughput(self, value):
        return self.clip01(np.asarray(value) / self.reference_mbps)


def normalize_material(walls):
    table = {0: 0.0, 1: 0.25, 3: 0.50, 6: 0.75, 10: 1.0}
    result = np.zeros_like(walls, dtype=np.float32)
    for key, value in table.items():
        result[np.asarray(walls) == key] = value
    return result


def build_state(walls, forbidden, ap_positions, client_positions, client_demands,
                total_rssi, network_data, t, ap_amount, max_nodes=None):
    max_nodes = config.MAX_NODES if max_nodes is None else max_nodes
    ap_count, client_count = len(ap_positions), len(client_positions)
    if ap_count + client_count > max_nodes:
        raise ValueError('MAX_NODES must fit all APs and clients')
    walls = np.asarray(walls)
    height, width = walls.shape
    normalizer = NetworkNormalizer(network_data['rate_reference_mbps'])
    rssi = np.nan_to_num(np.asarray(total_rssi), nan=-100.0, neginf=-100.0, posinf=-20.0)
    ap_map = np.zeros(walls.shape, dtype=np.float32)
    demand_map = np.zeros(walls.shape, dtype=np.float32)
    for x, y in ap_positions:
        ap_map[y, x] = 1.0
    for demand, (x, y) in zip(client_demands, client_positions):
        demand_map[y, x] = float(normalizer.throughput(demand))
    remaining = np.full(walls.shape, max(0.0, 1.0 - t / max(ap_amount, 1)), np.float32)
    raster = np.stack([normalize_material(walls), NetworkNormalizer.clip01(forbidden),
                       ap_map, demand_map, NetworkNormalizer.rssi(rssi), remaining], axis=-1)
    # [AP, client, x, y, RSSI, SNR, full capacity, demand, delivered,
    #  satisfaction, in_mesh/associated, RU data share, scheduling time share]
    nodes = np.zeros((max_nodes, NODE_FEATURES), np.float32)
    mask = np.zeros(max_nodes, np.float32)
    adjacency = np.eye(max_nodes, dtype=np.float32)
    allocations = network_data['allocation_results']
    resources = network_data['ap_resources']
    for i, (x, y) in enumerate(ap_positions):
        nodes[i, 0] = 1.0
        nodes[i, 2:4] = [x / max(width - 1, 1), y / max(height - 1, 1)]
        members = [row for row in allocations if row['ap'] == i]
        nodes[i, 6] = float(normalizer.throughput(network_data['rate_reference_mbps']))
        nodes[i, 7] = float(normalizer.throughput(sum(row['demand_mbps'] for row in members)))
        nodes[i, 8] = float(normalizer.throughput(resources[i]['throughput_mbps']))
        nodes[i, 9] = np.mean([row['satisfaction_ratio'] for row in members]) if members else 0.0
        nodes[i, 10] = float(network_data['in_mesh'][i])
        mask[i] = 1.0
    for client, (x, y) in enumerate(client_positions):
        index = ap_count + client
        row = allocations[client]
        ap = row['ap']
        nodes[index, 1] = 1.0
        nodes[index, 2:4] = [x / max(width - 1, 1), y / max(height - 1, 1)]
        if ap >= 0:
            link = network_data['client_links'][client]['links'][ap]
            nodes[index, 4] = float(normalizer.rssi(link['rssi_dbm']))
            nodes[index, 5] = float(normalizer.snr(link['snr_db']))
            adjacency[index, ap] = adjacency[ap, index] = 1.0
        nodes[index, 6] = float(normalizer.throughput(row['access_capacity_mbps']))
        nodes[index, 7] = float(normalizer.throughput(row['demand_mbps']))
        nodes[index, 8] = float(normalizer.throughput(row['throughput_mbps']))
        nodes[index, 9] = row['satisfaction_ratio']
        nodes[index, 10] = float(ap >= 0)
        # Allocation shares are relative to the AP's full-channel HE data tones.
        nodes[index, 11] = (RU_DATA_SUBCARRIERS.get(row['ru_tones'], 0)
                             / network_data.get('full_channel_data_subcarriers', 980))
        nodes[index, 12] = row['time_fraction']
        mask[index] = 1.0
    adjacency[:ap_count, :ap_count] += network_data['mesh_adjacency']
    degree = adjacency.sum(axis=1)
    inverse = 1.0 / np.sqrt(degree)
    adjacency = (inverse[:, None] * adjacency) * inverse[None, :]
    return {'map': raster.astype(np.float32), 'nodes': nodes,
            'adjacency': adjacency.astype(np.float32), 'mask': mask}
