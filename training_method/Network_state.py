"""Residual-demand rasters and spatially explicit AP/client node features."""
import numpy as np
import config
import config_radio as radio
from method.Cal_link_capacity import RU_DATA_SUBCARRIERS

CNN_CHANNELS = 9
NODE_FEATURES = 15
# Channels 0..5 retain their spatial roles; channel 3 now tracks remaining demand.
REMAINING_DEMAND_CHANNEL = 3
ORIGINAL_DEMAND_CHANNEL = 6
DENSITY_CHANNELS = (7, 8)
REMAINING_DEMAND_FEATURE = 13
REMAINING_RATIO_FEATURE = 14


class NetworkNormalizer:
    RSSI_MIN, RSSI_MAX = -100.0, -20.0
    SINR_MIN, SINR_MAX = -40.0, 40.0

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
    def sinr(cls, value):
        return cls.clip01((np.asarray(value) - cls.SINR_MIN) / (cls.SINR_MAX - cls.SINR_MIN))

    def throughput(self, value):
        return self.clip01(np.asarray(value) / self.reference_mbps)


def normalize_material(walls):
    table = {0: 0.0, 1: 0.25, 3: 0.50, 6: 0.75, 10: 1.0}
    result = np.zeros_like(walls, dtype=np.float32)
    for key, value in table.items():
        result[np.asarray(walls) == key] = value
    return result


def demand_density_maps(shape, clients, remaining_mbps, grid_size_meters, settings):
    """Two physical-distance demand summaries, not propagation/SINR estimates.

    Gaussian peaks add in Mbps. Fixed scaling preserves demand magnitude across
    scenarios; no per-image max normalization or wall-dependent attenuation.
    """
    sigmas = np.asarray(settings['CLIENT_DENSITY_SIGMAS_METERS'], dtype=float)
    reference = float(settings['CLIENT_DENSITY_REFERENCE_MBPS'])
    if (sigmas.shape != (2,) or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0)
            or not np.isfinite(reference) or reference <= 0
            or not np.isfinite(grid_size_meters) or grid_size_meters <= 0):
        raise ValueError('Demand density requires two positive finite sigmas, reference and grid size')
    height, width = shape
    xs = np.arange(width, dtype=np.float32) * grid_size_meters
    ys = np.arange(height, dtype=np.float32) * grid_size_meters
    result = []
    for sigma in sigmas:
        density = np.zeros(shape, np.float32)
        for (x, y), residual in zip(clients, remaining_mbps):
            if residual > 0:
                gx = np.exp(-0.5 * ((xs - x * grid_size_meters) / sigma) ** 2)
                gy = np.exp(-0.5 * ((ys - y * grid_size_meters) / sigma) ** 2)
                density += residual * gy[:, None] * gx[None, :]
        result.append(density / (density + reference))
    return result


def build_state(walls, forbidden, ap_positions, client_positions, client_demands,
                total_rssi, network_data, t, ap_amount, max_nodes=None, *,
                feature_settings=None, grid_size_meters=None):
    max_nodes = config.MAX_NODES if max_nodes is None else max_nodes
    ap_count, client_count = len(ap_positions), len(client_positions)
    if ap_count + client_count > max_nodes:
        raise ValueError('MAX_NODES must fit all APs and clients')
    walls = np.asarray(walls)
    height, width = walls.shape
    normalizer = NetworkNormalizer(network_data['rate_reference_mbps'])
    settings = config.snapshot() if feature_settings is None else feature_settings
    grid_size = radio.GRID_SIZE_METERS if grid_size_meters is None else grid_size_meters
    allocations = network_data['allocation_results']
    demands = np.asarray(client_demands, dtype=float)
    delivered = np.asarray([row['throughput_mbps'] for row in allocations], dtype=float)
    if (demands.shape != (client_count,) or delivered.shape != demands.shape
            or not np.all(np.isfinite(demands)) or np.any(demands < 0)
            or not np.all(np.isfinite(delivered)) or np.any(delivered < 0)):
        raise ValueError('Provide one finite nonnegative demand and allocation per client')
    residual = np.maximum(demands - delivered, 0.0)
    # Match the allocator's demand-met tolerance, including zero-demand clients.
    residual[delivered + 1e-9 >= demands] = 0.0
    residual_ratio = np.divide(residual, demands, out=np.zeros_like(residual), where=demands > 0)
    rssi = np.nan_to_num(np.asarray(total_rssi), nan=-100.0, neginf=-100.0, posinf=-20.0)
    ap_map = np.zeros(walls.shape, dtype=np.float32)
    demand_map = np.zeros(walls.shape, dtype=np.float32)
    original_demand_map = np.zeros(walls.shape, dtype=np.float32)
    for x, y in ap_positions:
        ap_map[y, x] = 1.0
    for demand, unmet, (x, y) in zip(demands, residual, client_positions):
        demand_map[y, x] = float(normalizer.throughput(unmet))
        original_demand_map[y, x] = float(normalizer.throughput(demand))
    density_maps = demand_density_maps(walls.shape, client_positions, residual, grid_size, settings)
    remaining = np.full(walls.shape, max(0.0, 1.0 - t / max(ap_amount, 1)), np.float32)
    raster = np.stack([normalize_material(walls), NetworkNormalizer.clip01(forbidden),
                       ap_map, demand_map, NetworkNormalizer.rssi(rssi), remaining,
                       original_demand_map, *density_maps], axis=-1)
    # [AP, client, x, y, RSSI, access SINR, full capacity, demand, delivered,
    #  satisfaction, in_mesh/associated, RU data share, scheduling time share,
    #  remaining demand (capacity-normalized), remaining demand / original demand]
    nodes = np.zeros((max_nodes, NODE_FEATURES), np.float32)
    mask = np.zeros(max_nodes, np.float32)
    adjacency = np.eye(max_nodes, dtype=np.float32)
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
        # AP nodes expose the same allocation-share fields as client nodes.
        # Exact associated-client counts are read from graph edges by AP attention.
        nodes[i, 11] = (RU_DATA_SUBCARRIERS.get(resources[i]['ru_tones'], 0)
                         / network_data.get('full_channel_data_subcarriers', 980))
        nodes[i, 12] = resources[i]['client_time_fraction']
        nodes[i, REMAINING_DEMAND_FEATURE] = float(normalizer.throughput(
            sum(residual[row['client']] for row in members)))
        requested = sum(row['demand_mbps'] for row in members)
        nodes[i, REMAINING_RATIO_FEATURE] = (
            sum(residual[row['client']] for row in members) / requested if requested > 0 else 0.0)
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
            nodes[index, 5] = float(normalizer.sinr(link['sinr_db']))
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
        nodes[index, REMAINING_DEMAND_FEATURE] = float(normalizer.throughput(residual[client]))
        nodes[index, REMAINING_RATIO_FEATURE] = residual_ratio[client]
        mask[index] = 1.0
    adjacency[:ap_count, :ap_count] += network_data['mesh_adjacency']
    degree = adjacency.sum(axis=1)
    inverse = 1.0 / np.sqrt(degree)
    adjacency = (inverse[:, None] * adjacency) * inverse[None, :]
    return {'map': raster.astype(np.float32), 'nodes': nodes,
            'adjacency': adjacency.astype(np.float32), 'mask': mask}
