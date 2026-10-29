"""Shared AP mesh, strongest-SNR association, and OFDMA evaluation."""
import numpy as np
import config_radio as radio
from method.Cal_SNR import as_numpy, build_ap_mesh, inspect_client_links
from method.Cal_link_capacity import calculate_link_capacity, FULL_CHANNEL_RU, RU_DATA_SUBCARRIERS
from method.Cal_throughput_allocation import allocate_throughput


def run_network_environment(ap_positions, client_positions, client_demands, rssi_maps,
                            cfg=None, *, mac_efficiency=None):
    cfg = radio.channel_config() if cfg is None else cfg
    if len(client_positions) != len(client_demands):
        raise ValueError('Client positions and demands must align')
    maps = [as_numpy(signal) for signal in rssi_maps]
    mesh = build_ap_mesh(ap_positions, maps, cfg)
    links, associated, snrs = inspect_client_links(client_positions, ap_positions, maps,
                                                  mesh['in_mesh'], cfg)
    allocations, ap_resources = allocate_throughput(client_demands, associated, snrs,
                                                   len(ap_positions), cfg,
                                                   mac_efficiency=mac_efficiency)
    reference = calculate_link_capacity(37.0, mac_efficiency,
                                       bandwidth_mhz=cfg.bandwidth_mhz,
                                       spatial_streams=cfg.spatial_streams,
                                       guard_interval_us=cfg.guard_interval_us)['link_capacity_mbps']
    summary = {'total_throughput_mbps': sum(r['throughput_mbps'] for r in allocations),
               'mean_satisfaction': float(np.mean([r['satisfaction_ratio'] for r in allocations])) if allocations else 0.0,
               'demand_met_ratio': float(np.mean([r['demand_met'] for r in allocations])) if allocations else 0.0,
               'associated_clients': sum(ap >= 0 for ap in associated)}
    return dict(mesh, client_links=links, connected_AP=associated, connected_AP_SNR=snrs,
                client_capacities=[r['access_capacity_mbps'] for r in allocations],
                allocation_results=allocations, ap_resources=ap_resources,
                rate_reference_mbps=max(reference, 1.0),
                full_channel_data_subcarriers=RU_DATA_SUBCARRIERS[FULL_CHANNEL_RU[cfg.bandwidth_mhz]],
                network_summary=summary)
