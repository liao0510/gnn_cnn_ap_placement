"""Per-AP equal-RU downlink OFDMA with long-term round-robin service.

Choose the largest homogeneous legal RU layout that fits all active clients.
If even 26-tone RUs cannot fit everyone, rotate over the maximum disjoint slots.
Each client gets one RU per selected TXOP. Demand caps do not reclaim idle RUs.
SNR is unchanged across RUs under constant power spectral density / flat fading.
"""
import math
from method.Cal_link_capacity import RU_LAYOUTS, calculate_link_capacity


def txop_allocation(resource, txop_index):
    """Return disjoint RU slots for a particular equal-duration TXOP."""
    if not isinstance(txop_index, int) or txop_index < 0:
        raise ValueError('TXOP index must be a nonnegative integer')
    active, slots = resource['active_clients'], resource['concurrent_clients']
    if not active:
        return []
    start = (txop_index * slots) % len(active)
    return [{'client': active[(start + offset) % len(active)],
             'ru_index': offset, 'ru_tones': resource['ru_tones']}
            for offset in range(slots)]


def allocate_throughput(client_demands, connected_AP, client_snrs, ap_count, cfg,
                        *, mac_efficiency=None):
    count = len(client_demands)
    if not len(connected_AP) == len(client_snrs) == count:
        raise ValueError('Provide one association and SNR per client demand')
    if not isinstance(ap_count, int) or ap_count < 0:
        raise ValueError('ap_count must be a nonnegative integer')
    demands = [float(d) for d in client_demands]
    if any(not math.isfinite(d) or d < 0 for d in demands):
        raise ValueError('Client demands must be finite and nonnegative')

    def capacity(snr, ru=None):
        return calculate_link_capacity(snr, mac_efficiency, ru_tones=ru,
                                       bandwidth_mhz=cfg.bandwidth_mhz,
                                       spatial_streams=cfg.spatial_streams,
                                       guard_interval_us=cfg.guard_interval_us)

    results = []
    for client, (ap, snr, demand) in enumerate(zip(connected_AP, client_snrs, demands)):
        if ap is None or ap is False or not isinstance(ap, int) or not -1 <= ap < ap_count:
            raise ValueError('Client associations must use -1 or a valid integer AP ID')
        full = capacity(snr)
        results.append({'client': client, 'ap': ap, 'snr_db': snr,
                        'demand_mbps': demand, 'mcs': full['mcs'],
                        'access_capacity_mbps': full['link_capacity_mbps'] if ap >= 0 else 0.0,
                        'ru_tones': 0, 'ru_capacity_mbps': 0.0, 'time_fraction': 0.0,
                        'scheduled_capacity_mbps': 0.0, 'throughput_mbps': 0.0,
                        'satisfaction_ratio': 1.0 if demand == 0 else 0.0,
                        'demand_met': demand == 0})
    summaries = []
    for ap in range(ap_count):
        active = [r['client'] for r in results if r['ap'] == ap and r['demand_mbps'] > 0
                  and r['access_capacity_mbps'] > 0]
        layouts = RU_LAYOUTS[cfg.bandwidth_mhz]
        tones = max((ru for ru, slots in layouts.items() if slots >= len(active)),
                    default=26) if active else 0
        slots = layouts[tones] if active else 0
        concurrent = min(len(active), slots)
        duty = concurrent / len(active) if active else 0.0
        # Advance the example's start offset by concurrent slots each TXOP.
        representative = [{'client': client, 'ru_index': index, 'ru_tones': tones}
                          for index, client in enumerate(active[:concurrent])]
        for client in active:
            row = results[client]
            ru = capacity(row['snr_db'], tones)
            scheduled = ru['link_capacity_mbps'] * duty
            delivered = min(row['demand_mbps'], scheduled)
            row.update(ru_tones=tones, ru_capacity_mbps=ru['link_capacity_mbps'],
                       time_fraction=duty, scheduled_capacity_mbps=scheduled,
                       throughput_mbps=delivered,
                       satisfaction_ratio=delivered / row['demand_mbps'],
                       demand_met=delivered + 1e-9 >= row['demand_mbps'])
        summaries.append({'ap': ap, 'associated_clients': [r['client'] for r in results if r['ap'] == ap],
                          'active_clients': active, 'ru_tones': tones,
                          'available_ru_slots': slots, 'concurrent_clients': concurrent,
                          'client_time_fraction': duty, 'txop_example': representative,
                          'round_robin_stride': concurrent,
                          'throughput_mbps': sum(results[c]['throughput_mbps'] for c in active)})
    return results, summaries
