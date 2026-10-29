"""AP/client SNR and OFDMA demonstration with topology and detailed plots."""
import json
import argparse
from pathlib import Path
import numpy as np
import config_radio as radio
from method.Cal_average_gpu import Cal_db_gpu
from method.Cal_SNR import as_numpy
from method.Graph_visualize import plot_prediction_results
from training_method.Network_environment import run_network_environment
from training_method.Network_reward import mesh_rssi_map, coverage_score


def main(map_path=None, output_dir=None, show=True):
    root = Path(__file__).resolve().parent
    map_path = root / 'maps_100_100' / 'map_0000.json' if map_path is None else Path(map_path)
    floorplan = json.loads(map_path.read_text(encoding='utf-8'))
    walls = np.asarray(floorplan['wall_map'])
    forbidden = np.asarray(floorplan.get('forbidden', np.zeros_like(walls)))
    aps = [(21, 29), (30, 60), (50, 70), (80, 24)]
    clients = [(11, 33), (24, 44), (45, 44), (51, 37), (75, 24), (75, 12), (91, 50), (15, 46)]
    demands = [100.] * len(clients)
    simulator = Cal_db_gpu(walls=walls)
    maps = [as_numpy(simulator.calculate_signal_map(position)[0]) for position in aps]
    net = run_network_environment(aps, clients, demands, maps, radio.channel_config())
    print('Mesh edges:', net['mesh_edges'])
    for row in net['allocation_results']:
        ap_label = f'AP{row["ap"]}' if row['ap'] >= 0 else 'Unassociated'
        print(f'C{row["client"]} -> {ap_label}, SNR={row["snr_db"]:.2f}, '
              f'RU={row["ru_tones"]}, throughput={row["throughput_mbps"]:.2f} Mbps')
    total_rssi = mesh_rssi_map(maps, net['in_mesh'], walls.shape)
    coverage = coverage_score(total_rssi, radio.COVERAGE_THRESHOLD_DBM, walls == 0)
    report = {'ap_positions': aps, 'client_positions': clients, 'client_demands': demands,
              'initial_coverage': 0.0, 'coverage': coverage,
              'coverage_threshold_dbm': radio.COVERAGE_THRESHOLD_DBM,
              'radio_config': radio.snapshot(), 'network': net}
    print(f'Coverage: {coverage:.2%}; mean throughput satisfaction: '
          f'{net["network_summary"]["mean_satisfaction"]:.2%}; clients meeting demand: '
          f'{net["network_summary"]["demand_met_ratio"]:.2%}')
    output_dir = root / 'test_results' if output_dir is None else Path(output_dir)
    plot_prediction_results(walls, forbidden, total_rssi, report,
                            output_dir=output_dir, show=show)
    print(f'Topology and detailed results saved to: {output_dir.resolve()}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', dest='map_path', help='Floorplan JSON path')
    parser.add_argument('--output', dest='output_dir', help='Directory for generated plots')
    parser.add_argument('--no-show', action='store_true', help='Save figures without opening windows')
    args = parser.parse_args()
    main(map_path=args.map_path, output_dir=args.output_dir, show=not args.no_show)
