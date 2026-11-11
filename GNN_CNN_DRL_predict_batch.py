"""Paired final-deployment evaluation. Run: python GNN_CNN_DRL_predict_batch.py"""

# ====================== 可調整設定 ======================
NUM_PREDICTIONS = 100
RANDOM_MAP_EACH_RUN = False       # False: 同張地圖；True: 隨機換圖，避免連續重複
MAP_PATH = 'maps_200_200/map_0000.json'
MAP_DIRECTORY = 'maps_200_200'    # True 時從此資料夾選 JSON，至少需要兩張圖
MODEL_PATH = None               # None: 沿用 config；未指定則取最新完成訓練的 best model
AP_COUNT = 5
CLIENT_COUNT = 10
CLIENT_DEMAND_MBPS = 100.0
CLIENT_DEMANDS = None           # 可指定每位 client 的 demand；None 使用上面的統一需求
RANDOM_SEED = 42                # 固定可重現，但每次 trial 仍抽新 client；None 不固定
BOOTSTRAP_SAMPLES = 2000
OUTPUT_DIRECTORY = 'batch_prediction_results'
SHOW_PLOTS = False              # 一律儲存 PNG/PDF；True 另開圖表視窗
# ======================================================

import csv
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import numpy as np
import tensorflow as tf
import config
import config_radio as radio
from method.Cal_average_gpu import Cal_db_gpu
from method.Graph_visualize import plot_batch_prediction_analysis
from training_method.Deployment import json_safe, read_floorplan, evaluate_deployment
from training_method.Experiment import latest_model, model_snapshot, project_path, validate_model_snapshot
from training_method.GNN_CNN_Network import build_network
from training_method.Network_policy import build_policy_model
from training_method.Network_state import CNN_CHANNELS, NODE_FEATURES
from training_method.Prediction_analysis import ANALYSIS_METRICS, METHOD_LABELS, summarize_trials


def write_csv(path, fieldnames, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_batch_predictions():
    for name, value, minimum in (('NUM_PREDICTIONS', NUM_PREDICTIONS, 1),
                                 ('AP_COUNT', AP_COUNT, 1), ('CLIENT_COUNT', CLIENT_COUNT, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not isinstance(RANDOM_MAP_EACH_RUN, bool):
        raise ValueError('RANDOM_MAP_EACH_RUN must be True or False')
    if type(BOOTSTRAP_SAMPLES) is not int or BOOTSTRAP_SAMPLES < 100:
        raise ValueError('BOOTSTRAP_SAMPLES must be an integer >= 100')
    if config.MAX_NODES < AP_COUNT + CLIENT_COUNT:
        raise ValueError('MAX_NODES must fit APs and clients')
    demands = ([float(CLIENT_DEMAND_MBPS)] * CLIENT_COUNT if CLIENT_DEMANDS is None
               else [float(value) for value in CLIENT_DEMANDS])
    if len(demands) != CLIENT_COUNT or any(not np.isfinite(value) or value < 0 for value in demands):
        raise ValueError('Provide one finite, nonnegative demand per client')
    paths = (sorted(project_path(MAP_DIRECTORY).glob('*.json')) if RANDOM_MAP_EACH_RUN
             else [project_path(MAP_PATH)])
    if not paths or (RANDOM_MAP_EACH_RUN and len(paths) < 2):
        raise ValueError('Random-map mode requires at least two JSON maps; fixed mode requires one map')
    # Validate before reserving a result directory or running an expensive model.
    for path in paths:
        read_floorplan(path, AP_COUNT + CLIENT_COUNT)
    requested = config.PREDICT_MODEL_PATH if MODEL_PATH is None else MODEL_PATH
    model_path = latest_model(config.MODEL_DIR) if requested is None else project_path(requested)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    saved = model_snapshot(model_path)
    validate_model_snapshot(saved)
    radio_settings = saved['radio']
    training_settings = saved['training']
    cfg = radio.channel_config(radio_settings)
    # Prevent TF from reserving all VRAM needed by the CuPy propagation model.
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # The caller may already have initialized the device.
    model = build_network(CNN_CHANNELS, NODE_FEATURES)
    model.load_weights(str(model_path))
    policy = build_policy_model(model)
    seed_sequence = np.random.SeedSequence(RANDOM_SEED)
    map_seed, client_seed, baseline_seed, analysis_seed = seed_sequence.spawn(4)
    map_rng, client_rng, baseline_rng = [np.random.default_rng(seed)
                                         for seed in (map_seed, client_seed, baseline_seed)]
    analysis_entropy = int(analysis_seed.generate_state(1)[0])
    created = datetime.now(timezone(timedelta(hours=8)))
    output = project_path(OUTPUT_DIRECTORY) / f'batch_{created:%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        'created_at': created.isoformat(), 'timezone': 'Asia/Taipei',
        'num_predictions': NUM_PREDICTIONS, 'random_map_each_run': RANDOM_MAP_EACH_RUN,
        'map_candidates': [str(path) for path in paths], 'model_path': str(model_path),
        'training_run_id': saved['run_id'], 'environment_version': config.ENVIRONMENT_VERSION,
        'model_architecture': saved['model_architecture'],
        'ap_count': AP_COUNT, 'client_count': CLIENT_COUNT, 'client_demands_mbps': demands,
        'seed_entropy': seed_sequence.entropy, 'bootstrap_samples': BOOTSTRAP_SAMPLES,
        'radio': radio_settings, 'training_snapshot': training_settings,
        'assumptions': {'mesh_link_metric': 'SNR', 'access_link_metric': 'SINR',
                        'access_interference_model': radio_settings['ACCESS_INTERFERENCE_MODEL'],
                        'all_deployed_aps_always_transmit': True,
                        'interferers_include_nonserving_mesh_aps': True,
                        'sinr_independent_of_ru_position': True,
                        'constant_power_spectral_density': True},
        'methods': {'gnn_cnn': 'sequential masked argmax, same as single predict',
                    'random': 'uniform legal AP placement without replacement; no mesh constraint'},
        'paired_design': 'identical map, clients, demands, AP budget, radio and OFDMA per trial',
        'metric_definitions': {
            'coverage': 'covered non-wall cells / all non-wall cells, serving mesh only',
            'mean_satisfaction': 'mean min(actual throughput / demand, 1); zero-demand client = 1',
            'demand_met_ratio': 'fraction of clients whose actual throughput meets demand',
            'mesh_connectivity': 'largest connected AP component size / all APs; not step conn reward',
            'mesh_connected': '1 if all APs belong to a single component, else 0',
            'client_throughput_p05_mbps': '5th percentile within each trial, including unassociated zeros',
            'satisfaction_fairness': 'Jain index of achievement ratios for positive-demand clients; all-zero=0',
            'client_ecdf': 'descriptive pooled client values; confidence intervals resample trials instead',
        },
        'references': ['https://www.cs.wustl.edu/~jain/papers/ftp/fairness.pdf',
                       'https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/bootplot.htm'],
    }
    with (output / 'evaluation_config.json').open('w', encoding='utf-8') as stream:
        json.dump(json_safe(metadata), stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'Model: {model_path}\nBatch results: {output}')
    trials, current_path, simulator = [], None, None
    with (output / 'trials.jsonl').open('w', encoding='utf-8') as stream:
        for index in range(NUM_PREDICTIONS):
            candidates = [path for path in paths if path != current_path] if RANDOM_MAP_EACH_RUN else paths
            path = candidates[int(map_rng.integers(len(candidates)))] if RANDOM_MAP_EACH_RUN else paths[0]
            if path != current_path:
                walls, forbidden = read_floorplan(path, AP_COUNT + CLIENT_COUNT)
                simulator = Cal_db_gpu(walls=walls,
                    frequency_ghz=radio_settings['ACCESS_FREQUENCY_GHZ'], tx_power=radio_settings['TX_POWER_DBM'],
                    grid_size=radio_settings['GRID_SIZE_METERS'], ray_steps=training_settings['RSSI_RAY_STEPS'],
                    row_chunk=training_settings['RSSI_ROW_CHUNK'])
                current_path = path
            ids = client_rng.choice(np.flatnonzero((walls == 0) & (forbidden == 0)),
                                    size=CLIENT_COUNT, replace=False)
            clients = [(int(i % walls.shape[1]), int(i // walls.shape[1])) for i in ids]
            trial = {'trial': index + 1, 'map_path': str(path),
                     'client_positions': clients, 'client_demands_mbps': demands}
            for method in METHOD_LABELS:
                trial[method] = evaluate_deployment(walls, forbidden, clients, demands, AP_COUNT,
                    simulator, cfg, radio_settings, config.MAX_NODES,
                    policy=policy if method == 'gnn_cnn' else None,
                    rng=baseline_rng if method == 'random' else None,
                    feature_settings=training_settings)
            trials.append(trial)
            stream.write(json.dumps(json_safe(trial), ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            ap_coordinates = ','.join(f'({x},{y})' for x, y in trial['gnn_cnn']['ap_positions'])
            print(f'{index + 1}/{NUM_PREDICTIONS} {path.name} AP:{ap_coordinates}')
            for method, label in (('gnn_cnn', 'GNN'), ('random', 'random')):
                metrics = trial[method]['metrics']
                satisfaction = metrics['mean_satisfaction']
                satisfaction_text = 'N/A' if satisfaction is None else f'{satisfaction:.0%}'
                print(f'{label:<6}: coverage = {metrics["coverage"]:.0%}, '
                      f'satisfaction = {satisfaction_text}, mesh = {metrics["mesh_connectivity"]:.1%}')
    summary = summarize_trials(trials, BOOTSTRAP_SAMPLES, analysis_entropy)
    with (output / 'summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    keys = [key for key, _, _ in ANALYSIS_METRICS]
    write_csv(output / 'trial_metrics.csv', ['trial', 'map_path', 'method'] + keys,
        [dict(trial=trial['trial'], map_path=trial['map_path'], method=method, **trial[method]['metrics'])
         for trial in trials for method in METHOD_LABELS])
    client_rows = [dict(trial=trial['trial'], method=method, **row)
                   for trial in trials for method in METHOD_LABELS for row in trial[method]['allocation_results']]
    if client_rows:
        write_csv(output / 'client_metrics.csv', list(client_rows[0]), client_rows)
    summary_rows = []
    for group, metrics in list(summary['methods'].items()) + [
            ('gnn_minus_random', summary['paired_difference_gnn_minus_random'])]:
        for key, stats in metrics.items():
            interval = stats['mean_ci95'] or [None, None]
            summary_rows.append(dict(method=group, metric=key, n=stats['n'], mean=stats['mean'],
                std=stats['std'], median=stats['median'], ci95_low=interval[0], ci95_high=interval[1],
                win_rate=stats.get('win_rate'), tie_rate=stats.get('tie_rate')))
    write_csv(output / 'summary.csv', list(summary_rows[0]), summary_rows)
    plot_batch_prediction_analysis(trials, summary, output, show=SHOW_PLOTS)
    print(f'Finished {len(trials)} paired trials. Results: {output}')
    return {'output_dir': str(output), 'summary': summary, 'trials': trials}


if __name__ == '__main__':
    run_batch_predictions()
