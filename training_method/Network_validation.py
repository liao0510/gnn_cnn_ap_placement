"""Fixed, deterministic validation scenarios; never used for gradient updates."""
import json
import numpy as np
import config_radio as radio
from method.Cal_average_gpu import Cal_db_gpu
from training_method.Deployment import read_floorplan, evaluate_deployment
from training_method.Experiment import project_path


def prepare_validation(files, settings):
    """Use a private RNG, so validation cannot consume training random draws."""
    rng = np.random.default_rng(settings['VALIDATION_RANDOM_SEED'])
    files = sorted(path.resolve() for path in files)
    directory = settings['VALIDATION_MAP_DIR']
    if directory is not None:
        validation_files = sorted(project_path(directory).resolve().glob('*.json'))
        if not validation_files:
            raise ValueError('VALIDATION_MAP_DIR has no JSON maps')
        overlap = set(files) & set(validation_files)
        # An explicit subdirectory of MAP_DIR is fine; identical files are not.
        if overlap:
            raise ValueError('Training and explicit validation map paths must not overlap')
        training_files, scope = files, 'held_out_maps'
    elif len(files) > 1:
        count = min(len(files) - 1, max(1, int(np.ceil(len(files) * settings['VALIDATION_MAP_FRACTION']))))
        chosen = set(int(i) for i in rng.choice(len(files), size=count, replace=False))
        validation_files = [path for i, path in enumerate(files) if i in chosen]
        training_files = [path for i, path in enumerate(files) if i not in chosen]
        scope = 'held_out_maps'
    else:
        training_files, validation_files = files, files
        scope = 'same_map_fixed_clients_only'
    scenarios = []
    for index in range(settings['VALIDATION_EPISODES']):
        path = validation_files[index % len(validation_files)]
        walls, forbidden = read_floorplan(path, settings['AP_AMOUNT'] + settings['N_CLIENT'])
        ids = rng.choice(np.flatnonzero((walls == 0) & (forbidden == 0)),
                         size=settings['N_CLIENT'], replace=False)
        clients = [[int(i % walls.shape[1]), int(i // walls.shape[1])] for i in ids]
        scenarios.append({'scenario': index + 1, 'map_path': str(path),
                          'clients': clients, 'demands_mbps': [float(settings['CLIENT_DEMAND'])] * len(clients)})
    manifest = {'seed': settings['VALIDATION_RANDOM_SEED'], 'scope': scope,
                'training_maps': [str(path) for path in training_files],
                'validation_maps': [str(path) for path in validation_files], 'scenarios': scenarios,
                'selection': 'mean final weighted objective; strict improvement; earliest wins ties',
                'weights': {key: settings[key] for key in (
                    'COVERAGE_REWARD_WEIGHT', 'THROUGHPUT_REWARD_WEIGHT', 'CONNECTIVITY_REWARD_WEIGHT')}}
    return training_files, manifest


def evaluate_validation(policy, manifest, settings, radio_settings):
    """Greedy final deployments, with fixed maps/clients and no random sampling."""
    cfg = radio.channel_config(radio_settings)
    trials, current_path = [], None
    for scenario in manifest['scenarios']:
        path = project_path(scenario['map_path'])
        if path != current_path:
            walls, forbidden = read_floorplan(path, settings['AP_AMOUNT'] + settings['N_CLIENT'])
            simulator = Cal_db_gpu(walls=walls, frequency_ghz=radio_settings['ACCESS_FREQUENCY_GHZ'],
                tx_power=radio_settings['TX_POWER_DBM'], grid_size=radio_settings['GRID_SIZE_METERS'],
                ray_steps=settings['RSSI_RAY_STEPS'], row_chunk=settings['RSSI_ROW_CHUNK'])
            current_path = path
        result = evaluate_deployment(walls, forbidden, scenario['clients'], scenario['demands_mbps'],
            settings['AP_AMOUNT'], simulator, cfg, radio_settings, settings['MAX_NODES'], policy=policy,
            feature_settings=settings)
        trials.append({'scenario': scenario['scenario'], **result})
    metrics = {key: (float(np.mean([row['metrics'][key] for row in trials
                                   if row['metrics'][key] is not None]))
                     if any(row['metrics'][key] is not None for row in trials) else None)
               for key in trials[0]['metrics']}
    # With zero clients the throughput term is zero, never an artificial 100%.
    metrics['score'] = (settings['COVERAGE_REWARD_WEIGHT'] * metrics['coverage']
                        + settings['THROUGHPUT_REWARD_WEIGHT'] * (metrics['mean_satisfaction'] or 0.0)
                        + settings['CONNECTIVITY_REWARD_WEIGHT'] * metrics['mesh_connectivity'])
    if not np.isfinite(metrics['score']):
        raise ValueError('Validation score must be finite')
    return metrics, trials


def write_json(path, data):
    with path.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
