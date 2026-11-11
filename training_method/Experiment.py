"""Independent experiment directories and model configuration snapshots."""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT/path


def create_run(model_dir, prefix, model_filename, settings, radio_settings):
    """Reserve a fresh directory atomically; never reuse another run's files."""
    root = project_path(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    if Path(model_filename).name != model_filename or not model_filename.endswith('.keras'):
        raise ValueError('MODEL_FILENAME must be a .keras filename without directories')
    if not prefix or any(char in prefix for char in '/\\:'):
        raise ValueError('RUN_PREFIX must be a nonempty directory-name prefix')
    created = datetime.now(timezone(timedelta(hours=8)))
    while True:
        run_dir = root/f'{prefix}_{created:%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}'
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            continue
    metadata = {'schema_version': 2, 'environment_version': settings['ENVIRONMENT_VERSION'],
                'model_architecture': settings['MODEL_ARCHITECTURE'], 'run_id': run_dir.name,
                'created_at': created.isoformat(), 'timezone': 'Asia/Taipei',
                'training': settings, 'radio': radio_settings,
                'artifacts': {'model': model_filename, 'history': 'training_history.npz',
                              'plot': 'training_history.png', 'checkpoints': 'checkpoints'}}
    with (run_dir/'config.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    return run_dir


def save_model(model, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite model: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def save_best_model(model, run_dir, filename, validation):
    """Atomically replace only this fresh run's designated best checkpoint."""
    run_dir = Path(run_dir)
    with (run_dir / 'config.json').open(encoding='utf-8') as stream:
        metadata = json.load(stream)
    if (Path(filename).name != filename or not filename.endswith('.keras')
            or filename == metadata['artifacts']['model']):
        raise ValueError('BEST_MODEL_FILENAME must be a distinct .keras filename')
    temporary = run_dir / f'.best_{uuid4().hex}.keras'
    save_model(model, temporary)
    os.replace(str(temporary), str(run_dir / filename))
    metadata['artifacts']['best_model'] = filename
    metadata['artifacts']['validation_history'] = 'validation_history.json'
    metadata['artifacts']['validation_scenarios'] = 'validation_scenarios.json'
    metadata['best_validation'] = validation
    temporary_metadata = run_dir / f'.config_{uuid4().hex}.json'
    with temporary_metadata.open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(str(temporary_metadata), str(run_dir / 'config.json'))


def model_snapshot(model_path):
    """Read the snapshot for a final model or an intermediate checkpoint."""
    path = Path(model_path)
    for directory in (path.parent, path.parent.parent):
        metadata = directory/'config.json'
        if metadata.is_file():
            with metadata.open(encoding='utf-8') as stream:
                return json.load(stream)
    return None


def validate_model_snapshot(snapshot):
    """Reject weights from a different physical environment or architecture."""
    if snapshot is None or snapshot.get('environment_version') != config.ENVIRONMENT_VERSION:
        raise ValueError('Model environment is incompatible with AP-AP SNR / AP-client SINR; '
                         'retrain in the new environment. Existing model files are retained.')
    if snapshot.get('model_architecture') != config.MODEL_ARCHITECTURE:
        found = snapshot.get('model_architecture', 'legacy CNN')
        raise ValueError(f'Model architecture {found} is incompatible with {config.MODEL_ARCHITECTURE}; '
                         'retrain using GNN_CNN_DRL_train.py. Existing model files are retained.')


def latest_model(model_dir):
    """Latest completed compatible run: prefer its best validation checkpoint."""
    root = project_path(model_dir)
    candidates = []
    for metadata in root.glob('*/config.json'):
        try:
            with metadata.open(encoding='utf-8') as stream:
                settings = json.load(stream)
            validate_model_snapshot(settings)
            model = metadata.parent/settings['artifacts']['model']
        except (OSError, ValueError, KeyError, TypeError):
            # A new/interrupted run may not have a complete snapshot yet.
            continue
        if model.is_file():
            best = metadata.parent / settings['artifacts'].get('best_model', model.name)
            candidates.append((model.stat().st_mtime_ns, str(model), best if best.is_file() else model))
    if candidates:
        return max(candidates)[2]
    raise FileNotFoundError(f'No completed {config.MODEL_ARCHITECTURE} model for '
                            f'{config.ENVIRONMENT_VERSION} in {root}; '
                            'retrain using GNN_CNN_DRL_train.py in the SINR environment')
