"""Independent experiment directories and model configuration snapshots."""
import json
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
    metadata = {'schema_version': 2, 'environment_version': settings['ENVIRONMENT_VERSION'], 'run_id': run_dir.name,
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


def model_snapshot(model_path):
    """Read the snapshot for a final model or an intermediate checkpoint."""
    path = Path(model_path)
    for directory in (path.parent, path.parent.parent):
        metadata = directory/'config.json'
        if metadata.is_file():
            with metadata.open(encoding='utf-8') as stream:
                return json.load(stream)
    return None


def latest_model(model_dir):
    """Select completed models for this environment only."""
    root = project_path(model_dir)
    candidates = []
    for metadata in root.glob('*/config.json'):
        try:
            with metadata.open(encoding='utf-8') as stream:
                settings = json.load(stream)
            if settings.get('environment_version') != config.ENVIRONMENT_VERSION:
                continue
            model = metadata.parent/settings['artifacts']['model']
        except (OSError, ValueError, KeyError, TypeError):
            # A new/interrupted run may not have a complete snapshot yet.
            continue
        if model.is_file():
            candidates.append((model.stat().st_mtime_ns, str(model), model))
    if candidates:
        return max(candidates)[2]
    raise FileNotFoundError(f'No completed {config.ENVIRONMENT_VERSION} model in {root}; train a new model')
