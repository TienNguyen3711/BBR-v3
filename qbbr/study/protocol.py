"""Immutable run identities and strict evidence/protocol validation."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import subprocess
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'qbbr' / 'configs'


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def source_digest():
    h = sha256()
    for path in sorted((ROOT / 'qbbr').rglob('*.py')):
        if 'tests' not in path.parts:
            h.update(str(path.relative_to(ROOT)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(data + '\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_protocol(path):
    path = Path(path).resolve()
    study = yaml.safe_load(path.read_text())
    base_path = (path.parent / study['base_protocol']).resolve()
    base = yaml.safe_load(base_path.read_text())
    study['base'] = base
    validate_protocol(study)
    return study


def validate_protocol(study):
    from qbbr.eval.state_ablation import FEATURE_SETS
    base = study['base']
    if base['control']['reward_contract'] != 'counterfactual-difference-v1':
        raise ValueError('All new study arms require the manuscript difference reward')
    if base['simulator']['reward_mode'] != 'difference':
        raise ValueError('Declared and implemented reward differ')
    if base['control']['fixed_action_count'] != 5:
        raise ValueError('Native action set must remain fixed at five')
    if base['agent']['selection']['min_logit_advantage'] != 0:
        raise ValueError('Suite deployment uses hard masked argmax; margin must be zero')
    for name in ('training_seeds', 'holdout_seeds'):
        seeds = study[name]
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(f'{name} must be unique nonnegative integers')
    if set(study['training_seeds']) & set(study['holdout_seeds']):
        raise ValueError('Training and holdout seeds overlap')
    train_episode_seeds = {s * 1_000_000 + ep for s in study['training_seeds'] for ep in range(study['episodes'])}
    if train_episode_seeds & set(study['holdout_seeds']):
        raise ValueError('Episode forcing seeds overlap holdout seeds')
    if type(study['episodes']) is not int or study['episodes'] < 1:
        raise ValueError('episodes must be positive')
    if not math.isfinite(study['duration_s']) or study['duration_s'] <= 0:
        raise ValueError('duration_s must be finite and positive')
    for key, allowed in [('variants', set(FEATURE_SETS)), ('cores', {'qa2c', 'a2c', 'qdqn'}),
                         ('directions', {'uplink', 'downlink'}), ('forcing', {'synthetic', 'replay'})]:
        if not study[key] or len(set(study[key])) != len(study[key]) or not set(study[key]) <= allowed:
            raise ValueError(f'Invalid {key}')
    if not study['locations'] or len(set(study['locations'])) != len(study['locations']):
        raise ValueError('Locations must be nonempty and unique')
    if set(study['trace_split']['train_runs']) & set(study['trace_split']['test_runs']):
        raise ValueError('Training and held-out trace runs overlap')


def jobs(study):
    for forcing in study['forcing']:
        for location in study['locations']:
            for direction in study['directions']:
                for seed in study['training_seeds']:
                    for core in study['cores']:
                        # QDQN is an estimator/history ablation, not a 126-param match.
                        for variant in (['full'] if core == 'qdqn' else study['variants']):
                            yield dict(forcing=forcing, location=location, direction=direction,
                                       seed=seed, core=core, variant=variant)


def job_id(job):
    return '__'.join(str(job[k]) for k in ('forcing', 'location', 'direction', 'core', 'variant', 'seed'))
