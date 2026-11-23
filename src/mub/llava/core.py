"""Shared paths and small file helpers."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'data/mub'
RESULTS = ROOT / 'results/mub/llava'
CONFIG = HERE / 'config.json'
MODEL = 'llava'
BACKEND = 'llava_next'
LETTERS = 'ABCD'


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    return [json.loads(s) for s in resolve(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def write_json(path, value):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, values):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values), encoding='utf-8')


def ensure_manifest(output, value):
    path = Path(output) / 'manifest.json'
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Existing run has different model/data/config: {output}; use another --output')
        return
    write_json(path, value)


def config_file(path, model_path=None):
    config = read_json(path)
    if model_path:
        config['model_path'] = str(model_path)
    return config
