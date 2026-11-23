"""Fit and select a raw Dplus-Dminus direction using only paired pressure states."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from ..io import read_json, write_json, write_jsonl
from .run import COLLECTION_SCHEMA, METHOD, STAGES, configured_layers, dev_items, pressure_states, validate_record
EPS = 1e-12
FIT_FILES = ('summary.json', 'subspace.pt', 'crossfit_axes.pt', 'crossfit.jsonl')

def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= EPS:
        raise ValueError('Direction must be finite and nonzero')
    return vector / norm

def fit_direction(contrasts: np.ndarray) -> np.ndarray:
    contrasts = np.asarray(contrasts, dtype=np.float32)
    if contrasts.ndim != 2 or not contrasts.shape[0] or (not contrasts.shape[1]):
        raise ValueError('Expected a nonempty [items, hidden] contrast matrix')
    norms = np.linalg.norm(contrasts, axis=1, keepdims=True)
    if not np.isfinite(contrasts).all() or not np.isfinite(norms).all() or np.any(norms <= EPS):
        raise ValueError('Training contrasts must be finite and nonzero')
    return unit((contrasts / norms).mean(axis=0))

def make_folds(pair_ids: list[str], categories: list[str], seed: int) -> list[int]:
    plan_path = Path(__file__).resolve().parents[4] / 'data/splits/crossfit_folds.json'
    if plan_path.is_file():
        for plan in read_json(plan_path):
            saved = {row['pair_id']: row for row in plan['items']}
            if plan['seed'] == seed and set(saved) == set(pair_ids) and all((saved[pid]['category'] == category for pid, category in zip(pair_ids, categories))):
                return [saved[pid]['fold'] for pid in pair_ids]
    groups = defaultdict(list)
    for index, category in enumerate(categories):
        groups[category].append(index)
    folds = [-1] * len(pair_ids)
    for category in sorted(groups):
        ordered = sorted(groups[category], key=lambda i: pair_ids[i])
        np.random.default_rng(seed).shuffle(ordered)
        for offset, index in enumerate(ordered):
            folds[index] = offset % 3
    return folds

def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:

    def group(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {'n': len(values), 'mean_signed_cosine': float(np.mean([r['signed_cosine'] for r in values])), 'positive_fraction': sum((r['signed_coefficient'] > 0 for r in values)) / len(values)}
    return {'pooled': group(rows), 'by_category': {category: group([r for r in rows if r['category'] == category]) for category in sorted({r['category'] for r in rows})}}

def fit_states(states: np.ndarray, pair_ids: list[str], categories: list[str], layers: list[int], *, seed: int) -> dict[str, Any]:
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 4 or states.shape[:3] != (len(pair_ids), 2, len(layers)) or states.shape[-1] < 1 or (not pair_ids) or (len(categories) != len(pair_ids)) or (len(set(pair_ids)) != len(pair_ids)) or (not layers) or (len(set(layers)) != len(layers)) or any((layer < 1 for layer in layers)):
        raise ValueError('Expected unique items/layers and [items, Dplus/Dminus, layers, hidden] states')
    if not np.isfinite(states).all():
        raise ValueError('Pressure states contain non-finite values')
    if any((n < 3 or n % 3 for n in Counter(categories).values())):
        raise ValueError('Each category must divide into three nonempty folds')
    differences = states[:, 0] - states[:, 1]
    fold_by_item = make_folds(pair_ids, categories, seed)
    rows, crossfit = ([], {})
    for layer_index, layer in enumerate(layers):
        vectors = differences[:, layer_index]
        crossfit[layer] = {}
        for fold in range(3):
            held = [i for i, f in enumerate(fold_by_item) if f == fold]
            training = [i for i, f in enumerate(fold_by_item) if f != fold]
            # Every held-out item is scored against the same training-only axis.
            axis = fit_direction(vectors[training])
            for index in held:
                rows.append({'pair_id': pair_ids[index], 'category': categories[index], 'fold': fold, 'layer': layer, 'signed_coefficient': float(np.dot(vectors[index], axis)), 'signed_cosine': float(np.dot(unit(vectors[index]), axis))})
            crossfit[layer][fold] = {'held_out_pair_ids': [pair_ids[i] for i in held], 'training_pair_ids': [pair_ids[i] for i in training], 'categories': {'pooled': axis}}
    layer_metrics = {layer: metrics([r for r in rows if r['layer'] == layer]) for layer in layers}
    selected = max(layers, key=lambda layer: (layer_metrics[layer]['pooled']['mean_signed_cosine'], -layer))
    selected_vectors = differences[:, layers.index(selected)]
    categories_out = {'pooled': {'selective': fit_direction(selected_vectors), 'pair_ids': list(pair_ids)}}
    return {'summary': {'schema_version': 'deepseek_paired_steering_fit_pooled_v1', **METHOD, 'seed': seed, 'items': len(pair_ids), 'counts_by_category': dict(Counter(categories)), 'selected_layer': selected, 'selection_rule': 'maximize pooled held-out signed cosine; smaller layer', 'layer_metrics': {str(layer): value for layer, value in layer_metrics.items()}, 'selected_layer_metrics': layer_metrics[selected]}, 'categories': categories_out, 'folds': crossfit[selected], 'rows': rows}

def validate_fit(config: dict[str, Any], output: Path) -> dict[str, Any]:
    items = dev_items(config)
    manifest = read_json(output / 'manifest.json')
    if manifest['schema_version'] != COLLECTION_SCHEMA or manifest['pair_ids'] != [item.pair_id for item in items] or manifest['stage_names'] != list(STAGES) or (manifest['method'] != METHOD) or (manifest['layer_numbers'] != configured_layers(config)):
        raise RuntimeError('Saved fit does not match the configured items, layers, or method')
    for name in FIT_FILES:
        if not (output / name).is_file():
            raise FileNotFoundError(output / name)
    return read_json(output / 'summary.json')


def run_extraction(config: dict[str, Any], output: Path) -> dict[str, Any]:
    import torch
    items = dev_items(config)
    layers = configured_layers(config)
    pair_ids = [item.pair_id for item in items]
    manifest = read_json(output / 'manifest.json')
    if manifest['schema_version'] != COLLECTION_SCHEMA or manifest['pair_ids'] != pair_ids or manifest['stage_names'] != list(STAGES) or (manifest['layer_numbers'] != layers) or (manifest['method'] != METHOD):
        raise RuntimeError('Collection does not match the paired experiment')
    states = []
    for item in items:
        validate_record(read_json(output / 'records' / f'{item.pair_id}.json'), item)
        payload = torch.load(output / 'activations' / f'{item.pair_id}.pt', map_location='cpu', weights_only=True)
        states.append(pressure_states(payload, item, layers))
    result = fit_states(np.stack(states), pair_ids, [i.category for i in items], layers, seed=int(config['seed']))
    layer = result['summary']['selected_layer']
    subspace = {'schema_version': 'deepseek_paired_steering_subspace_pooled_v1', 'selected_layer': layer, 'sign': 'positive direction is Dplus minus Dminus', 'method': METHOD, 'categories': {category: {**value, 'selective': torch.from_numpy(value['selective'].copy())} for category, value in result['categories'].items()}}
    crossfit = {'schema_version': 'deepseek_paired_steering_crossfit_pooled_v1', 'selected_layer': layer, 'pair_ids': pair_ids, 'folds': {fold: {**value, 'categories': {category: torch.from_numpy(axis.copy()) for category, axis in value['categories'].items()}} for fold, value in result['folds'].items()}}
    for name, value in (('subspace.pt', subspace), ('crossfit_axes.pt', crossfit)):
        temporary = output / f'{name}.tmp'
        torch.save(value, temporary)
        temporary.replace(output / name)
    write_jsonl(output / 'crossfit.jsonl', result['rows'])
    write_json(output / 'summary.json', result['summary'])
    return result['summary']
