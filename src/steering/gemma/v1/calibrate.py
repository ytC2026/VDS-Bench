"""Cross-fitted V1 text-answer sign check at ±10% of each hidden-state norm."""
from __future__ import annotations
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any
from PIL import Image
from ..data import PressurePair, load_pairs
from ..io import read_json, resolve_path, write_json
from .protocol import initial_messages, post_pressure_messages
from ..protocol import role_option_texts
from .scoring import compact_score
from .selection import select_sign
STAGES = ('initial', 'Dplus', 'Dminus')

def _compact(scored: dict[str, Any]) -> dict[str, Any]:
    result, _states = compact_score(scored)
    result['intervention_metrics'] = scored['intervention_metrics']
    return result

def calibrate_item(item: PressurePair, baseline_record: dict[str, Any], fold: int, engine: Any, layer: int, direction: Any, ratio: float, terminal_token_id: int | None) -> dict[str, Any]:
    candidates = role_option_texts(item.dplus, item.dminus)
    if baseline_record['pair_id'] != item.pair_id or baseline_record['role_texts'] != candidates:
        raise RuntimeError(f'V1 baseline record mismatch: {item.pair_id}')
    initial_answer = baseline_record['initial_answer_used_in_both_branches']
    stages: dict[str, Any] = {}
    with Image.open(item.image_path) as handle:
        image = handle.convert('RGB')
    try:
        for stage, sample in (('initial', item.dplus), ('Dplus', item.dplus), ('Dminus', item.dminus)):
            messages = initial_messages(sample, image) if stage == 'initial' else post_pressure_messages(sample, image, initial_answer)
            processed = engine.process_messages(messages)
            prepared = engine.prepare(processed)
            stages[stage] = {'baseline': baseline_record['stages'][stage]}
            for condition, signed_ratio in (('negative', -ratio), ('positive', ratio)):
                scored = engine.score_actual_options(processed, candidates, prepared=prepared, terminal_token_id=terminal_token_id, intervention_layer=layer, intervention_direction=direction, intervention_ratio=signed_ratio)
                stages[stage][condition] = _compact(scored)
    finally:
        image.close()
    return {'pair_id': item.pair_id, 'category': item.category, 'fold': fold, 'selected_layer': layer, 'ratio': ratio, 'baseline_initial_answer_used_in_both_branches': initial_answer, 'stages': stages}

def select_from_saved_records(config: dict[str, Any], v1_output: Path) -> dict[str, Any]:
    """Re-orient an existing dev90 run without loading the model or torch."""
    items, counts = load_pairs(config)
    if len(items) != 90 or sorted(Counter((item.category for item in items)).values()) != [30, 30, 30]:
        raise RuntimeError(f'Expected dev90, got {counts}')
    pair_ids = [item.pair_id for item in items]
    v1_manifest = read_json(v1_output / 'manifest.json')
    calibration_manifest = read_json(v1_output / 'calibration' / 'manifest.json')
    if v1_manifest['pair_ids'] != pair_ids or calibration_manifest['pair_ids'] != pair_ids:
        raise RuntimeError('Saved calibration is incompatible with the fixed V1 run')
    ratio = float(config['dev_sign_ratio'])
    layer = int(calibration_manifest['selected_layer'])
    if calibration_manifest['ratio'] != ratio:
        raise RuntimeError('Saved calibration ratio differs from config')
    rows = [read_json(v1_output / 'calibration' / 'records' / f'{pair_id}.json') for pair_id in pair_ids]
    for item, row in zip(items, rows):
        if row['pair_id'] != item.pair_id or row['category'] != item.category or row['fold'] != calibration_manifest['fold_by_pair'][item.pair_id] or (row['selected_layer'] != layer) or (row['ratio'] != ratio):
            raise RuntimeError(f'Invalid saved calibration row: {item.pair_id}')
    selection = select_sign(rows, ratio=ratio)
    selection['selected_layer'] = layer
    write_json(v1_output / 'calibration' / 'selection_margin.json', selection)
    return selection

def run_calibration(config: dict[str, Any], v1_output: Path) -> dict[str, Any]:
    import torch
    if int(config['outer_folds']) != 3:
        raise ValueError('Paired extraction and calibration require three folds')
    ratio = float(config['dev_sign_ratio'])
    if not 0 < ratio < 1:
        raise ValueError('dev_sign_ratio must be in (0, 1)')
    items, counts = load_pairs(config)
    if len(items) != 90 or sorted(Counter((item.category for item in items)).values()) != [30, 30, 30]:
        raise RuntimeError(f'Expected dev90, got {counts}')
    v1_manifest = read_json(v1_output / 'manifest.json')
    if v1_manifest['pair_ids'] != [item.pair_id for item in items]:
        raise RuntimeError('V1 manifest does not match the fixed dev90 split')
    axes_path = v1_output / 'crossfit_axes.pt'
    subspace_path = v1_output / 'subspace.pt'
    axes = torch.load(axes_path, map_location='cpu', weights_only=True)
    subspace = torch.load(subspace_path, map_location='cpu', weights_only=True)
    layer = int(subspace['selected_layer'])
    if axes['selected_layer'] != layer or axes['pair_ids'] != v1_manifest['pair_ids']:
        raise RuntimeError('Cross-fit directions do not match the V1 subspace')
    fold_by_pair = {}
    for fold, value in axes['folds'].items():
        if set(value['held_out_pair_ids']) & set(value['training_pair_ids']):
            raise RuntimeError(f'Fold {fold} training/held-out pair IDs overlap')
        for pair_id in value['held_out_pair_ids']:
            if pair_id in fold_by_pair:
                raise RuntimeError(f'Pair {pair_id} appears in multiple held-out folds')
            fold_by_pair[pair_id] = int(fold)
    if set(fold_by_pair) != {item.pair_id for item in items}:
        raise RuntimeError('Held-out folds do not cover dev90 exactly once')
    output = v1_output / 'calibration'
    records_dir = output / 'records'
    records_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 'gemma_paired_text_sign_calibration_v1', 'pair_ids': [item.pair_id for item in items], 'fold_by_pair': fold_by_pair, 'selected_layer': layer, 'ratio': ratio, 'protocol': 'V1 unlabeled actual-option text', 'history': 'saved unperturbed V1 initial option text'}
    manifest_path = output / 'manifest.json'
    write_json(manifest_path, manifest)
    pending = items
    if pending:
        from ..adapter import load_engine
        engine = load_engine(config)
        try:
            if layer > engine.decoder_depth:
                raise ValueError(f'Selected layer {layer} exceeds depth {engine.decoder_depth}')
            for index, item in enumerate(pending, 1):
                fold = fold_by_pair[item.pair_id]
                direction = axes['folds'][fold]['categories']['pooled']
                baseline = read_json(v1_output / 'records' / f'{item.pair_id}.json')
                row = calibrate_item(item, baseline, fold, engine, layer, direction, ratio, config.get('terminal_token_id'))
                write_json(records_dir / f'{item.pair_id}.json', row)
                print(f'[calibration {index}/{len(pending)}] {item.pair_id}', flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    rows = [read_json(records_dir / f'{item.pair_id}.json') for item in items]
    for row in rows:
        if row['fold'] != fold_by_pair[row['pair_id']] or row['selected_layer'] != layer or row['ratio'] != ratio:
            raise RuntimeError(f"Invalid resumed calibration row: {row['pair_id']}")
    return select_from_saved_records(config, v1_output)
