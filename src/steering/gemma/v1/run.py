"""Run V1 paired-state collection, direction fitting and sign calibration."""
from __future__ import annotations
import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'steering.gemma.v1'
from ..data import PressurePair, load_pairs
from ..io import ROOT, read_json, resolve_path, write_json
from .protocol import initial_messages, initial_prompt, post_pressure_messages, pressure_prompt
from ..protocol import question_stem, role_option_texts
from .scoring import compact_score
STAGES = ('Dplus', 'Dminus')
COLLECTION_SCHEMA = 'gemma_paired_steering_collection_v2'
METHOD = {'direction_formula': 'h_Dplus - h_Dminus', 'direction_scope': 'pooled', 'initial_hidden_state_used_for_fit': False, 'pressure_orthogonalization': False, 'normalization': 'unit(mean(unit(per-item contrast))) across all fitting pairs', 'layer_selection': 'three-fold dev-only pooled held-out signed cosine', 'sign_calibration': 'dev-only GOLD-minus-target margin under both intervention signs'}


def dev_items(config: dict[str, Any]) -> list[PressurePair]:
    if int(config['outer_folds']) != 3:
        raise ValueError('Paired layer selection requires three folds')
    items, counts = load_pairs(config)
    if len(items) != 90 or sorted(Counter((i.category for i in items)).values()) != [30, 30, 30]:
        raise ValueError(f'Expected dev90 with 30 items/category: {counts}')
    return items

def configured_layers(config: dict[str, Any]) -> list[int]:
    layers = sorted((int(value) for value in config['decoder_layers']))
    if not layers or layers[0] < 1 or len(set(layers)) != len(layers):
        raise ValueError('decoder_layers must contain distinct positive layer numbers')
    return layers

def collect_item(item: PressurePair, engine: Any, layers: list[int], terminal_token_id: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = role_option_texts(item.dplus, item.dminus)
    stages, captured, diagnostics = ({}, {}, {})
    with Image.open(item.image_path) as handle:
        image = handle.convert('RGB')
    try:
        processed = engine.process_messages(initial_messages(item.dplus, image))
        diagnostics['initial_input_tokens'] = int(processed['input_ids'].shape[1])
        scored = engine.score_actual_options(processed, candidates, capture_layers=(), terminal_token_id=terminal_token_id)
        stages['initial'], unused_states = compact_score(scored)
        if unused_states:
            raise RuntimeError('Initial forward unexpectedly captured hidden states')
        stages['initial']['prompt_text'] = initial_prompt(item.dplus)
        initial_answer = stages['initial']['answer_text']
        for stage, sample in (('Dplus', item.dplus), ('Dminus', item.dminus)):
            processed = engine.process_messages(post_pressure_messages(sample, image, initial_answer))
            diagnostics[f'{stage}_input_tokens'] = int(processed['input_ids'].shape[1])
            scored = engine.score_actual_options(processed, candidates, capture_layers=layers, terminal_token_id=terminal_token_id)
            stages[stage], captured[stage] = compact_score(scored)
            stages[stage]['prompt_text'] = pressure_prompt(sample)
    finally:
        image.close()
    states = np.stack([np.stack([np.asarray(captured[stage][layer], dtype=np.float32) for layer in layers]) for stage in STAGES])[None, :, :, None, :]
    record = {'pair_id': item.pair_id, 'category': item.category, 'image_id': str(item.dplus['image_id']), 'question': question_stem(item.dplus['question']), 'role_texts': candidates, 'initial_answer_used_in_both_branches': initial_answer, 'behavior': {'initial_correct': stages['initial']['answer_role'] == 'GOLD', 'Dplus_adopts_target': stages['Dplus']['answer_role'] == 'Dplus', 'Dminus_adopts_target': stages['Dminus']['answer_role'] == 'Dminus'}, 'stages': stages, 'diagnostics': diagnostics, 'activation_shape': list(states.shape)}
    return (record, {'pair_id': item.pair_id, 'category': item.category, 'condition_names': ['original'], 'stage_names': list(STAGES), 'layer_numbers': layers, 'position_names': ['generation_query'], 'states': states})

def pressure_states(payload: dict[str, Any], item: PressurePair, layers: list[int]) -> np.ndarray:
    """Read a fresh two-stage pressure cache produced by this project."""
    names = payload['stage_names']
    tensor = payload['states']
    if names != list(STAGES) or payload['pair_id'] != item.pair_id or payload['category'] != item.category or (payload['layer_numbers'] != layers) or (payload['condition_names'] != ['original']) or (payload['position_names'] != ['generation_query']) or (tensor.ndim != 5) or (tuple(tensor.shape[:4]) != (1, len(names), len(layers), 1)) or (tensor.shape[-1] < 1):
        raise RuntimeError(f'Incompatible pressure state cache: {item.pair_id}')
    values = np.stack([np.asarray(tensor[0, names.index(stage), :, 0, :], dtype=np.float32) for stage in STAGES])
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite pressure states: {item.pair_id}')
    return values

def validate_record(record: dict[str, Any], item: PressurePair) -> None:
    candidates = role_option_texts(item.dplus, item.dminus)
    if record['pair_id'] != item.pair_id or record['category'] != item.category or record['role_texts'] != candidates or (record['initial_answer_used_in_both_branches'] != record['stages']['initial']['answer_text']) or (record['initial_answer_used_in_both_branches'] not in candidates.values()) or any((record['stages'][stage]['role_texts'] != candidates for stage in ('initial', *STAGES))):
        raise RuntimeError(f'Incompatible baseline record: {item.pair_id}')

def save_pair(output: Path, record: dict[str, Any], payload: dict[str, Any]) -> None:
    import torch
    payload = {**payload, 'states': torch.from_numpy(np.asarray(payload['states'], dtype=np.float32).copy())}
    target = output / 'activations' / f"{record['pair_id']}.pt"
    temporary = target.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(target)
    write_json(output / 'records' / f"{record['pair_id']}.json", record)

def run_collection(config: dict[str, Any], output: Path) -> dict[str, Any]:
    items = dev_items(config)
    layers = configured_layers(config)
    pair_ids = [item.pair_id for item in items]
    manifest = {'schema_version': COLLECTION_SCHEMA, 'model_id': config['model_id'], 'model_revision': config['model_revision'], 'pair_ids': pair_ids, 'counts_by_category': dict(Counter((i.category for i in items))), 'stage_names': list(STAGES), 'layer_numbers': layers, 'condition_names': ['original'], 'position_names': ['generation_query'], 'answer_format': 'actual_option_text', 'method': METHOD, 'dtype': config['dtype'], 'device_map': config['device_map']}
    manifest_path = output / 'manifest.json'
    for directory in ('records', 'activations'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    pending = items
    write_json(manifest_path, manifest)
    if pending:
        import torch
        from ..adapter import load_engine
        torch.manual_seed(int(config['seed']))
        engine = load_engine(config)
        try:
            if layers[-1] > engine.decoder_depth:
                raise ValueError(f'Layer {layers[-1]} exceeds depth {engine.decoder_depth}')
            write_json(output / 'runtime.json', engine.metadata())
            for index, item in enumerate(pending, 1):
                record, payload = collect_item(item, engine, layers, config.get('terminal_token_id'))
                save_pair(output, record, payload)
                print(f'[collect {index}/{len(pending)}] {item.pair_id}', flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {'collected': len(items), 'output': str(output), 'method': METHOD}
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config.json'
DEFAULT_OUTPUT = ROOT / 'results/steering/gemma/v1'

def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--stage', choices=('all', 'collect', 'fit', 'calibrate'), default='all')
    parser.add_argument('--selection-only', action='store_true', help='For --stage calibrate: select sign from saved calibration records')
    args = parser.parse_args(argv)
    if args.selection_only and args.stage != 'calibrate':
        parser.error('--selection-only requires --stage calibrate')
    return args

def main(argv: list[str] | None=None) -> int:
    args = parse_args(argv)
    config = read_json(resolve_path(args.config))
    output = resolve_path(args.output)
    print(f'[v1] output={output}', flush=True)
    result = None
    if args.stage in ('all', 'collect'):
        result = run_collection(config, output)
    if args.stage in ('all', 'fit', 'calibrate'):
        from .extract import run_extraction, validate_fit
        if args.stage in ('all', 'fit'):
            result = run_extraction(config, output)
            print(f"[fit] selected_layer={result['selected_layer']} direction=Dplus-Dminus", flush=True)
        if args.stage in ('all', 'calibrate'):
            from .calibrate import run_calibration, select_from_saved_records
            validate_fit(config, output)
            result = (select_from_saved_records if args.selection_only else run_calibration)(config, output)
            print(f"[calibrate] selected_sign={result['selected_sign']}", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
