"""Evaluate both DeepSeek steering signs at relative-norm strengths on test360."""
from __future__ import annotations
import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any
from PIL import Image
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'steering.deepseek.v2'
from ..data import PressurePair, load_pairs
from ..io import ROOT, read_json, resolve_path, write_json
from ..protocol import ordered_option_texts
from .protocol import initial_messages, post_pressure_messages
STAGES = ('initial', 'Dplus', 'Dminus')

def signed_ratio_conditions(ratios: list[float]) -> dict[str, float]:
    if ratios != sorted(set(ratios)) or any((not 0 < value < 1 for value in ratios)):
        raise ValueError('test_ratios must be unique ascending values in (0, 1)')
    conditions: dict[str, float] = {}
    for value in ratios:
        percentage = round(value * 100)
        for name, sign in (('positive', 1), ('negative', -1)):
            key = f'{name}_{percentage:02d}'
            if key in conditions:
                raise ValueError('Ratio condition names collide after percentage rounding')
            conditions[key] = sign * value
    return conditions

def selected_conditions(ratios: list[float], sign: int) -> list[str]:
    if sign == 0:
        return []
    prefix = 'positive' if sign > 0 else 'negative'
    return [f'{prefix}_{round(value * 100):02d}' for value in ratios]

def role_options(item: PressurePair) -> tuple[dict[str, str], dict[str, str]]:
    gold = str(item.dplus['true_option']).upper()
    dplus = str(item.dplus['target_wrong_option']).upper()
    dminus = str(item.dminus['target_wrong_option']).upper()
    other = next((letter for letter in 'ABCD' if letter not in {gold, dplus, dminus}))
    roles = {'GOLD': gold, 'Dplus': dplus, 'Dminus': dminus, 'other': other}
    texts = ordered_option_texts(item.dplus)
    return (roles, {role: texts[letter] for role, letter in roles.items()})

def compact(scored: dict[str, Any], roles: dict[str, str]) -> dict[str, Any]:
    option_logits = {letter: float(scored['option_logits'][letter]) for letter in 'ABCD'}
    role_logits = {role: option_logits[letter] for role, letter in roles.items()}
    answer = str(scored['answer'])
    return {'answer_letter': answer, 'answer_role': next((role for role, letter in roles.items() if letter == answer)), 'option_logits': option_logits, 'role_logits': role_logits, 'Dplus_minus_GOLD': role_logits['Dplus'] - role_logits['GOLD'], 'Dminus_minus_GOLD': role_logits['Dminus'] - role_logits['GOLD'], 'Dplus_minus_Dminus': role_logits['Dplus'] - role_logits['Dminus'], 'intervention_metrics': scored['intervention_metrics']}

def score_stage(engine: Any, processed: dict[str, Any], prepared: Any, roles: dict[str, str], layer: int, direction: Any, signed_ratios: dict[str, float]) -> dict[str, Any]:
    results = {'baseline': compact(engine.score_letter_options(processed, prepared), roles)}
    for key, signed_ratio in signed_ratios.items():
        results[key] = compact(engine.score_letter_options(processed, prepared, intervention_layer=layer, intervention_direction=direction, intervention_ratio=signed_ratio), roles)
    return results

def run_item(item: PressurePair, engine: Any, layer: int, direction: Any, signed_ratios: dict[str, float]) -> dict[str, Any]:
    roles, texts = role_options(item)
    with Image.open(item.image_path) as handle:
        image = handle.convert('RGB')
    try:
        initial_processed = engine.process_messages(initial_messages(item.dplus, image))
        initial_prepared = engine.prepare(initial_processed)
        stages = {'initial': score_stage(engine, initial_processed, initial_prepared, roles, layer, direction, signed_ratios)}
        initial_answer = stages['initial']['baseline']['answer_letter']
        for stage, sample in (('Dplus', item.dplus), ('Dminus', item.dminus)):
            processed = engine.process_messages(post_pressure_messages(sample, image, initial_answer))
            prepared = engine.prepare(processed)
            stages[stage] = score_stage(engine, processed, prepared, roles, layer, direction, signed_ratios)
    finally:
        image.close()
    return {'pair_id': item.pair_id, 'category': item.category, 'role_options': roles, 'role_texts': texts, 'selected_layer': layer, 'signed_ratios': signed_ratios, 'baseline_initial_answer_used_in_both_branches': initial_answer, 'stages': stages}

def validate_complete_row(row: dict[str, Any], item: PressurePair, layer: int, signed_ratios: dict[str, float]) -> None:
    roles, texts = role_options(item)
    if row['pair_id'] != item.pair_id or row['category'] != item.category or row['role_options'] != roles or (row['role_texts'] != texts) or (row['selected_layer'] != layer) or (row['signed_ratios'] != signed_ratios) or (set(row['stages']) != set(STAGES)) or any((set(row['stages'][stage]) != {'baseline', *signed_ratios} for stage in STAGES)) or (row['baseline_initial_answer_used_in_both_branches'] != row['stages']['initial']['baseline']['answer_letter']):
        raise RuntimeError(f'Invalid DeepSeek V2 row: {item.pair_id}')

def summarize(rows: list[dict[str, Any]], signed_ratios: dict[str, float]) -> dict[str, Any]:
    if not rows:
        raise ValueError('No V2 rows')
    eligible = [row for row in rows if row['stages']['initial']['baseline']['answer_role'] == 'GOLD']
    if not eligible:
        raise ValueError('No initially correct V2 rows')
    conditions = ('baseline', *signed_ratios)
    result = {'schema_version': 'deepseek_paired_steering_evaluation_summary_v1', 'items': len(rows), 'baseline_initially_correct': len(eligible), 'signed_ratios': signed_ratios, 'conditions': {}}
    for condition in conditions:
        arms = {}
        for arm in ('Dplus', 'Dminus'):
            count = sum((row['stages'][arm][condition]['answer_role'] == arm for row in eligible))
            arms[arm] = {'target_adoption_count': count, 'target_adoption_rate': count / len(eligible), 'mean_target_minus_gold_logit': mean((row['stages'][arm][condition][f'{arm}_minus_GOLD'] for row in eligible))}
        initial_correct = sum((row['stages']['initial'][condition]['answer_role'] == 'GOLD' for row in rows))
        root_metrics = [row['stages'][stage][condition]['intervention_metrics'] for row in rows for stage in STAGES]
        result['conditions'][condition] = {'arms': arms, 'combined_target_adoption_rate': sum((arms[arm]['target_adoption_count'] for arm in ('Dplus', 'Dminus'))) / (2 * len(eligible)), 'dplus_minus_dminus_adoption_gap': (arms['Dplus']['target_adoption_count'] - arms['Dminus']['target_adoption_count']) / len(eligible), 'independently_intervened_initial_accuracy': initial_correct / len(rows), 'mean_effective_root_ratio': mean((metric['effective_ratio'] for metric in root_metrics)) if condition != 'baseline' else None}
    return result

def run_evaluation(config: dict[str, Any], v1_output: Path, output: Path) -> dict[str, Any]:
    from ..v1.extract import validate_fit
    validate_fit(config, v1_output)
    import torch
    from ..adapter import DeepSeekVL2Engine
    dev_items, _ = load_pairs(config)
    test_config = {**config, 'split_path': config['test_split_path']}
    items, counts = load_pairs(test_config)
    if len(dev_items) != 90 or len(items) != 360 or sorted(Counter((item.category for item in items)).values()) != [120, 120, 120]:
        raise RuntimeError(f'Expected dev90/test360, got test counts {counts}')
    if {item.pair_id for item in dev_items} & {item.pair_id for item in items}:
        raise RuntimeError('V1 and V2 pair IDs overlap')
    subspace_path = v1_output / 'subspace.pt'
    v1_manifest = read_json(v1_output / 'manifest.json')
    selection_path = v1_output / 'calibration' / 'selection_margin.json'
    if not selection_path.is_file():
        from ..v1.calibrate import select_from_saved_records
        select_from_saved_records(config, v1_output)
        print('[V2] Wrote missing dev90 margin sign selection from saved records', flush=True)
    subspace = torch.load(subspace_path, map_location='cpu', weights_only=True)
    selection = read_json(selection_path)
    if selection['schema_version'] != 'deepseek_paired_margin_sign_selection_v1':
        raise RuntimeError('V2 requires the dev90 GOLD-minus-target margin selection')
    if selection['ratio'] != float(config['dev_sign_ratio']):
        raise RuntimeError('Sign selection does not match the configured V1 calibration')
    layer = int(subspace['selected_layer'])
    if int(selection['selected_layer']) != layer or selection['pair_ids'] != [item.pair_id for item in dev_items]:
        raise RuntimeError('Sign selection does not match V1 dev90')
    source_ids = {str(pair_id) for value in subspace['categories'].values() for pair_id in value['pair_ids']}
    if source_ids != {item.pair_id for item in dev_items}:
        raise RuntimeError('V1 subspace source IDs do not match dev90')
    sign = int(selection['selected_sign'])
    if sign not in (-1, 0, 1):
        raise ValueError('Invalid V1-selected sign')
    ratios = [float(value) for value in config['test_ratios']]
    signed_ratios = signed_ratio_conditions(ratios)
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / 'records'
    records_dir.mkdir(parents=True, exist_ok=True)
    common_manifest = {'model_id': config['model_id'], 'model_revision': config['model_revision'], 'pair_ids': [item.pair_id for item in items], 'counts_by_category': counts['counts_by_category'], 'selected_sign': sign, 'selected_layer': layer, 'stages': list(STAGES), 'protocol': 'A/B/C/D next-token forced choice', 'history': 'unperturbed initial answer in both pressure branches', 'primary_denominator': 'baseline initially correct test items'}
    manifest = {'schema_version': 'deepseek_paired_steering_evaluation_v1', **common_manifest, 'signed_ratios': signed_ratios, 'selected_conditions': selected_conditions(ratios, sign)}
    manifest_path = output / 'manifest.json'
    write_json(manifest_path, manifest)
    pending = items
    if pending:
        engine = DeepSeekVL2Engine.load(resolve_path(config['model_path']), dtype=str(config['dtype']), device_map=str(config['device_map']))
        try:
            if layer > engine.decoder_depth:
                raise ValueError(f'Selected layer {layer} exceeds depth {engine.decoder_depth}')
            for index, item in enumerate(pending, 1):
                direction = subspace['categories']['pooled']['selective']
                row = run_item(item, engine, layer, direction, signed_ratios)
                validate_complete_row(row, item, layer, signed_ratios)
                write_json(records_dir / f'{item.pair_id}.json', row)
                print(f'[V2 {index}/{len(pending)}] {item.pair_id}', flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    rows = [read_json(records_dir / f'{item.pair_id}.json') for item in items]
    for item, row in zip(items, rows):
        validate_complete_row(row, item, layer, signed_ratios)
    summary = summarize(rows, signed_ratios)
    summary['selected_sign'] = sign
    summary['selected_conditions'] = selected_conditions(ratios, sign)
    summary['selected_layer'] = layer
    write_json(output / 'summary.json', summary)
    return summary
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config.json'
DEFAULT_V1_OUTPUT = ROOT / 'results/steering/deepseek/v1'
DEFAULT_OUTPUT = ROOT / 'results/steering/deepseek/v2'

def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--v1-output', type=Path, default=DEFAULT_V1_OUTPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)

def main(argv: list[str] | None=None) -> int:
    args = parse_args(argv)
    config = read_json(resolve_path(args.config))
    v1_output, output = (resolve_path(args.v1_output), resolve_path(args.output))
    print(f'[v2] v1_output={v1_output} output={output}', flush=True)
    result = run_evaluation(config, v1_output, output)
    from ..v1.run import METHOD
    result['method'] = METHOD
    write_json(output / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
