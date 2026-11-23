"""VDS90 extraction with existing MUB prompts, followed by frozen MUB478 testing."""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from mub.qwen.core import ROOT, ensure_manifest, read_json, resolve, write_json, write_jsonl
from steering.qwen.v1.extract import fit_states
from steering.qwen.v2.run import selected_conditions, signed_ratio_conditions
from vds_mub.data import digest, prepare
from vds_mub.report import compare

MODEL_DIRS = {'qwen': 'qwen2.5_vl_7b', 'llava': 'llava_next_7b', 'internvl': 'internvl3_8b'}
HERE = Path(__file__).resolve().parent


def backend(model):
    if model not in MODEL_DIRS:
        raise ValueError(f'Unknown model: {model}')
    return SimpleNamespace(**{key: importlib.import_module(f'mub.{model}.{suffix}') for key, suffix in {
        'core': 'core', 'adapter': 'adapter', 'protocol': 'protocol',
        'collection': 'v1.run', 'calibration': 'v1.calibrate', 'test': 'v2.run',
    }.items()})


def code_identity(model):
    paths = list(HERE.glob('*.py'))
    paths += list((ROOT / f'src/mub/{model}').rglob('*.py'))
    paths += list((ROOT / 'src/steering/qwen').rglob('*.py'))
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(paths))}


def fit_source(config, rows, output):
    ids = [r['pair_id'] for r in rows]
    states = []
    for pid in ids:
        with np.load(output / 'activations' / f'{pid}.npz', allow_pickle=False) as archive:
            states.append(archive['states'])
    # Use one pooled axis, exactly as the MUB30 control; no target category is needed.
    result = fit_states(np.stack(states), ids, ['pooled'] * len(ids), config['decoder_layers'], seed=config['seed'])
    np.save(output / 'axis.npy', result['categories']['pooled']['selective'], allow_pickle=False)
    np.savez_compressed(output / 'crossfit_axes.npz', **{
        f'fold_{fold}': item['categories']['pooled'] for fold, item in result['folds'].items()})
    write_json(output / 'folds.json', [{
        'fold': fold, 'train_ids': item['training_pair_ids'], 'held_out_ids': item['held_out_pair_ids']
    } for fold, item in result['folds'].items()])
    write_jsonl(output / 'crossfit.jsonl', result['rows'])
    summary = {**result['summary'], 'training_source': 'vds90',
               'source_counts_by_category': dict(Counter(r['selection_category'] for r in rows))}
    write_json(output / 'summary.json', summary)
    return summary


def frozen(config, source, v1):
    fit = read_json(v1 / 'summary.json')
    selection = read_json(v1 / 'calibration/selection.json')
    axis = np.load(v1 / 'axis.npy', allow_pickle=False)
    if (fit['items'] != len(source) or fit['selected_layer'] not in config['decoder_layers']
            or set(selection['pair_ids']) != {r['pair_id'] for r in source}
            or selection['selected_layer'] != fit['selected_layer']
            or selection['selected_sign'] not in (-1, 0, 1)
            or axis.ndim != 1 or not np.isfinite(axis).all() or not np.linalg.norm(axis) > 0):
        raise ValueError('Invalid source-only frozen fit/selection')
    return fit['selected_layer'], selection['selected_sign'], axis


def run(model, config, output, stage='all', reference=None, engine=None):
    modules = backend(model)
    modules.adapter.check_backend(config, model)
    if config['outer_folds'] != 3:
        raise ValueError('Use the existing three-fold extraction method')
    signed_ratio_conditions(config['test_ratios'])
    source, test, audit = prepare()
    output = resolve(output)
    reference = resolve(reference or ROOT / f'results/mub/{model}')
    identity = {
        'experiment': 'vds90_to_mub478_v1', 'model': model, 'config': config,
        'training_protocol': modules.protocol.TRAINING_PROTOCOL,
        'source_rows': source, 'test_rows': test, 'data_audit': audit,
        'source_code': code_identity(model),
    }
    ensure_manifest(output, identity)
    write_json(output / 'preflight.json', {k: v for k, v in audit.items() if not k.endswith('_images')})
    write_json(output / 'source_vds90.json', source)
    if stage == 'validate':
        return {**read_json(output / 'preflight.json'), 'model': model,
                'model_path': config['model_path'], 'weights_loaded': False}
    if stage == 'compare':
        return compare(config, test, output / 'v2', reference, output, modules.test)
    v1 = output / 'v1'
    ensure_manifest(v1, {'training_protocol': modules.protocol.TRAINING_PROTOCOL,
                         'training_source': 'vds90', 'config': config, 'items': source,
                         'source_images': audit['source_images'], 'source_code': identity['source_code']})
    if stage in ('all', 'collect'):
        missing = any(not (v1 / 'records' / f"{r['pair_id']}.json").exists()
                      or not (v1 / 'activations' / f"{r['pair_id']}.npz").exists() for r in source)
        if missing:
            engine = engine or modules.adapter.load_engine(config, model)
            write_json(v1 / 'model.json', engine.metadata())
        modules.collection.collect(config, source, v1, engine, protocol=modules.protocol)
    if stage in ('all', 'fit'):
        fit_source(config, source, v1)
    if stage in ('all', 'calibrate'):
        missing = any(not (v1 / 'calibration/records' / f"{r['pair_id']}.json").exists() for r in source)
        if missing:
            engine = engine or modules.adapter.load_engine(config, model)
        selection = modules.calibration.calibrate(config, source, v1, engine, protocol=modules.protocol)
        selection['selection_population'] = f'all {len(source)} VDS source pairs, both arms equally weighted'
        selection['selection_reason'] = selection['selection_reason'].replace('train30', 'VDS source90')
        selection['training_source'] = 'vds90'
        write_json(v1 / 'calibration/selection.json', selection)
    if stage not in ('all', 'evaluate'):
        return {'stage': stage, 'model': model, 'source_pairs': len(source), 'output': str(output)}
    layer, sign, axis = frozen(config, source, v1)
    v2 = output / 'v2'
    ensure_manifest(v2, {
        'training_protocol': modules.protocol.TRAINING_PROTOCOL, 'training_source': 'vds90',
        'config': config, 'items': test, 'layer': layer, 'selected_sign': sign, 'axis': axis.tolist(),
        'source_manifest_sha256': digest(v1 / 'manifest.json'),
        'selection_sha256': digest(v1 / 'calibration/selection.json'),
    })
    if any(not (v2 / 'records' / f"{r['pair_id']}.json").exists() for r in test):
        engine = engine or modules.adapter.load_engine(config, model)
        write_json(v2 / 'model.json', engine.metadata())
    modules.test.evaluate(config, test, v2, engine, layer, axis, protocol=modules.protocol)
    records = [read_json(v2 / 'records' / f"{r['pair_id']}.json") for r in test]
    result = {
        'protocol': modules.protocol.TRAINING_PROTOCOL, 'training_source': 'vds90', 'source_pairs': len(source),
        'layer': layer, 'selected_sign': sign,
        'selected_conditions': selected_conditions(config['test_ratios'], sign),
        'scores': modules.test.summarize(test, records, ['baseline', *signed_ratio_conditions(config['test_ratios'])]),
    }
    write_json(v2 / 'summary.json', result)
    comparison = compare(config, test, v2, reference, output, modules.test)
    return {'model': model, 'source_pairs': len(source), 'test_pairs': len(test),
            'selected_layer': layer, 'selected_sign': sign, 'comparison_status': comparison['status'], 'output': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(MODEL_DIRS), default='qwen')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path, help='Existing matching MUB30 run, containing v1/ and v2/')
    parser.add_argument('--stage', choices=['all', 'validate', 'collect', 'fit', 'calibrate', 'evaluate', 'compare'], default='all')
    args = parser.parse_args(argv)
    config_path = args.config or HERE / MODEL_DIRS[args.model] / 'config.json'
    config = read_json(config_path)
    if args.model_path:
        config['model_path'] = str(args.model_path)
    output = args.output or ROOT / 'results/vds_mub' / MODEL_DIRS[args.model]
    print(json.dumps(run(args.model, config, output, args.stage, args.reference), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
