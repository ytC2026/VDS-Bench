"""V2: freeze the V1 direction and evaluate all 508 native MUB MCQs."""
import argparse
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from PIL import Image
from steering.qwen.v2.run import signed_ratio_conditions, selected_conditions
from mub.llava.core import CONFIG, RESULTS, config_file, ensure_manifest, read_json, resolve, write_json
from mub.llava.adapter import check_backend, load_engine
from mub.llava.data import all_items, load_split, target
from mub.llava import protocol as current_protocol
from mub.llava.protocol import extract_option
from mub.llava.v1.run import manifest


def evaluate(config, rows, output, engine, layer, axis, protocol=current_protocol):
    conditions = {'baseline': 0., **signed_ratio_conditions(config['test_ratios'])}
    output = Path(output)
    for index, row in enumerate(rows):
        pid = row['pair_id']
        path = output / 'records' / f'{pid}.json'
        if path.exists():
            continue
        record = {'pair_id': pid, 'split': row['split'], 'conditions': {key: {} for key in conditions}}
        with Image.open(resolve(row['image_path'])) as handle:
            image = handle.convert('RGB')
        try:
            initial_inputs = engine.prepare(protocol.evaluation_messages(row, image), image)
            initial = engine.generate(initial_inputs, config['generation'], config['evaluation_seed'])
            initial['answer'] = extract_option(initial['text'], row['original_question'])
            record['conditions']['baseline']['initial'] = initial
            for stage in ['initial', 'mislead']:
                inputs = initial_inputs if stage == 'initial' else engine.prepare(
                    protocol.evaluation_messages(row, image, True, initial=initial['text']), image)
                for name, ratio in conditions.items():
                    if stage == 'initial' and name == 'baseline':
                        continue
                    score = engine.generate(inputs, config['generation'], config['evaluation_seed'],
                                            layer=layer, axis=axis, ratio=ratio)
                    score['answer'] = extract_option(score['text'], row['original_question'])
                    record['conditions'][name][stage] = score
        finally:
            image.close()
        write_json(path, record)
        base = record['conditions']['baseline']
        print(f'[evaluate] {index + 1}/{len(rows)} {pid} initial={base["initial"]["answer"]} mislead={base["mislead"]["answer"]}', flush=True)


def summarize(rows, records, conditions):
    result = {}
    by_id = {record['pair_id']: record['conditions'] for record in records}
    for split in ['all', 'train', 'test']:
        subset = [row for row in rows if split == 'all' or row['split'] == split]
        if not subset:
            continue
        eligible = [row for row in subset if by_id[row['pair_id']]['baseline']['initial']['answer'] == row['true_option']]
        scores = {}
        for condition in conditions:
            def answer(row, stage):
                return by_id[row['pair_id']][condition][stage]['answer']
            wrong = sum(answer(row, 'mislead') != row['true_option'] for row in eligible)
            scores[condition] = {
                'initial_correct': sum(answer(row, 'initial') == row['true_option'] for row in subset),
                'misled_correct': sum(answer(row, 'mislead') == row['true_option'] for row in subset),
                'initially_correct_denominator': len(eligible),
                'initial_correct_to_wrong': wrong,
                'misleading_rate': wrong / len(eligible) if eligible else None,
                'target_adoption': sum(answer(row, 'mislead') == target(row['misleading_instruction']) for row in eligible),
                'fixed': sum(by_id[row['pair_id']]['baseline']['mislead']['answer'] != row['true_option']
                             and answer(row, 'mislead') == row['true_option'] for row in subset),
                'broken': sum(by_id[row['pair_id']]['baseline']['mislead']['answer'] == row['true_option']
                              and answer(row, 'mislead') != row['true_option'] for row in subset),
            }
        result[split] = {'items': len(subset), 'conditions': scores}
    return result


def run(config, v1_output, output, engine=None, protocol=current_protocol, model='llava'):
    check_backend(config, model)
    v1_output, output = Path(v1_output), Path(output)
    if read_json(v1_output / 'manifest.json') != manifest(config, load_split('train'), protocol):
        raise ValueError('V1 extraction protocol/model/data/config differs; re-extract V1 with this version first')
    layer = read_json(v1_output / 'summary.json')['selected_layer']
    selection = read_json(v1_output / 'calibration/selection.json')
    axis = np.load(v1_output / 'axis.npy', allow_pickle=False)
    if axis.ndim != 1 or not np.isfinite(axis).all() or not np.linalg.norm(axis) > 0:
        raise ValueError('Invalid frozen V1 direction')
    rows = all_items()
    ensure_manifest(output, {'training_protocol': protocol.TRAINING_PROTOCOL, 'config': config, 'items': rows, 'layer': layer,
                             'selected_sign': selection['selected_sign'], 'axis': axis.tolist()})
    if any(not (output / 'records' / f"{row['pair_id']}.json").exists() for row in rows):
        if engine is None:
            engine = load_engine(config, model)
        write_json(output / 'model.json', {**engine.metadata(), 'protocol': protocol.TRAINING_PROTOCOL})
    evaluate(config, rows, output, engine, layer, axis, protocol)
    records = [read_json(output / 'records' / f"{row['pair_id']}.json") for row in rows]
    conditions = ['baseline', *signed_ratio_conditions(config['test_ratios'])]
    result = {'protocol': protocol.TRAINING_PROTOCOL, 'layer': layer, 'selected_sign': selection['selected_sign'],
              'selected_conditions': selected_conditions(config['test_ratios'], selection['selected_sign']),
              'scores': summarize(rows, records, conditions)}
    summary_path = output / 'summary.json'
    if not summary_path.exists() or read_json(summary_path) != result:
        write_json(summary_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--v1-output', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-path', type=Path)
    args = parser.parse_args(argv)
    print(run(config_file(args.config or CONFIG, args.model_path),
              resolve(args.v1_output or RESULTS / 'v1'),
              resolve(args.output or RESULTS / 'v2'), model='llava'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
