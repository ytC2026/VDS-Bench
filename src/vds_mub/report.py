"""Compare extraction sources on the same MUB test items and audit baselines."""
import csv
from mub.qwen.core import read_json, write_json, resolve
from mub.qwen.data import load_split


def compact(config):
    return {**config, 'model_path': str(resolve(config['model_path']))}


def compare(config, rows, transfer, reference, output, test_module):
    transfer, reference, output = map(resolve, (transfer, reference, output))
    new = read_json(transfer / 'summary.json')
    if not (reference / 'v2/summary.json').exists():
        result = {'status': 'reference_unavailable', 'transfer': str(transfer), 'reference': str(reference)}
        write_json(output / 'comparison.json', result)
        return result
    old = read_json(reference / 'v2/summary.json')
    manifest = read_json(reference / 'v2/manifest.json')
    source_manifest = read_json(reference / 'v1/manifest.json')
    if source_manifest['items'] != load_split('train'):
        raise ValueError('Reference must be the existing fixed MUB30 extraction run')
    if compact(config) != compact(manifest['config']):
        raise ValueError('MUB30 reference has a different model/config; use a matching reference run')
    protocol_match = new['protocol'] == old['protocol'] == manifest['training_protocol']
    expected = {r['pair_id']: r for r in rows}
    saved = {r['pair_id']: r for r in manifest['items'] if r['pair_id'] in expected}
    if saved != expected or not protocol_match:
        raise ValueError('Reference MUB test data or protocol differs')
    sets = {}
    for label, folder in [('MUB30', reference / 'v2'), ('VDS90', transfer)]:
        sets[label] = {row['pair_id']: read_json(folder / 'records' / f"{row['pair_id']}.json") for row in rows}
    mismatches = []
    for row in rows:
        pid = row['pair_id']
        before, after = [sets[label][pid]['conditions']['baseline'] for label in ('MUB30', 'VDS90')]
        for stage in ('initial', 'mislead'):
            if any(before[stage][key] != after[stage][key] for key in ('text', 'answer')):
                mismatches.append({'pair_id': pid, 'stage': stage})
    flat = []
    for label, summary in [('MUB30', old), ('VDS90', new)]:
        records = sets[label]
        names = ['baseline', *summary['selected_conditions']]
        stats = test_module.summarize(rows, list(records.values()), names)['test']['conditions']
        for name in names:
            score = stats[name]
            n = score['initially_correct_denominator']
            eligible = [r for r in rows if records[r['pair_id']]['conditions']['baseline']['initial']['answer'] == r['true_option']]
            clean_harm = sum(records[r['pair_id']]['conditions'][name]['initial']['answer'] != r['true_option'] for r in eligible)
            flat.append({
                'source': label, 'condition': name, 'test_n': len(rows), 'primary_n': n,
                'initial_accuracy': score['initial_correct'] / len(rows),
                'initial_correct_preservation_rate': 1 - clean_harm / n if n else None,
                'initial_harm_count': clean_harm, 'misled_accuracy': score['misled_correct'] / len(rows),
                'misleading_rate_any_wrong': score['misleading_rate'],
                'target_adoption_rate': score['target_adoption'] / n if n else None,
                'repair_count': score['fixed'], 'harm_count': score['broken'],
            })
    result = {
        'status': 'aligned' if not mismatches else 'baseline_mismatch',
        'baseline_mismatches': mismatches, 'test_pairs': len(rows),
        'source_sizes': {'MUB30': 30, 'VDS90': 90}, 'rows': flat,
        'interpretation': 'Source sample counts differ; evaluate cross-dataset transfer, not equal-budget source superiority.',
    }
    write_json(output / 'comparison.json', result)
    with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ['# VDS90 to MUB478', '', f"Baseline alignment: {result['status']}; mismatched stage records: {len(mismatches)}.", '',
             '| Source | Condition | n initially correct | Initial acc | Correct preservation | Wrong adoption | Any wrong after cue |',
             '|---|---|---:|---:|---:|---:|---:|']
    pct = lambda x: 'NA' if x is None else f'{100*x:.2f}%'
    for r in flat:
        lines.append(f"| {r['source']} | {r['condition']} | {r['primary_n']} | {pct(r['initial_accuracy'])} | "
                     f"{pct(r['initial_correct_preservation_rate'])} | {pct(r['target_adoption_rate'])} | {pct(r['misleading_rate_any_wrong'])} |")
    lines += ['', result['interpretation']]
    if mismatches:
        lines += ['', 'Baselines differ: investigate the model/runtime/history before drawing a direct method comparison.']
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result
