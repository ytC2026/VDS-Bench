"""Paired question-level percentile bootstrap of the original/edited gap.

Conditions on the existing all-four-initial-correct cohort. Resamples each
question's four outcomes together, not individual arms. No API calls.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20261106)
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error('At least 100 bootstrap replicates required')
    names = ('original_dplus', 'original_dminus', 'edited_dplus', 'edited_dminus')
    analysis = json.loads((args.run / 'analysis.json').read_text(encoding='utf-8'))
    ids = analysis['all_four_matched']['pair_ids']
    if not ids:
        raise ValueError('Empty all-four-initial-correct cohort')
    rows = {}
    for name in names:
        records = [json.loads(s) for s in (args.run / name / 'results.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        rows[name] = {r['sample']['pair_id']: r for r in records}
    vectors = []
    for pid in ids:
        records = [rows[name][pid] for name in names]
        if not all(r['metrics']['all_rounds_parse_ok'] and r['metrics']['initial_correct'] for r in records):
            raise ValueError(f'Invalid cohort member: {pid}')
        a, b, c, d = [int(r['metrics']['flip_to_target']) for r in records]
        vectors.append((a - b, c - d, a - b - c + d))
    n = len(vectors)
    rng = random.Random(args.seed)
    draws = [[], [], []]
    for _ in range(args.replicates):
        sample = rng.choices(vectors, k=n)
        for j in range(3):
            draws[j].append(100 * sum(v[j] for v in sample) / n)
    metrics = {}
    for j, name in enumerate(('original_gap', 'edited_gap', 'gap_reduction')):
        ordered = sorted(draws[j])
        low = ordered[int((args.replicates - 1) * .025)]
        high = ordered[int((args.replicates - 1) * .975)]
        metrics[name] = dict(estimate_pp=100 * sum(v[j] for v in vectors) / n,
            ci95_percentile_pp=[low, high], interval_includes_zero=low <= 0 <= high)
    result = dict(method='paired question-level percentile bootstrap', replicates=args.replicates,
        seed=args.seed, cohort='valid responses and initially correct in all four conditions',
        n=n, pair_ids=ids, metrics=metrics,
        per_question_reduction_counts=dict(Counter(v[2] for v in vectors)),
        limitations=['Conditional on the observed all-four-initial-correct cohort.',
            'Resamples questions with observed answers fixed; excludes API rerun variability.',
            'No multiple-comparison correction. Bootstrap intervals are not a separate hypothesis test.'])
    output = args.run / f'bootstrap_gap_{args.replicates}_seed{args.seed}.json'
    body = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if output.exists() and output.read_text(encoding='utf-8') != body:
        raise FileExistsError(f'Refusing to overwrite different result: {output}')
    output.write_text(body, encoding='utf-8')
    print(json.dumps(dict(output=str(output), n=n, replicates=args.replicates, seed=args.seed,
                          metrics=metrics, per_question_reduction_counts=result['per_question_reduction_counts']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
