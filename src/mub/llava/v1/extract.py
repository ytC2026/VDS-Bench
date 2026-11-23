"""Use steering's exact three-fold paired-state fitting and layer selection."""
from pathlib import Path
import numpy as np

from steering.qwen.v1.extract import fit_states
from mub.llava.core import write_json, write_jsonl


def fit(config, rows, output):
    output = Path(output)
    ids = [r['pair_id'] for r in rows]
    states = []
    for pid in ids:
        with np.load(output / 'activations' / f'{pid}.npz', allow_pickle=False) as archive:
            states.append(archive['states'])
    # MUB uses one pooled direction, rather than small singleton-type directions.
    result = fit_states(np.stack(states), ids, ['mub'] * len(ids), config['decoder_layers'], seed=config['seed'])
    np.save(output / 'axis.npy', result['categories']['pooled']['selective'], allow_pickle=False)
    np.savez_compressed(output / 'crossfit_axes.npz', **{
        f'fold_{fold}': item['categories']['pooled'] for fold, item in result['folds'].items()})
    write_json(output / 'folds.json', [{
        'fold': fold, 'train_ids': item['training_pair_ids'], 'held_out_ids': item['held_out_pair_ids']
    } for fold, item in result['folds'].items()])
    write_jsonl(output / 'crossfit.jsonl', result['rows'])
    write_json(output / 'summary.json', result['summary'])
    return result['summary']
