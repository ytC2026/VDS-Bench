"""Use steering's fold-held-out GOLD-minus-target sign calibration."""
from pathlib import Path
import numpy as np
from PIL import Image

from steering.qwen.v1.selection import select_sign
from mub.llava.core import read_json, write_json, resolve
from mub.llava import protocol as current_protocol
from mub.llava.protocol import training_options


def compact(score, row):
    roles = {row['true_option']: 'GOLD', row['dplus']: 'Dplus', row['dminus']: 'Dminus'}
    logits = score['choice_log_scores']
    return {'answer_role': roles.get(score['answer'], 'other'),
            'Dplus_minus_GOLD': logits[row['dplus']] - logits[row['true_option']],
            'Dminus_minus_GOLD': logits[row['dminus']] - logits[row['true_option']]}


def select(config, rows, output):
    output = Path(output)
    records = [read_json(output / 'calibration/records' / f"{r['pair_id']}.json") for r in rows]
    result = select_sign(records, ratio=config['dev_sign_ratio'])
    result.update(selection_population='all 30 train pairs, both arms equally weighted',
                  selected_layer=read_json(output / 'summary.json')['selected_layer'])
    result['selection_reason'] = result['selection_reason'].replace('dev90', 'train30')
    write_json(output / 'calibration/selection.json', result)
    return result


def calibrate(config, rows, output, engine, protocol=current_protocol):
    output = Path(output)
    layer = read_json(output / 'summary.json')['selected_layer']
    folds = read_json(output / 'folds.json')
    fold_by_id = {pid: fold['fold'] for fold in folds for pid in fold['held_out_ids']}
    with np.load(output / 'crossfit_axes.npz', allow_pickle=False) as saved:
        axes = {key: saved[key] for key in saved.files}
    for row in rows:
        pid = row['pair_id']
        path = output / 'calibration/records' / f'{pid}.json'
        if path.exists():
            continue
        baseline = read_json(output / 'records' / f'{pid}.json')
        fold = fold_by_id[pid]
        stages = {}
        with Image.open(resolve(row['image_path'])) as handle:
            image = handle.convert('RGB')
        try:
            for name, branch in [('initial', None), ('Dplus', 'dplus'), ('Dminus', 'dminus')]:
                inputs = engine.prepare(protocol.training_messages(
                    row, image, branch, initial=baseline['initial']['answer_text']), image)
                stage = {'baseline': compact(baseline[branch or 'initial'], row)}
                for condition, sign in [('negative', -1), ('positive', 1)]:
                    score = engine.score(inputs, options=training_options(row), layer=layer,
                                         axis=axes[f'fold_{fold}'], ratio=sign * config['dev_sign_ratio'])[0]
                    stage[condition] = compact(score, row)
                stages[name] = stage
        finally:
            image.close()
        write_json(path, {'pair_id': pid, 'category': 'mub', 'fold': fold, 'stages': stages})
        print(f'[calibrate] {pid} fold={fold}', flush=True)
    return select(config, rows, output)
