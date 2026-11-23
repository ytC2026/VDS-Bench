"""Data checks without weights, plus an optional real-checkpoint interface check."""
import numpy as np
from PIL import Image

from mub.internvl.adapter import check_backend, load_engine
from mub.internvl.core import resolve
from mub.internvl.data import all_items, load_split
from mub.internvl.protocol import training_options


def validate(config, model='internvl'):
    check_backend(config, model)
    if (config['attn_implementation'] != 'eager' or config['transformers_version'] != '4.48.3'
            or not 1 <= config['min_dynamic_patch'] <= config['max_dynamic_patch'] <= 12
            or config['image_size'] != 448 or not config['use_thumbnail']):
        raise ValueError('Use the fixed InternVL3 native image settings and separate environment')
    rows, train = all_items(), load_split('train')
    layers, depth = config['decoder_layers'], 28
    if (config['outer_folds'] != 3 or not layers or len(set(layers)) != len(layers)
            or any(not isinstance(l, int) or not 1 <= l <= depth for l in layers)
            or not 0 < config['dev_sign_ratio'] < 1
            or not config['test_ratios'] or any(not 0 < r < 1 for r in config['test_ratios'])):
        raise ValueError('Use valid candidate layers, norm ratios and three folds')
    if len(rows) != 508 or len(train) != 30 or sum(r['split'] == 'test' for r in rows) != 478:
        raise ValueError('Expected shared MUB 30/478 split')
    for row in rows:
        if not resolve(row['image_path']).is_file():
            raise FileNotFoundError(row['image_path'])
    for row in train:
        training_options(row)
    return {'model': model, 'train': 30, 'test': 478, 'evaluate': 508,
            'candidate_layers': layers, 'weights_loaded': False}


def smoke(config, model, protocol):
    """Real forward/generation with one train image; never write experiment results."""
    engine, row = load_engine(config, model), load_split('train')[0]
    options = training_options(row)
    with Image.open(resolve(row['image_path'])) as handle:
        image = handle.convert('RGB')
    try:
        initial = engine.score(engine.prepare(protocol.training_messages(row, image), image),
                               options=options)[0]
        paired = []
        for arm in ('dplus', 'dminus'):
            inputs = engine.prepare(protocol.training_messages(row, image, arm, initial=initial['answer_text']), image)
            _, states = engine.score(inputs, options=options, capture=config['decoder_layers'])
            paired.append(states)
        layer = config['decoder_layers'][0]
        axis = paired[0][str(layer)] - paired[1][str(layer)]
        if not np.isfinite(axis).all() or np.linalg.norm(axis) <= 0:
            raise ValueError('Invalid paired activation difference in interface check')
        inputs = engine.prepare(protocol.training_messages(row, image, 'dplus', initial=initial['answer_text']), image)
        scored = engine.score(inputs, options=options, layer=layer, axis=axis, ratio=.1)[0]
        generation = {**config['generation'], 'max_new_tokens': 8}
        first = engine.generate(engine.prepare(protocol.evaluation_messages(row, image), image),
                                generation, config['evaluation_seed'])
        inputs = engine.prepare(protocol.evaluation_messages(row, image, True, initial=first['text']), image)
        second = engine.generate(inputs, generation, config['evaluation_seed'], layer=layer, axis=axis, ratio=.1)
        return {'model': model, 'real_checkpoint_smoke': True, 'pair_id': row['pair_id'],
                'state_shapes': {k: list(v.shape) for k, v in paired[0].items()},
                'score_intervention': scored['intervention'],
                'generation_intervention': second['intervention'], 'experiment_results_written': False}
    finally:
        image.close()
