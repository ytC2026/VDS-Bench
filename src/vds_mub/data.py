"""Fixed source/target data and image identity checks, without model imports."""
from collections import Counter
import hashlib

from mub.qwen.core import ROOT, read_json, read_jsonl, resolve
from mub.qwen.data import load_split
from steering.qwen.protocol import question_stem

SOURCE_SPLIT = 'data/splits/binding_subspace_dev90.json'


def digest(path):
    value = hashlib.sha256()
    with resolve(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def source_rows():
    split = read_json(SOURCE_SPLIT)
    ids = split['pair_ids']
    if len(ids) != 90 or len(set(ids)) != 90:
        raise ValueError('Use the fixed VDS dev90 extraction split')
    test_ids = set(read_json('data/splits/binding_subspace_test360.json')['pair_ids'])
    if set(ids) & test_ids:
        raise ValueError('VDS extraction and test360 overlap')
    indexed = []
    for filename in ('vds_dplus.jsonl', 'vds_dminus.jsonl'):
        rows = read_jsonl(f'data/benchmark/{filename}')
        index = {r['pair_id']: r for r in rows}
        if len(index) != len(rows):
            raise ValueError('Duplicate VDS pair IDs')
        indexed.append(index)
    result = []
    for pid in ids:
        plus, minus = [index[pid] for index in indexed]
        for field in ('question', 'options', 'true_option', 'image_path', 'image_id', 'selection_category'):
            if plus[field] != minus[field]:
                raise ValueError(f'{pid}: paired {field} differs')
        gold, dp, dm = plus['true_option'], plus['target_wrong_option'], minus['target_wrong_option']
        if set(plus['options']) != set('ABCD') or len({gold, dp, dm}) != 3:
            raise ValueError(f'{pid}: invalid gold/D+/D- options')
        options = {key: plus['options'][key] for key in 'ABCD'}
        if len(set(options.values())) != 4 or any(not value.strip() for value in options.values()):
            raise ValueError(f'{pid}: option texts must be distinct and nonempty')
        image = resolve(plus['image_path'])
        if image.parent != (ROOT / 'data/images').resolve() or not image.is_file():
            raise ValueError(f'{pid}: expected the original VDS image')
        result.append({
            'pair_id': pid, 'image_path': plus['image_path'], 'image_id': plus['image_id'],
            'question': question_stem(plus['question']), 'options': options,
            'true_option': gold, 'dplus': dp, 'dminus': dm,
            'selection_category': plus['selection_category'], 'split': 'source_vds90',
        })
    counts = dict(Counter(r['selection_category'] for r in result))
    if counts != split['counts_by_category'] or sorted(counts.values()) != [30, 30, 30]:
        raise ValueError('Expected 30 VDS source pairs per category')
    return result


def prepare():
    source, test = source_rows(), load_split('test')
    if len(test) != 478 or {r['pair_id'] for r in source} & {r['pair_id'] for r in test}:
        raise ValueError('Expected a disjoint MUB478 target test set')
    cache = {}

    def fingerprint(row):
        path = resolve(row['image_path'])
        if path not in cache:
            cache[path] = digest(path)
        return {'pair_id': row['pair_id'], 'image_sha256': cache[path]}

    source_images, test_images = ([fingerprint(r) for r in rows] for rows in (source, test))
    if {r['image_sha256'] for r in source_images} & {r['image_sha256'] for r in test_images}:
        raise ValueError('Exact image overlap between VDS source and MUB test')
    audit = {
        'source': 'VDS fixed dev90', 'source_pairs': len(source), 'target': 'MUB locked test478',
        'test_pairs': len(test), 'mub_reference_train_pairs': len(load_split('train')),
        'counts_by_category': dict(Counter(r['selection_category'] for r in source)),
        'exact_image_overlap': 0, 'source_split_sha256': digest(SOURCE_SPLIT),
        'source_images': source_images, 'test_images': test_images,
    }
    return source, test, audit
