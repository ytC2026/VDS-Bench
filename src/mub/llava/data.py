"""Read the compact annotations; original full questions stay in raw downloads."""
from functools import lru_cache
import json
import re

from .core import DATA, read_json, read_jsonl


@lru_cache(maxsize=4)
def raw_objects(filename):
    text = (DATA / 'raw/annotations' / filename).read_bytes().decode('utf-8-sig', errors='ignore')
    decoder, rows, pos = json.JSONDecoder(), [], text.index('[') + 1
    while pos < len(text):
        while pos < len(text) and text[pos] in ' \t\r\n,':
            pos += 1
        if pos == len(text) or text[pos] == ']':
            break
        try:
            row, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        rows.append(row)
    return rows


def raw_source(reference):
    filename, index = reference.rsplit('#', 1)
    return raw_objects(filename)[int(index)]


def target(instruction):
    match = re.fullmatch(r'And the true answer is\s*:\s*([A-D])', instruction.strip())
    if not match:
        raise ValueError(f'Invalid MUB false cue: {instruction}')
    return match[1]


def load_split(name):
    rows = read_jsonl(DATA / f'annotations/{name}.jsonl')
    indexed = {r['pair_id']: r for r in rows}
    ids = read_json(DATA / 'splits/split.json')[name + '_ids']
    if len(indexed) != len(rows) or set(indexed) != set(ids):
        raise ValueError(f'Annotation/split mismatch: {name}')
    result = []
    for pid in ids:
        row = dict(indexed[pid])
        if set(row['options']) != set('ABCD') or target(row['misleading_instruction']) == row['true_option']:
            raise ValueError(f'Invalid options or false cue: {pid}')
        if name == 'train':
            edits = row.get('training_option_edits', {})
            if len({row['true_option'], row['dplus'], row['dminus']}) != 3 or len(edits) > 1 or row['true_option'] in edits:
                raise ValueError(f'Invalid D pair or training edit: {pid}')
        source = row.get('source') or {}
        raw = raw_source(source['question']) if source.get('question') else None
        row['original_question'] = raw['question'] if raw else row['question']
        row['inline_options'] = bool(raw and not raw.get('choices'))
        row['split'] = name
        result.append(row)
    return result


def all_items():
    return sorted(load_split('train') + load_split('test'), key=lambda r: r['pair_id'])
