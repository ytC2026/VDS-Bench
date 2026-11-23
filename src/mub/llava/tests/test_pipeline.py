"""CPU checks for current post-answer prompts, cross-fitting and frozen history."""
import ast
from copy import deepcopy
from pathlib import Path
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

from mub.llava.core import DATA, CONFIG, ROOT, RESULTS, read_json, read_jsonl, write_json
from mub.llava.data import all_items, load_split, raw_source, target
from mub.llava.protocol import (MUB_SYSTEM, evaluation_messages, evaluation_prompt, extract_option,
                          training_messages, training_options)
from mub.llava.v1.run import collect, manifest, run as train_run
from mub.llava.v1.extract import fit
from mub.llava.v1.calibrate import calibrate
from mub.llava.v2.run import run as test_run, summarize
from mub.llava import protocol as post_answer
from mub.llava import protocol
from mub.llava.run import main
from steering.qwen.v1.extract import fit_states


class FakeEngine:
    """Exercise orchestration only; these outputs are not model results."""
    def __init__(self):
        self.calls = []

    def metadata(self):
        return {'fake': True}

    def prepare(self, history, image):
        self.calls.append(history)
        return history

    def score(self, history, *, options, capture=(), ratio=0., **kwargs):
        content = history[-1]['content']
        cue = content if isinstance(content, str) else next(c['text'] for c in content if c['type'] == 'text')
        plus = 'I think the correct answer is: ' in cue
        scores = {key: -index for index, key in enumerate('ABCD')}
        # Paired states differ in a consistent direction, independently of gold.
        value = list(options.values()).index(cue.split('I think the correct answer is: ', 1)[1].split(
            '\nOptions:\n', 1)[0].split('\n\nReply with', 1)[0]) if plus else 0
        states = {str(layer): np.array([value, value * .1, 1], dtype=np.float32) for layer in capture}
        if ratio:
            scores = {key: score + ratio * (index + 1) for index, (key, score) in enumerate(scores.items())}
        answer = max(scores, key=scores.get)
        return {'answer': answer, 'answer_text': options[answer], 'choice_log_scores': scores}, states

    def generate(self, history, generation, seed, ratio=0., **kwargs):
        assert [m['role'] for m in history] in [['user'], ['user', 'assistant', 'user']]
        if len(history) == 3:
            assert history[1]['content'] == 'A'
        assert generation['temperature'] == .4 and seed == 42
        return {'text': 'B' if ratio > 0 else 'A', 'intervention': None}


class PipelineTests(unittest.TestCase):
    def test_compact_data_and_training_protocol(self):
        train, test = load_split('train'), load_split('test')
        self.assertEqual((len(train), len(test)), (30, 478))
        self.assertEqual(set(read_json(DATA / 'splits/split.json')), {'train_ids', 'test_ids'})
        self.assertEqual(sum(not row['source']['misleading_instruction'] for row in train + test), 248)
        expected = {'pair_id', 'image_path', 'question', 'options', 'true_option', 'misleading_instruction', 'source'}
        for row in read_jsonl(DATA / 'annotations/test.jsonl'):
            self.assertEqual(set(row), expected)
        for row in train + test:
            self.assertTrue((ROOT / row['image_path']).is_file())
            self.assertNotEqual(target(row['misleading_instruction']), row['true_option'])
            self.assertEqual(row['original_question'], raw_source(row['source']['question'])['question'])
        for row in train:
            options = training_options(row)
            self.assertEqual(len(set(options.values())), 4)
            prompts = [training_messages(row, None, arm, initial=options['A']) for arm in ['dplus', 'dminus']]
            normalized = []
            for arm, history in zip(['dplus', 'dminus'], prompts):
                self.assertEqual([m['role'] for m in history], ['system', 'user', 'assistant', 'user'])
                text = history[-1]['content']
                cue = 'I think the correct answer is: ' + options[row[arm]]
                self.assertIn(cue, text)
                self.assertIn('one option text', text)
                self.assertNotIn('official', text)
                self.assertNotIn('one option letter', text)
                normalized.append(text.replace(cue, '<CUE>', 1))
            self.assertEqual(normalized[0], normalized[1])

    def test_official_prompt_and_extraction_parity(self):
        # Load just the pure author definitions, without their external imports.
        source = ast.parse((ROOT / 'src/mub/official_utils.py').read_text(encoding='utf-8-sig'))
        nodes = [node for node in source.body if (
            isinstance(node, ast.FunctionDef) and node.name in ['construct_query_base_MC', 'dictionary_to_text', 'extract_option']
            or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_MSG_MC' for t in node.targets))]
        scope = {'re': re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<official-pure-functions>', 'exec'), scope)
        self.assertEqual(MUB_SYSTEM, scope['SYSTEM_MSG_MC'])
        for row in all_items():
            question = row['original_question']
            expected = (MUB_SYSTEM + question if row['inline_options'] else
                        scope['construct_query_base_MC'](question, row['options']))
            self.assertEqual(evaluation_prompt(row), expected)
            self.assertEqual([m['role'] for m in evaluation_messages(row, None)], ['user'])
            history = evaluation_messages(row, None, True, initial='A')
            self.assertEqual([m['role'] for m in history], ['user', 'assistant', 'user'])
            for response in ['A', '(B)', 'The answer is C.', 'D\n', 'unknown', row['options']['A']]:
                self.assertEqual(extract_option(response, row['original_question']),
                                 scope['extract_option'](response, row['original_question']))

    def test_three_fold_fit_and_calibration(self):
        rows = load_split('train')
        config = read_json(CONFIG)
        # Ensure enough initially correct cases for steering's calibration rule.
        rows = [{**r, 'true_option': 'A', 'dplus': 'B', 'dminus': 'C'} for r in rows]
        with tempfile.TemporaryDirectory() as temp:
            output, engine = Path(temp), FakeEngine()
            collect(config, rows, output, engine)
            result = fit(config, rows, output)
            folds = read_json(output / 'folds.json')
            self.assertEqual(len(folds), 3)
            held = []
            for fold in folds:
                self.assertEqual((len(fold['train_ids']), len(fold['held_out_ids'])), (20, 10))
                self.assertFalse(set(fold['train_ids']) & set(fold['held_out_ids']))
                held += fold['held_out_ids']
            self.assertEqual(set(held), {row['pair_id'] for row in rows})
            states = np.stack([np.load(output / 'activations' / f"{r['pair_id']}.npz")['states'] for r in rows])
            reference = fit_states(states, [r['pair_id'] for r in rows], ['mub'] * 30,
                                   config['decoder_layers'], seed=config['seed'])
            self.assertEqual(result['selected_layer'], reference['summary']['selected_layer'])
            np.testing.assert_array_equal(np.load(output / 'axis.npy'), reference['categories']['pooled']['selective'])
            selection = calibrate(config, rows, output, engine)
            self.assertTrue(all([m['role'] for m in history] in [
                ['system', 'user'], ['system', 'user', 'assistant', 'user']] for history in engine.calls))
            self.assertEqual(selection['folds'], [0, 1, 2])
            self.assertEqual(selection['selection_metrics']['baseline']['margin_denominator'], 60)
            # Cached collection and calibration must not require a GPU engine.
            collect(config, rows, output, None)
            self.assertEqual(calibrate(config, rows, output, None), selection)

    def test_old_multiturn_cache_is_rejected(self):
        config, rows = read_json(CONFIG), load_split('train')
        with tempfile.TemporaryDirectory() as temp:
            v1 = Path(temp) / 'v1'
            write_json(v1 / 'manifest.json', {'config': config, 'items': rows})
            with self.assertRaises(ValueError):
                train_run(config, v1, stage='fit')
            with self.assertRaises(ValueError):
                test_run(config, v1, Path(temp) / 'v2')

    def test_frozen_evaluation_and_resume(self):
        config, rows = read_json(CONFIG), all_items()[:2]
        with tempfile.TemporaryDirectory() as temp:
            v1, v2 = Path(temp) / 'v1', Path(temp) / 'v2'
            write_json(v1 / 'manifest.json', manifest(config, load_split('train')))
            write_json(v1 / 'summary.json', {'selected_layer': 14})
            write_json(v1 / 'calibration/selection.json', {'selected_sign': -1})
            np.save(v1 / 'axis.npy', np.array([1., 0., 0.]))
            with patch('mub.llava.v2.run.all_items', return_value=rows):
                first = test_run(config, v1, v2, FakeEngine())
                # Resuming an uploaded Linux run must preserve its original bytes.
                saved = {}
                for path in [v2 / 'manifest.json', v2 / 'summary.json']:
                    saved[path] = path.read_bytes().replace(b'\r\n', b'\n')
                    path.write_bytes(saved[path])
                self.assertEqual(test_run(config, v1, v2), first)
                self.assertEqual({path: path.read_bytes() for path in saved}, saved)
                self.assertEqual(first['selected_conditions'], ['negative_10', 'negative_20', 'negative_30'])
                changed = deepcopy(config)
                changed['generation']['temperature'] = .3
                with self.assertRaises(ValueError):
                    test_run(changed, v1, v2)

    def test_misleading_rate_uses_fixed_baseline_population(self):
        rows = [{'pair_id': 'x', 'split': 'test', 'true_option': 'A', 'misleading_instruction': 'And the true answer is : B'},
                {'pair_id': 'y', 'split': 'test', 'true_option': 'C', 'misleading_instruction': 'And the true answer is : B'}]
        def stages(first, second):
            return {'initial': {'answer': first}, 'mislead': {'answer': second}}
        records = [{'pair_id': 'x', 'conditions': {'baseline': stages('A', 'B'), 'positive': stages('B', 'A')}},
                   {'pair_id': 'y', 'conditions': {'baseline': stages('B', 'B'), 'positive': stages('C', 'B')}}]
        result = summarize(rows, records, ['baseline', 'positive'])['all']['conditions']
        self.assertEqual(result['baseline']['misleading_rate'], 1)
        self.assertEqual(result['positive']['misleading_rate'], 0)
        self.assertEqual(result['positive']['initially_correct_denominator'], 1)

    def test_post_answer_messages_share_real_initial_answer(self):
        for row in load_split('train'):
            options = training_options(row)
            first = options['A']
            arms = [post_answer.training_messages(row, None, arm, initial=first) for arm in ['dplus', 'dminus']]
            for arm, history in zip(['dplus', 'dminus'], arms):
                self.assertEqual([m['role'] for m in history], ['system', 'user', 'assistant', 'user'])
                self.assertEqual(history[-2]['content'], first)
                self.assertIn(options[row[arm]], history[-1]['content'])
                self.assertIn('one option text', history[-1]['content'])
                self.assertNotIn('one option letter', history[-1]['content'])
                self.assertNotIn('I think the correct answer is:', history[1]['content'][1]['text'])
            self.assertEqual(arms[0][:-1], arms[1][:-1])
            evaluation = post_answer.evaluation_messages(row, None, True, initial='My answer is A.')
            self.assertEqual([m['role'] for m in evaluation], ['user', 'assistant', 'user'])
            self.assertEqual(evaluation[1]['content'], 'My answer is A.')
            self.assertTrue(evaluation[-1]['content'].startswith(row['misleading_instruction']))
            self.assertIn('one option letter', evaluation[-1]['content'])
            with self.assertRaises(ValueError):
                post_answer.training_messages(row, None, 'dplus')
            with self.assertRaises(ValueError):
                post_answer.evaluation_messages(row, None, True)

    def test_post_answer_collection_calibration_and_cache_isolation(self):
        config = read_json(CONFIG)
        rows = [{**r, 'true_option': 'A', 'dplus': 'B', 'dminus': 'C'} for r in load_split('train')]
        with tempfile.TemporaryDirectory() as temp:
            output, engine = Path(temp), FakeEngine()
            collect(config, rows, output, engine, protocol=post_answer)
            fit(config, rows, output)
            calibrate(config, rows, output, engine, protocol=post_answer)
            histories = [h for h in engine.calls if len(h) == 4]
            self.assertTrue(histories)
            self.assertTrue(all(h[-2]['role'] == 'assistant' for h in histories))
            collect(config, rows, output, None, protocol=post_answer)
            calibrate(config, rows, output, None, protocol=post_answer)
            write_json(output / 'manifest.json', {
                **manifest(config, load_split('train')), 'training_protocol': 'independent_content_v1'})
            with self.assertRaises(ValueError):
                train_run(config, output, stage='fit')

    def test_post_answer_evaluation_uses_unsteered_history_for_every_strength(self):
        config, rows = read_json(CONFIG), all_items()[:2]
        with tempfile.TemporaryDirectory() as temp:
            v1, v2 = Path(temp) / 'v1', Path(temp) / 'v2'
            write_json(v1 / 'manifest.json', manifest(config, load_split('train'), protocol=post_answer))
            write_json(v1 / 'summary.json', {'selected_layer': 14})
            write_json(v1 / 'calibration/selection.json', {'selected_sign': 1})
            np.save(v1 / 'axis.npy', np.array([1., 0., 0.]))
            engine = FakeEngine()
            with patch('mub.llava.v2.run.all_items', return_value=rows):
                result = test_run(config, v1, v2, engine, protocol=post_answer)
                self.assertEqual(test_run(config, v1, v2, protocol=post_answer), result)
                self.assertEqual(test_run(config, v1, v2), result)
            self.assertEqual(result['protocol'], 'post_answer_content_v1')
            self.assertTrue(all(h[1]['content'] == 'A' for h in engine.calls if len(h) == 3))
            self.assertEqual(read_json(v2 / 'records' / f"{rows[0]['pair_id']}.json")['conditions']['positive_10']['initial']['text'], 'B')

    def test_model_entry_uses_post_answer_and_own_results(self):
        self.assertEqual(protocol.TRAINING_PROTOCOL, 'post_answer_content_v1')
        self.assertIs(post_answer.training_messages, training_messages)
        self.assertIs(post_answer.evaluation_messages, evaluation_messages)
        with patch('mub.llava.v2.run.run', return_value={}) as evaluate, redirect_stdout(StringIO()):
            self.assertEqual(main(['--stage', 'evaluate']), 0)
            self.assertEqual(evaluate.call_args.args[1:3], (RESULTS / 'v1', RESULTS / 'v2'))
            self.assertIs(evaluate.call_args.kwargs['protocol'], protocol)


if __name__ == '__main__':
    unittest.main()
