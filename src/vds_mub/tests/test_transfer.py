"""CPU orchestration checks; synthetic engine outputs are not experiment results."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from mub.qwen.core import read_json, write_json
from mub.qwen.data import load_split
from mub.qwen.tests.test_pipeline import FakeEngine
from vds_mub.data import source_rows, prepare
from vds_mub.run import HERE, MODEL_DIRS, backend, run
from vds_mub.report import compare


class TransferTests(unittest.TestCase):
    def test_fixed_source_targets_and_prompt_parity(self):
        from mub.qwen.core import read_jsonl
        plus = {r['pair_id']: r for r in read_jsonl('data/benchmark/vds_dplus.jsonl')}
        minus = {r['pair_id']: r for r in read_jsonl('data/benchmark/vds_dminus.jsonl')}
        rows = source_rows()
        self.assertEqual(len(rows), 90)
        for row in rows:
            pid = row['pair_id']
            self.assertEqual(row['options'], plus[pid]['options'])
            self.assertEqual(row['true_option'], plus[pid]['true_option'])
            self.assertEqual(row['dplus'], plus[pid]['target_wrong_option'])
            self.assertEqual(row['dminus'], minus[pid]['target_wrong_option'])
            expected = backend('qwen').protocol.training_messages(row, None, 'dplus', initial=row['options']['A'])
            for model in MODEL_DIRS:
                actual = backend(model).protocol.training_messages(row, None, 'dplus', initial=row['options']['A'])
                self.assertEqual(actual, expected)
                self.assertTrue(actual[-1]['content'].startswith('I think the correct answer is: '))
                self.assertEqual(actual[-2]['content'], row['options']['A'])

    def test_real_data_has_no_exact_image_overlap(self):
        source, test, audit = prepare()
        self.assertEqual((len(source), len(test)), (90, 478))
        self.assertEqual(audit['exact_image_overlap'], 0)

    def test_three_models_source_only_fit_evaluation_resume_and_comparison(self):
        # Relabel only these ephemeral test fixtures to give the fake scorer a known signal.
        source = [{**r, 'true_option': 'A', 'dplus': 'B', 'dminus': 'C'} for r in source_rows()[:3]]
        test = load_split('test')[:2]
        audit = {'source_pairs': 3, 'test_pairs': 2, 'source_images': [], 'test_images': []}
        for model, directory in MODEL_DIRS.items():
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                cfg = read_json(HERE / directory / 'config.json')
                modules = backend(model)
                reference, output = Path(temporary) / 'reference', Path(temporary) / 'transfer'
                write_json(reference / 'v1/manifest.json', {'items': load_split('train')})
                write_json(reference / 'v2/manifest.json', {
                    'config': cfg, 'items': test, 'training_protocol': modules.protocol.TRAINING_PROTOCOL})
                with redirect_stdout(StringIO()):
                    modules.test.evaluate(cfg, test, reference / 'v2', FakeEngine(), cfg['decoder_layers'][0], np.ones(3))
                write_json(reference / 'v2/summary.json', {
                    'protocol': modules.protocol.TRAINING_PROTOCOL, 'selected_conditions': ['positive_10']})
                engine = FakeEngine()
                with patch('vds_mub.run.prepare', return_value=(source, test, audit)), redirect_stdout(StringIO()):
                    result = run(model, cfg, output, reference=reference, engine=engine)
                    self.assertEqual(result['comparison_status'], 'aligned')
                    with patch.object(modules.adapter, 'load_engine', side_effect=AssertionError('GPU reload')):
                        self.assertEqual(run(model, cfg, output, reference=reference), result)
                frozen_ids = set(read_json(output / 'v1/calibration/selection.json')['pair_ids'])
                self.assertEqual(frozen_ids, {r['pair_id'] for r in source})
                self.assertFalse(frozen_ids & {r['pair_id'] for r in test})
                folds = read_json(output / 'v1/folds.json')
                for fold in folds:
                    self.assertFalse(set(fold['train_ids']) & set(fold['held_out_ids']))
                saved = read_json(output / 'v2/manifest.json')
                self.assertEqual(saved['items'], test)
                self.assertTrue((output / 'comparison.csv').exists())
                broken = output / 'v2/records' / f"{test[0]['pair_id']}.json"
                record = read_json(broken)
                record['conditions']['baseline']['initial']['text'] = 'different baseline history'
                write_json(broken, record)
                self.assertEqual(compare(cfg, test, output / 'v2', reference, output, modules.test)['status'], 'baseline_mismatch')

    def test_source_code_or_config_changes_reject_resume(self):
        source, test = source_rows()[:3], load_split('test')[:2]
        cfg = read_json(HERE / MODEL_DIRS['qwen'] / 'config.json')
        with tempfile.TemporaryDirectory() as temporary, patch('vds_mub.run.prepare', return_value=(source, test, {})):
            run('qwen', cfg, temporary, stage='validate')
            with self.assertRaises(ValueError):
                run('qwen', {**cfg, 'seed': cfg['seed'] + 1}, temporary, stage='validate')
            with patch('vds_mub.run.code_identity', return_value={'changed.py': 'new'}), self.assertRaises(ValueError):
                run('qwen', cfg, temporary, stage='validate')


if __name__ == '__main__':
    unittest.main()
