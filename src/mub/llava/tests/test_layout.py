"""Standalone model layouts, isolation and LLaVA message normalization."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mub.qwen import core as qwen_core
from mub.qwen.v1.run import manifest as qwen_manifest, run as qwen_train
from mub.llava import core
from mub.llava.core import read_json, ensure_manifest
from mub.llava.data import all_items, load_split
from mub.llava.messages import native_messages
from mub.llava.protocol import training_messages, training_options, evaluation_messages
from mub.llava.tests.test_pipeline import FakeEngine
from mub.llava.v1.run import run as train_run
from mub.llava.v2.run import run as test_run
from mub.llava.validate import smoke


class LlavaFakeEngine(FakeEngine):
    def prepare(self, history, image):
        self.calls.append(native_messages(history))
        return history


class LayoutTests(unittest.TestCase):
    def test_model_directories_and_one_dataset(self):
        root = core.HERE.parent
        self.assertEqual({p.name for p in root.iterdir() if p.is_dir() and p.name != '__pycache__'}, {'qwen', 'llava', 'internvl'})
        self.assertEqual(core.DATA, qwen_core.DATA)
        for model_core in [qwen_core, core]:
            for name in ['run.py', 'config.json', 'model.py', 'protocol.py', 'requirements.txt', 'v1/run.py', 'v2/run.py']:
                self.assertTrue((model_core.HERE / name).is_file())
            self.assertEqual(model_core.RESULTS.name, model_core.HERE.name)

    def test_native_messages_preserve_text_and_real_initial_answer(self):
        for row in load_split('train'):
            first = training_options(row)['A']
            for arm in (None, 'dplus', 'dminus'):
                original = training_messages(row, None, arm, initial=first)
                normalized = native_messages(original)
                self.assertEqual([m['role'] for m in original], [m['role'] for m in normalized])
                for before, after in zip(original, normalized):
                    expected = [{'type': 'text', 'text': before['content']}] if isinstance(before['content'], str) else before['content']
                    self.assertEqual(after['content'], expected)
            original = evaluation_messages(row, None, True, initial='My answer is A.')
            normalized = native_messages(original)
            self.assertEqual(normalized[1]['content'][0]['text'], 'My answer is A.')
            self.assertEqual(normalized[2]['content'][0]['text'], original[2]['content'])
        with self.assertRaises(ValueError):
            native_messages([{'role': 'user', 'content': 'No image'}])

    def test_llava_full_pipeline_with_normalized_history_and_resume(self):
        config = read_json(core.CONFIG)
        rows = [{**r, 'true_option': 'A', 'dplus': 'B', 'dminus': 'C'} for r in load_split('train')]
        engine = LlavaFakeEngine()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            v1, v2 = Path(temp) / 'v1', Path(temp) / 'v2'
            with patch('mub.llava.v1.run.load_split', return_value=rows):
                train_run(config, v1, engine=engine)
                train_run(config, v1)
            with patch('mub.llava.v2.run.load_split', return_value=rows), patch('mub.llava.v2.run.all_items', return_value=all_items()[:2]):
                result = test_run(config, v1, v2, engine)
                self.assertEqual(test_run(config, v1, v2), result)
            self.assertIn(result['layer'], [8, 16, 24, 32])
            self.assertTrue(all(h[-2]['role'] == 'assistant' for h in engine.calls if len(h) in (3, 4)))

    def test_cross_model_configs_and_caches_rejected_before_loading(self):
        qwen, llava = read_json(qwen_core.CONFIG), read_json(core.CONFIG)
        with tempfile.TemporaryDirectory() as temp, patch('mub.llava.v1.run.load_engine') as load:
            v1 = Path(temp) / 'v1'
            ensure_manifest(v1, qwen_manifest(qwen, load_split('train')))
            with self.assertRaises(ValueError):
                train_run(llava, v1)
            with self.assertRaises(ValueError):
                train_run(qwen, Path(temp) / 'other')
            with self.assertRaises(ValueError):
                qwen_train(llava, Path(temp) / 'qwen')
            with self.assertRaises(ValueError):
                test_run(llava, v1, Path(temp) / 'v2')
            load.assert_not_called()

    def test_smoke_orchestration_without_result_writes(self):
        class SmokeEngine(LlavaFakeEngine):
            def score(self, *args, **kwargs):
                score, states = super().score(*args, **kwargs)
                return {**score, 'intervention': None}, states

            def generate(self, history, generation, seed, ratio=0., **kwargs):
                self.generation = generation
                return {'text': 'A', 'intervention': {'requested_signed_ratio': ratio} if ratio else None}
        engine = SmokeEngine()
        with patch('mub.llava.validate.load_engine', return_value=engine):
            from mub.llava import protocol
            report = smoke(read_json(core.CONFIG), 'llava', protocol)
        self.assertFalse(report['experiment_results_written'])
        self.assertEqual(engine.generation['max_new_tokens'], 8)


if __name__ == '__main__':
    unittest.main()
