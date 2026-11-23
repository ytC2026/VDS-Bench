"""Full 30/508 orchestration with a fake engine; never produce model results."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from mub.internvl import core
from mub.internvl.core import read_json
from mub.internvl.data import all_items, load_split
from mub.internvl.messages import render_messages
from mub.internvl.preprocess import image_tiles
from mub.internvl.protocol import training_messages, training_options
from mub.internvl.tests.test_pipeline import FakeEngine
from mub.internvl.v1.run import run as train_run
from mub.internvl.v2.run import run as test_run


class Template:
    roles = ('user', 'assistant')
    system_message = 'native default'
    def __init__(self): self.messages = []
    def append_message(self, role, text): self.messages.append((role, text))
    def get_prompt(self): return self.system_message + repr(self.messages)


class IntegrationTests(unittest.TestCase):
    def test_shared_dataset_and_model_scoped_settings(self):
        from mub.qwen.core import DATA as qwen_data
        from mub.llava.core import DATA as llava_data
        self.assertEqual(core.DATA, qwen_data)
        self.assertEqual(core.DATA, llava_data)
        config = read_json(core.CONFIG)
        self.assertEqual(config['decoder_layers'], [8, 14, 20, 28])
        self.assertEqual(config['generation']['max_new_tokens'], 256)
        self.assertEqual(core.RESULTS.name, 'internvl')
        from mub.internvl.adapter import check_backend
        with self.assertRaises(ValueError):
            check_backend({**config, 'backend': 'internvl2_5'})
        for other in ['qwen', 'llava']:
            with self.assertRaises(ValueError):
                check_backend(read_json(core.HERE.parent / other / 'config.json'))

    def test_rendering_preserves_system_answer_and_feedback(self):
        for row in load_split('train'):
            options = training_options(row)
            for arm in ('dplus', 'dminus'):
                messages = training_messages(row, None, arm, initial=options['A'])
                template = Template()
                rendered = render_messages(messages, template, tiles=3, image_tokens=4)
                self.assertEqual(rendered.count('<IMG_CONTEXT>'), 12)
                self.assertEqual(template.system_message, messages[0]['content'])
                self.assertEqual(template.messages[1], ('assistant', options['A']))
                self.assertEqual(template.messages[2], ('user', messages[-1]['content']))
                self.assertEqual(template.messages[-1], ('assistant', None))
        with self.assertRaises(ValueError):
            render_messages([{'role': 'user', 'content': 'No image'}], Template(), tiles=1, image_tokens=4)

    def test_dynamic_tiles_extreme_sizes_and_thumbnail(self):
        for size in [(30, 30), (1200, 100), (100, 1200), (1000, 700)]:
            image = Image.new('RGB', size)
            tiles = image_tiles(image, image_size=28, max_tiles=12)
            try:
                self.assertTrue(1 <= len(tiles) <= 13)
                self.assertTrue(all(tile.size == (28, 28) for tile in tiles))
                plain = image_tiles(image, image_size=28, max_tiles=12, thumbnail=False)
                self.assertEqual(len(tiles), len(plain) + (len(plain) > 1))
                for tile in plain: tile.close()
            finally:
                image.close()
                for tile in tiles: tile.close()

    def test_full_508_evaluation_resume_and_frozen_history(self):
        config = read_json(core.CONFIG)
        # Synthetic training roles make fake calibration deterministic.
        train = [{**r, 'true_option': 'A', 'dplus': 'B', 'dminus': 'C'} for r in load_split('train')]
        rows = all_items()
        class CountingEngine(FakeEngine):
            def __init__(self): super().__init__(); self.generated = 0
            def generate(self, *args, **kwargs):
                self.generated += 1
                return super().generate(*args, **kwargs)
        engine = CountingEngine()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            v1, v2 = Path(temp) / 'v1', Path(temp) / 'v2'
            with patch('mub.internvl.v1.run.load_split', return_value=train):
                train_run(config, v1, engine=engine)
            with patch('mub.internvl.v2.run.load_split', return_value=train):
                result = test_run(config, v1, v2, engine)
                self.assertEqual(test_run(config, v1, v2), result)
            self.assertEqual(engine.generated, 508 * 14)
            self.assertEqual(len(list((v2 / 'records').glob('*.json'))), 508)
            self.assertEqual(result['scores']['train']['items'], 30)
            self.assertEqual(result['scores']['test']['items'], 478)
            self.assertTrue(all(h[1]['content'] == 'A' for h in engine.calls if len(h) == 3))


if __name__ == '__main__':
    unittest.main()
