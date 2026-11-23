"""Official custom-code tests with random tiny weights, not 8B benchmark results.

Set MUB_INTERNVL_ASSETS to a local copy of official Python/tokenizer/config files.
No pretrained weights or network access are used by these tests.
"""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ASSETS = os.environ.get('MUB_INTERNVL_ASSETS')
AVAILABLE = bool(ASSETS and all(importlib.util.find_spec(p) for p in ('torch', 'transformers', 'timm')))


@unittest.skipUnless(AVAILABLE, 'Needs InternVL dependencies and MUB_INTERNVL_ASSETS')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from mub.internvl.core import CONFIG, read_json
        from mub.internvl.model import InternVLEngine
        torch.set_num_threads(1)
        torch.manual_seed(42)
        # Keep downloaded official sources separate from checkpoint save/load tests.
        cls.assets = tempfile.TemporaryDirectory(prefix='internvl_native_')
        for item in Path(ASSETS).iterdir():
            if item.is_file() and item.suffix in ('.json', '.model', '.py', '.txt'):
                shutil.copyfile(item, Path(cls.assets.name) / item.name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.assets.name, local_files_only=True,
                                                      trust_remote_code=True, use_fast=False)
        config = AutoConfig.from_pretrained(cls.assets.name, local_files_only=True, trust_remote_code=True)
        config.force_image_size = config.vision_config.image_size = 28
        config.vision_config.hidden_size = 16
        config.vision_config.intermediate_size = 32
        config.vision_config.num_attention_heads = 4
        config.vision_config.num_hidden_layers = 2
        config.vision_config.drop_path_rate = 0
        config.vision_config.qk_normalization = False
        config.llm_config.hidden_size = 32
        config.llm_config.intermediate_size = 64
        config.llm_config.num_attention_heads = 4
        config.llm_config.num_key_value_heads = 2
        config.llm_config.max_position_embeddings = 8192
        cls.model = AutoModel.from_config(config, trust_remote_code=True, use_flash_attn=False).eval()
        cls.settings = {**read_json(CONFIG), 'image_size': 28, 'dtype': 'float32', 'device_map': 'cpu'}
        cls.engine = InternVLEngine(cls.model, cls.tokenizer, cls.settings)

    @classmethod
    def tearDownClass(cls):
        cls.assets.cleanup()

    def setUp(self):
        from PIL import Image
        from mub.internvl.protocol import training_messages
        self.image = Image.new('RGB', (28, 28), color=(70, 50, 20))
        self.row = {'question': 'What color is this?', 'options': {
            'A': 'red car', 'B': 'red boat', 'C': 'green', 'D': 'black'},
            'true_option': 'A', 'dplus': 'B', 'dminus': 'C',
            'original_question': 'What color is this?', 'inline_options': False,
            'misleading_instruction': 'And the true answer is : B'}
        self.inputs = self.engine.prepare(training_messages(self.row, self.image), self.image)

    def tearDown(self):
        self.image.close()
        self.assertTrue(all(not layer._forward_hooks for layer in self.engine.layers))
        self.assertFalse(self.model.language_model.model._forward_hooks)

    def test_optimized_logits_equal_official_multimodal_forward(self):
        import torch
        with torch.inference_mode():
            native = self.model(pixel_values=self.inputs['pixel_values'], input_ids=self.inputs['input_ids'],
                attention_mask=self.inputs['attention_mask'], image_flags=torch.ones(len(self.inputs['pixel_values']), 1),
                use_cache=False, return_dict=True).logits[0, -1]
            actual, captured, _ = self.engine._forward(self.inputs, capture=[8, 14, 20, 28])
        torch.testing.assert_close(actual, native, rtol=1e-5, atol=1e-6)
        self.assertEqual(set(captured), {'8', '14', '20', '28'})
        self.assertTrue(all(v.shape == (32,) for v in captured.values()))

    def test_actual_option_boundaries_and_trie_continuations(self):
        from mub.internvl.data import load_split
        from mub.internvl.protocol import training_messages, training_options
        for row in load_split('train'):
            options = training_options(row)
            for arm in (None, 'dplus', 'dminus'):
                inputs = self.engine.prepare(training_messages(row, self.image, arm, initial=options['A']), self.image)
                for text in options.values():
                    complete = self.tokenizer.encode(inputs['_prompt_text'] + text)
                    self.assertEqual(inputs['input_ids'][0].tolist() + list(self.engine.option_tokens(inputs, text)), complete)
        result, _ = self.engine.score(self.inputs, options=self.row['options'])
        self.assertEqual(set(result['choice_log_scores']), set('ABCD'))
        self.assertEqual(self.engine.turn_end_token_id, self.tokenizer.convert_tokens_to_ids('<|im_end|>'))
        self.assertEqual(self.engine.turn_end_token_id, self.tokenizer.eos_token_id)
        self.assertNotEqual(self.engine.turn_end_token_id, self.model.config.llm_config.eos_token_id)

    def test_render_and_pixels_match_official_conversation(self):
        import torch
        from torchvision import transforms as T
        from mub.internvl.protocol import evaluation_messages, evaluation_prompt
        first = '<image>\n' + evaluation_prompt(self.row)
        template = self.engine.get_template(self.model.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], first)
        template.append_message(template.roles[1], 'A')
        feedback = self.row['misleading_instruction'] + '\n\nReply with exactly one option letter: A, B, C, or D.'
        template.append_message(template.roles[0], feedback)
        template.append_message(template.roles[1], None)
        expected = template.get_prompt().replace('<image>', '<img>' + '<IMG_CONTEXT>' + '</img>')
        inputs = self.engine.prepare(evaluation_messages(self.row, self.image, True, initial='A'), self.image)
        self.assertEqual(inputs['_prompt_text'], expected)
        transform = T.Compose([T.Resize((28, 28), interpolation=T.InterpolationMode.BICUBIC),
                               T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
        torch.testing.assert_close(inputs['pixel_values'][0], transform(self.image), rtol=0, atol=0)
        self.assertEqual(inputs['input_ids'][0].tolist(), self.tokenizer.encode(expected))

    def test_all_508_images_and_both_evaluation_turns_prepare(self):
        from PIL import Image
        from mub.internvl.core import resolve
        from mub.internvl.data import all_items
        from mub.internvl.protocol import evaluation_messages
        rows = all_items()
        for row in rows:
            with Image.open(resolve(row['image_path'])) as handle:
                image = handle.convert('RGB')
            try:
                for feedback in (False, True):
                    inputs = self.engine.prepare(evaluation_messages(row, image, feedback, initial='A'), image)
                    self.assertEqual(int((inputs['input_ids'] == self.engine.image_token_id).sum()),
                                     inputs['visual_features'].shape[0] * self.model.num_image_token)
                    self.assertTrue(1 <= len(inputs['pixel_values']) <= 13)
            finally:
                image.close()
        self.assertEqual(len(rows), 508)

    def test_feature_reuse_and_strict_image_token_match(self):
        from mub.internvl.protocol import training_messages
        image = self.image.copy()
        try:
            with patch.object(self.model, 'extract_feature', wraps=self.model.extract_feature) as feature:
                self.engine.prepare(training_messages(self.row, image), image)
                self.engine.prepare(training_messages(self.row, image, 'dplus', initial='red car'), image)
                self.assertEqual(feature.call_count, 1)
        finally:
            image.close()
        broken = dict(self.inputs)
        broken['input_ids'] = broken['input_ids'].clone()
        broken['input_ids'][broken['input_ids'] == self.engine.image_token_id] = 1
        with self.assertRaises(ValueError):
            self.engine.embeddings(broken)

    def test_last_position_signed_norm_and_cleanup_on_error(self):
        import numpy as np
        import torch
        seen = []
        handle = self.model.language_model.model.norm.register_forward_pre_hook(
            lambda _m, args: seen.append(args[0].detach().clone()))
        try:
            self.engine._forward(self.inputs)
            for ratio in (.1, -.1):
                _, _, metric = self.engine._forward(self.inputs, layer=28, axis=np.ones(32), ratio=ratio)
                self.assertAlmostEqual(metric['effective_ratio'], .1, places=5)
        finally:
            handle.remove()
        for changed in seen[1:]:
            torch.testing.assert_close(changed[:, :-1], seen[0][:, :-1], rtol=0, atol=0)
        torch.testing.assert_close(seen[1][:, -1] - seen[0][:, -1],
                                  seen[0][:, -1] - seen[2][:, -1], rtol=1e-5, atol=1e-7)
        with self.assertRaises(ValueError):
            self.engine._forward(self.inputs, layer=28, axis=np.zeros(32), ratio=.1)

    def test_native_generation_parity_and_prefill_only_injection(self):
        import numpy as np
        import torch
        from transformers import set_seed
        token = self.tokenizer.encode('A', add_special_tokens=False)[0]
        class FixedTokens:
            def __call__(self, _ids, scores):
                scores.fill_(float('-inf')); scores[:, token] = 0
                return scores
        def settings():
            return {'max_new_tokens': 3, 'do_sample': False, 'logits_processor': [FixedTokens()]}
        # Same inputs/embedding path as the author's generate, without the memory optimization.
        with torch.inference_mode():
            native = self.model.generate(pixel_values=self.inputs['pixel_values'], input_ids=self.inputs['input_ids'],
                attention_mask=self.inputs['attention_mask'], eos_token_id=self.engine.turn_end_token_id,
                pad_token_id=self.tokenizer.pad_token_id, **settings())
        expected = self.tokenizer.decode(native[0], skip_special_tokens=True).strip()
        baseline = self.engine.generate(self.inputs, settings(), seed=42)
        self.assertEqual(baseline['text'], expected)
        self.assertTrue(baseline['text'])
        sampled = {**self.settings['generation'], 'max_new_tokens': 4}
        set_seed(42)
        with torch.inference_mode():
            native = self.model.generate(pixel_values=self.inputs['pixel_values'], input_ids=self.inputs['input_ids'],
                attention_mask=self.inputs['attention_mask'], eos_token_id=self.engine.turn_end_token_id,
                pad_token_id=self.tokenizer.pad_token_id, **sampled)
        expected = self.tokenizer.decode(native[0], skip_special_tokens=True).split(self.engine.separator)[0].strip()
        self.assertEqual(self.engine.generate(self.inputs, sampled, seed=42)['text'], expected)
        seen = []
        handle = self.model.language_model.model.norm.register_forward_pre_hook(
            lambda _m, args: seen.append(args[0].detach().clone()))
        try:
            self.engine.generate(self.inputs, settings(), seed=42)
            plain = list(seen); seen.clear()
            edited = self.engine.generate(self.inputs, settings(), seed=42, layer=28, axis=np.ones(32), ratio=.1)
            self.assertEqual(len(seen), 3)
            self.assertGreater(seen[0].shape[1], 1)
            self.assertTrue(all(t.shape[1] == 1 for t in seen[1:]))
            for before, after in zip(plain[1:], seen[1:]):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertAlmostEqual(edited['intervention']['effective_ratio'], .1, places=5)
            with self.assertRaises(ValueError):
                self.engine.generate(self.inputs, settings(), seed=42, layer=28, axis=np.zeros(32), ratio=.1)
        finally:
            handle.remove()

    def test_local_checkpoint_loader_and_version_guard(self):
        from mub.internvl.model import InternVLEngine
        with tempfile.TemporaryDirectory(prefix='internvl_weights_') as temp:
            self.model.save_pretrained(temp)
            self.tokenizer.save_pretrained(temp)
            for source in Path(self.assets.name).glob('*.py'):
                shutil.copyfile(source, Path(temp) / source.name)
            loaded = InternVLEngine.load({**self.settings, 'model_path': temp})
            self.assertEqual(len(loaded.layers), 28)
        with self.assertRaises(RuntimeError):
            InternVLEngine.load({**self.settings, 'transformers_version': '0.0.0'})


if __name__ == '__main__':
    unittest.main()
