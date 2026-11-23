"""Optional real HF CPU interface tests with random tiny weights, never 7B results.

Set MUB_LLAVA_PROCESSOR to a local copy of the official processor/tokenizer
assets. No network access or pretrained weight download occurs in these tests.
"""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ASSETS = os.environ.get('MUB_LLAVA_PROCESSOR')
AVAILABLE = bool(ASSETS and all(importlib.util.find_spec(p) for p in ('torch', 'transformers', 'accelerate')))


@unittest.skipUnless(AVAILABLE, 'Needs torch/transformers and MUB_LLAVA_PROCESSOR local assets')
class NativeLlavaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import (AutoProcessor, CLIPVisionConfig, LlavaNextConfig,
                                  LlavaNextForConditionalGeneration, MistralConfig)
        from mub.llava.model import LlavaNextEngine
        torch.set_num_threads(1)
        torch.manual_seed(123)
        cls.processor = AutoProcessor.from_pretrained(ASSETS, local_files_only=True)
        # Keep the official tokenizer/template; shrink only random weights and pixels.
        image_processor = cls.processor.image_processor
        image_processor.size = {'shortest_edge': 28}
        image_processor.crop_size = {'height': 28, 'width': 28}
        image_processor.image_grid_pinpoints = [[28, 28]]
        text = MistralConfig(vocab_size=32064, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=4096, sliding_window=None)
        vision = CLIPVisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                                  num_attention_heads=2, image_size=28, patch_size=14)
        config = LlavaNextConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
                                 image_token_index=32000, image_grid_pinpoints=[[28, 28]],
                                 vision_feature_layer=-2)
        config._attn_implementation = 'sdpa'
        model = LlavaNextForConditionalGeneration(config).eval()
        cls.engine = LlavaNextEngine(model, cls.processor)
        cls.model = model

    def setUp(self):
        from PIL import Image
        from mub.llava.protocol import training_messages
        self.image = Image.new('RGB', (28, 28), color=(120, 40, 10))
        self.row = {'pair_id': 'random-interface-only', 'question': 'What color is the object?',
                    'options': {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'black'},
                    'true_option': 'A', 'dplus': 'B', 'dminus': 'C',
                    'original_question': 'What color is the object?', 'inline_options': False,
                    'misleading_instruction': 'And the true answer is : B'}
        self.inputs = self.engine.prepare(training_messages(self.row, self.image), self.image)

    def tearDown(self):
        self.image.close()
        self.assertTrue(all(not layer._forward_hooks for layer in self.engine.layers))

    def test_native_processor_full_forward_and_paired_content_scoring(self):
        import numpy as np
        from mub.llava.protocol import training_messages
        initial, states = self.engine.score(self.inputs, options=self.row['options'], capture=[8, 16, 24, 32])
        self.assertEqual(set(states), {'8', '16', '24', '32'})
        self.assertTrue(all(s.shape == (16,) and np.isfinite(s).all() for s in states.values()))
        for arm in ('dplus', 'dminus'):
            inputs = self.engine.prepare(training_messages(self.row, self.image, arm, initial=initial['answer_text']), self.image)
            result, captured = self.engine.score(inputs, options=self.row['options'], capture=[32])
            self.assertEqual(set(result['choice_log_scores']), set('ABCD'))
            self.assertEqual(captured['32'].shape, (16,))

    def test_actual_mistral_option_encoding_matches_completion_boundary(self):
        from mub.llava.data import load_split
        from mub.llava.protocol import training_messages, training_options
        from mub.llava.messages import native_messages
        tokenizer = self.processor.tokenizer
        for row in load_split('train'):
            options = training_options(row)
            for arm in (None, 'dplus', 'dminus'):
                text = self.processor.apply_chat_template(native_messages(
                    training_messages(row, None, arm, initial=options['A'])),
                    tokenize=False, add_generation_prompt=True)
                prefix = tokenizer.encode(text)
                for value in options.values():
                    self.assertEqual(tokenizer.encode(text + ' ' + value),
                                     prefix + tokenizer.encode(value, add_special_tokens=False))

    def test_last_position_only_signed_norm_ratio_and_hook_cleanup(self):
        import torch
        import numpy as np
        states = []
        handle = self.model.model.language_model.norm.register_forward_pre_hook(
            lambda _m, args: states.append(args[0].detach().clone()))
        try:
            self.engine._forward(self.inputs)
            for ratio in (.1, -.1):
                _, _, metric = self.engine._forward(self.inputs, layer=32, axis=np.ones(16), ratio=ratio)
                self.assertAlmostEqual(metric['effective_ratio'], .1, places=5)
        finally:
            handle.remove()
        baseline, positive, negative = states
        torch.testing.assert_close(positive[:, :-1], baseline[:, :-1], rtol=0, atol=0)
        torch.testing.assert_close(negative[:, :-1], baseline[:, :-1], rtol=0, atol=0)
        torch.testing.assert_close(positive[:, -1] - baseline[:, -1],
                                   baseline[:, -1] - negative[:, -1], rtol=1e-5, atol=1e-7)
        with self.assertRaises(ValueError):
            self.engine._forward(self.inputs, layer=32, axis=np.zeros(16), ratio=.1)

    def test_native_generation_injects_once_at_prefill(self):
        import numpy as np
        from mub.llava.protocol import evaluation_messages
        seen = []
        # At the final layer, later decoding input is unaffected by a previous
        # output edit: its KV entries were computed before that edit.
        hook = self.model.model.language_model.norm.register_forward_pre_hook(
            lambda _m, args: seen.append(args[0].detach().clone()))
        token = self.processor.tokenizer.encode('red', add_special_tokens=False)[0]
        class FixedContinuation:
            def __call__(self, _ids, scores):
                scores.fill_(float('-inf'))
                scores[:, token] = 0
                return scores
        generation = {'max_new_tokens': 3, 'do_sample': False, 'eos_token_id': None,
                      'pad_token_id': self.processor.tokenizer.pad_token_id,
                      'logits_processor': [FixedContinuation()]}
        try:
            first = self.engine.generate(self.inputs, generation, seed=42)
            baseline = list(seen)
            seen.clear()
            steered = self.engine.generate(self.inputs, generation, seed=42,
                                           layer=32, axis=np.ones(16), ratio=.1)
            self.assertEqual(len(seen), 3)
            self.assertGreater(seen[0].shape[1], 1)
            self.assertTrue(all(s.shape[1] == 1 for s in seen[1:]))
            self.assertAlmostEqual(steered['intervention']['effective_ratio'], .1, places=5)
            import torch
            for plain, edited in zip(baseline[1:], seen[1:]):
                torch.testing.assert_close(plain, edited, rtol=0, atol=0)
            # Exercise the actual assistant-string -> native-list conversion.
            history = evaluation_messages(self.row, self.image, True, initial=first['text'])
            inputs = self.engine.prepare(history, self.image)
            self.engine.generate(inputs, generation, seed=42, layer=32, axis=np.ones(16), ratio=-.1)
        finally:
            hook.remove()

    def test_local_loader_and_legacy_qwen_import(self):
        from mub.llava.adapter import load_engine
        from mub.llava.core import CONFIG, read_json
        from mub.qwen.model import Qwen25Engine
        self.assertTrue(callable(Qwen25Engine.load))
        with tempfile.TemporaryDirectory() as temp:
            self.model.save_pretrained(temp)
            self.processor.save_pretrained(temp)
            config = {**read_json(CONFIG), 'model_path': str(Path(temp).resolve()),
                      'dtype': 'float32', 'device_map': 'cpu'}
            loaded = load_engine(config, 'llava')
            self.assertEqual(len(loaded.layers), 32)
            inputs = loaded.prepare([{'role': 'user', 'content': [
                {'type': 'image', 'image': self.image}, {'type': 'text', 'text': 'What color?'}]}], self.image)
            _, states, _ = loaded._forward(inputs, capture=[32])
            self.assertEqual(states['32'].shape, (16,))


if __name__ == '__main__':
    unittest.main()
