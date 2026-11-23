"""Gemma image preparation and relative-norm hooks on a tiny CPU model."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "PyTorch is required for adapter numerics; run on the server")
class GemmaAdapterNumericsTests(unittest.TestCase):
    def setUp(self):
        import torch
        from steering.gemma.adapter import Gemma431BEngine

        self.torch = torch
        torch.manual_seed(7)
        nn = torch.nn

        class Language(nn.Module):
            hidden_size_per_layer_input = 2

            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(10, 3)
                self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
                with torch.no_grad():
                    for token, values in {
                        1: [6, 8, 0], 2: [3, 0, 4], 3: [0, 9, 12],
                        4: [0, 12, 16], 5: [9, 12, 0], 6: [3, 4, 0],
                    }.items():
                        self.embed_tokens.weight[token] = torch.tensor(values)

            def get_per_layer_inputs(self, input_ids, inputs_embeds):
                assert inputs_embeds is None
                self.per_layer_ids = input_ids.detach().clone()
                return input_ids.float().unsqueeze(-1).expand(-1, -1, 2)

            def forward(self, inputs_embeds, per_layer_inputs, **kwargs):
                self.per_layer_inputs = per_layer_inputs.detach().clone()
                self.mm_token_type_ids = kwargs["mm_token_type_ids"].detach().clone()
                hidden = inputs_embeds
                for layer in self.layers:
                    hidden = layer(hidden)
                self.last_seen = hidden.detach().clone()
                return SimpleNamespace(last_hidden_state=hidden)

        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = Language()

            def forward(self, **kwargs):
                return self.language_model(**kwargs)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Base()
                self.lm_head = nn.Linear(3, 10, bias=False)
                self.config = SimpleNamespace(
                    image_token_id=7,
                    text_config=SimpleNamespace(pad_token_id=0, final_logit_softcapping=9.0),
                )

            def get_image_features(self, pixel_values, **kwargs):
                self.image_position_ids = kwargs["image_position_ids"]
                return SimpleNamespace(pooler_output=(pixel_values,))

        tokenizer = SimpleNamespace(
            eos_token_id=9,
            encode=lambda text, **_: {
                "A": [1], "B": [2], "C": [3], "D": [4],
                "alpha": [1, 2], "beta": [1, 3], "gamma": [4], "delta": [5],
            }[text],
        )
        self.engine = Gemma431BEngine(Model(), SimpleNamespace(tokenizer=tokenizer))
        self.processed = {
            "input_ids": torch.tensor([[0, 7, 6]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "pixel_values": torch.tensor([[1.0, 2.0, 3.0]]),
            "image_position_ids": torch.tensor([[[0, 0]]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
            "position_ids": torch.tensor([[0, 1, 2]]),
        }
        self.condition = self.engine.prepare(self.processed)

    def assert_hooks_removed(self):
        self.assertTrue(all(not layer._forward_hooks for layer in self.engine.decoder_layers))

    def test_image_features_and_per_layer_inputs_follow_gemma_path(self):
        torch = self.torch
        self.engine._forward_once(self.processed, self.condition)
        language = self.engine.model.model.language_model
        torch.testing.assert_close(language.last_seen[0, 1], self.processed["pixel_values"][0])
        torch.testing.assert_close(language.per_layer_ids, torch.tensor([[0, 0, 6]]))
        torch.testing.assert_close(language.mm_token_type_ids, self.processed["mm_token_type_ids"])
        self.assertIs(self.engine.model.image_position_ids, self.processed["image_position_ids"])

    def test_image_token_feature_mismatch_is_rejected(self):
        from steering.gemma.adapter import FullVisualCondition

        wrong = FullVisualCondition(self.torch.ones((2, 3)))
        with self.assertRaisesRegex(RuntimeError, "Image tokens/features differ"):
            self.engine._forward_once(self.processed, wrong)
        self.assert_hooks_removed()

    def test_intervention_changes_only_last_query_and_layer_numbers_start_at_one(self):
        torch = self.torch
        self.engine._forward_once(self.processed, self.condition)
        before = self.engine.model.model.language_model.last_seen.clone()
        _logits, states, metrics = self.engine._forward_once(
            self.processed, self.condition, capture_layers=[2], intervention_layer=1,
            intervention_direction=torch.tensor([0.0, 0.0, 2.0]), intervention_ratio=-0.2,
        )
        after = self.engine.model.model.language_model.last_seen
        torch.testing.assert_close(after[:, :-1], before[:, :-1])
        torch.testing.assert_close(states[2], torch.tensor([3.0, 4.0, -1.0]))
        self.assertAlmostEqual(metrics["pre_hidden_norm"], 5.0)
        self.assertAlmostEqual(metrics["effective_ratio"], 0.2)
        self.assert_hooks_removed()
        with self.assertRaises(ValueError):
            self.engine._forward_once(self.processed, self.condition, capture_layers=[0])

    def test_forward_failure_removes_hooks(self):
        torch = self.torch
        with patch.object(self.engine.model.model, "forward", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                self.engine._forward_once(
                    self.processed, self.condition, capture_layers=[2], intervention_layer=1,
                    intervention_direction=torch.tensor([0.0, 0.0, 1.0]), intervention_ratio=0.1,
                )
        self.assert_hooks_removed()

    def test_multitoken_prefixes_recompute_norm_and_extend_gemma_input_fields(self):
        torch = self.torch
        prefixed = self.engine._with_prefix(self.processed, (1, 2))
        torch.testing.assert_close(prefixed["position_ids"], torch.tensor([[0, 1, 2, 3, 4]]))
        torch.testing.assert_close(prefixed["mm_token_type_ids"], torch.tensor([[0, 1, 0, 0, 0]]))
        self.assertEqual(prefixed["generation_query_index"], 4)
        result = self.engine.score_actual_options(
            self.processed, dict(zip(("GOLD", "Dplus", "Dminus", "other"),
                                     ("alpha", "beta", "gamma", "delta"))),
            prepared=self.condition, intervention_layer=1,
            intervention_direction=torch.tensor([0.0, 0.0, 1.0]), intervention_ratio=0.1,
        )
        metrics = result["intervention_metrics"]
        values = [metrics["root"], *metrics["prefixes"]]
        self.assertGreater(len(values), 1)
        self.assertGreater(len({value["pre_hidden_norm"] for value in values}), 1)
        for value in values:
            self.assertAlmostEqual(value["effective_ratio"], 0.1, places=6)
        self.assert_hooks_removed()

    def test_final_logits_preserve_gemma_softcap(self):
        torch = self.torch
        logits, _states, _metric = self.engine._forward_once(self.processed, self.condition)
        hidden = self.engine.model.model.language_model.last_seen[0, -1]
        expected = torch.tanh(self.engine.model.lm_head(hidden) / 9.0) * 9.0
        torch.testing.assert_close(logits, expected.float())


if __name__ == "__main__":
    unittest.main()
