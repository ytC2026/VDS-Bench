"""Tiny CPU-model checks; these require PyTorch but no model weights."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
MODEL = Path(__file__).resolve().parents[1].name
TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "PyTorch is required for adapter numerics; run on the server")
class AdapterNumericsTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        torch.manual_seed(7)
        nn = torch.nn
        adapter = importlib.import_module(f"steering.{MODEL}.adapter")

        class Vision(nn.Module):
            spatial_merge_size = 1

            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Linear(3, 3, bias=False)
                self.blocks = nn.ModuleList([nn.Identity()])

            def forward(self, pixels, **_kwargs):
                return SimpleNamespace(pooler_output=pixels)

        class Language(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(10, 3)
                self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
                self.norm = nn.Identity()
                with torch.no_grad():
                    self.embed_tokens.weight[1] = torch.tensor([6, 8, 0])
                    self.embed_tokens.weight[2] = torch.tensor([3, 0, 4])
                    self.embed_tokens.weight[3] = torch.tensor([0, 9, 12])
                    self.embed_tokens.weight[4] = torch.tensor([0, 12, 16])
                    self.embed_tokens.weight[5] = torch.tensor([9, 12, 0])
                    self.embed_tokens.weight[6] = torch.tensor([3, 4, 0])

            def forward(self, inputs_embeds, **_kwargs):
                hidden = inputs_embeds
                for layer in self.layers:
                    hidden = layer(hidden)
                self.last_seen = hidden.detach().clone()
                return SimpleNamespace(last_hidden_state=self.norm(hidden))

        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = Vision()
                self.vision_tower = Vision()
                self.language_model = Language()

            def get_rope_index(self, input_ids, **_kwargs):
                positions = torch.arange(input_ids.shape[1]).reshape(1, 1, -1)
                return positions.expand(3, 1, -1), None

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Base()
                self.lm_head = nn.Linear(3, 10, bias=False)
                self.config = SimpleNamespace(
                    image_token_id=7, video_token_id=8,
                    text_config=SimpleNamespace(output_multiplier=1.7, final_logit_softcapping=9.0),
                )

            def get_image_features(self, pixel_values, **_kwargs):
                return SimpleNamespace(pooler_output=(pixel_values,))

        tokenizer = SimpleNamespace(
            eos_token_id=9,
            encode=lambda text, **_: {
                "A": [1], "B": [2], "C": [3], "D": [4],
                "alpha": [1, 2], "beta": [1, 3], "gamma": [4], "delta": [5],
            }[text],
        )
        model = Model().eval()
        if MODEL == "qwen":
            self.engine = adapter.QwenActualTextEngine(model, SimpleNamespace(tokenizer=tokenizer))
        else:
            self.engine = adapter.MuseGlimmerEngine.__new__(adapter.MuseGlimmerEngine)
            self.engine.model, self.engine.tokenizer = model, tokenizer
            self.engine.option_token_ids = dict(zip("ABCD", [1, 2, 3, 4]))
            self.engine.turn_end_token_id = 9
        self.processed = {
            "input_ids": torch.tensor([[0, 7, 6]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "pixel_values": torch.tensor([[1.0, 2.0, 3.0]]),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }
        self.condition = self.engine.prepare(self.processed)

    def assert_hooks_removed(self):
        self.assertTrue(all(not layer._forward_hooks for layer in self.engine.decoder_layers))

    def test_hook_changes_only_last_query_at_current_relative_norm(self):
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
        self.assertAlmostEqual(metrics["effective_delta_norm"], 1.0)
        self.assertAlmostEqual(metrics["effective_ratio"], 0.2)
        self.assert_hooks_removed()

    def test_forward_failure_removes_capture_and_intervention_hooks(self):
        torch = self.torch
        with patch.object(self.engine.model.model.language_model, "forward", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                self.engine._forward_once(
                    self.processed, self.condition, capture_layers=[2], intervention_layer=1,
                    intervention_direction=torch.tensor([0.0, 0.0, 1.0]), intervention_ratio=0.1,
                )
        self.assert_hooks_removed()

    def test_multitoken_scoring_recomputes_norm_for_every_prefix(self):
        torch = self.torch
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

    def test_final_logits_preserve_model_output_scaling(self):
        torch = self.torch
        logits, _states, _metric = self.engine._forward_once(self.processed, self.condition)
        hidden = self.engine.model.model.language_model.last_seen[0, -1]
        expected = self.engine.model.lm_head(hidden)
        if MODEL == "muse":
            expected = torch.tanh(expected * 1.7 / 9.0) * 9.0
        torch.testing.assert_close(logits, expected.float())


if __name__ == "__main__":
    unittest.main()
