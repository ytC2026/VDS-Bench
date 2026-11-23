"""Exercise the real adapter's scoring interface without loading a checkpoint."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def load_adapter():
    # The fake backend only supplies the import-time decorator. Forward passes
    # are stubbed below; this test makes no claim about PyTorch numerics.
    fake_torch = SimpleNamespace(inference_mode=lambda: lambda function: function)
    name = "steering.muse._adapter_interface_test"
    path = Path(__file__).resolve().parents[1] / "adapter.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": fake_torch, name: module}):
        spec.loader.exec_module(module)
    return module


class AdapterInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def engine(self):
        engine = self.adapter.MuseGlimmerEngine.__new__(self.adapter.MuseGlimmerEngine)
        tokens = {"one answer": [1, 2], "one other": [1, 3], "third": [4], "fourth": [5]}
        engine.tokenizer = SimpleNamespace(encode=lambda text, **_: tokens[text], eos_token_id=99)
        engine.turn_end_token_id = 99
        engine.option_token_ids = dict(zip("ABCD", [1, 2, 3, 4]))
        engine.calls, engine.preparations = [], []
        condition = object()

        def prepare(processed):
            engine.preparations.append(processed)
            return condition

        class Logits:
            def __getitem__(self, token):
                value = {1: 2.0, 2: 3.0, 3: 0.5, 4: 1.0, 5: 0.0, 99: 0.0}[token]
                return SimpleNamespace(item=lambda: value)

        def forward(processed, prepared, **kwargs):
            engine.calls.append((processed, prepared, kwargs))
            states = {layer: np.array([3, 4, 0], dtype=np.float32)
                      for layer in kwargs.get("capture_layers", ())}
            ratio = kwargs.get("intervention_ratio")
            metric = None if ratio is None else {"effective_ratio": abs(ratio)}
            return Logits(), states, metric

        engine.prepare = prepare
        engine._forward_once = forward
        engine._with_prefix = lambda processed, prefix: {**processed, "prefix": prefix}
        return engine, condition

    def test_text_scoring_reuses_prepared_image_and_injects_every_prefix(self):
        engine, condition = self.engine()
        candidates = dict(zip(("GOLD", "Dplus", "Dminus", "other"),
                              ("one answer", "one other", "third", "fourth")))
        direction = np.array([0, 0, 1], dtype=np.float32)
        result = engine.score_actual_options(
            {"input_ids": np.zeros((1, 3))}, candidates, prepared=condition,
            capture_layers=[2], intervention_layer=2,
            intervention_direction=direction, intervention_ratio=-0.2,
        )
        self.assertEqual(result["answer_key"], "GOLD")
        self.assertEqual(result["answer_text"], "one answer")
        self.assertEqual(set(result["states"]), {2})
        self.assertTrue(all(math.isfinite(value) for value in result["choice_log_scores"].values()))
        self.assertAlmostEqual(sum(result["choice_probabilities"].values()), 1.0)
        self.assertEqual(engine.preparations, [])
        self.assertGreater(len(engine.calls), 1)
        for _processed, prepared, kwargs in engine.calls:
            self.assertIs(prepared, condition)
            self.assertIs(kwargs["intervention_direction"], direction)
            self.assertEqual(kwargs["intervention_ratio"], -0.2)
        self.assertEqual(result["intervention_metrics"]["root"]["effective_ratio"], 0.2)
        self.assertTrue(result["intervention_metrics"]["prefixes"])

    def test_text_scoring_can_prepare_once_and_letter_scoring_uses_same_interface(self):
        engine, condition = self.engine()
        processed = {"input_ids": np.zeros((1, 3))}
        engine.score_actual_options(processed, {"GOLD": "third", "Dplus": "fourth"})
        self.assertEqual(engine.preparations, [processed])
        result = engine.score_letter_options(processed, condition)
        self.assertEqual(result["answer"], "B")
        self.assertEqual(set(result["option_logits"]), set("ABCD"))
        self.assertIsNone(result["intervention_metrics"])

    def test_loader_passes_model_configuration(self):
        config = {"model_path": "Muse_Glimmer_30B", "dtype": "float32",
                  "device_map": "cuda:1", "attn_implementation": "eager",
                  "reasoning_strength": "medium"}
        with patch.object(self.adapter.MuseGlimmerEngine, "load", return_value="engine") as load:
            self.assertEqual(self.adapter.load_engine(config), "engine")
        self.assertEqual(load.call_args.kwargs,
                         {"dtype": "float32", "device_map": "cuda:1", "attn_implementation": "eager",
                          "reasoning_strength": "medium"})


if __name__ == "__main__":
    unittest.main()
