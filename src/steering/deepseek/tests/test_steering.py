"""CPU checks for the paired estimator and shared DeepSeek inference protocol."""

from __future__ import annotations

import copy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from steering.deepseek.v1.run import STAGES, collect_item, pressure_states
from steering.deepseek.v1.extract import fit_direction, fit_states, make_folds
from steering.deepseek.run import parse_args
from steering.deepseek.v1.run import parse_args as parse_v1_args
from steering.deepseek.v2.run import parse_args as parse_v2_args
from steering.deepseek.data import load_pairs
from steering.deepseek.io import ROOT, read_json
from steering.deepseek.v1.calibrate import calibrate_item
from steering.deepseek.v2.run import role_options, run_item, signed_ratio_conditions, validate_complete_row


class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"pair_{i}" for i in range(18)]
        self.categories = [f"category_{i // 6}" for i in range(18)]
        self.states = np.zeros((18, 2, 2, 3), dtype=np.float32)
        self.states[:, 0] = [12, 14, 8]
        self.states[:, 1] = [11, 12, 8]

    def fit(self, states=None, layers=None):
        return fit_states(
            self.states if states is None else states, self.ids, self.categories,
            [10, 16] if layers is None else layers, seed=314159,
        )

    def test_raw_pair_formula_and_no_orthogonalization(self):
        result = self.fit()
        for category in ("pooled",):
            np.testing.assert_allclose(result["categories"][category]["selective"], np.array([1, 2, 0]) / 5**0.5)
        self.assertFalse(result["summary"]["initial_hidden_state_used_for_fit"])
        self.assertFalse(result["summary"]["pressure_orthogonalization"])
        with self.assertRaises(ValueError):
            self.fit(np.zeros((18, 3, 2, 3)))

    def test_all_90_pairs_form_one_axis_instead_of_averaging_category_axes(self):
        ids = [f"sample_{i:03d}" for i in range(90)]
        categories = [f"category_{i // 30}" for i in range(90)]
        contrasts = np.zeros((90, 3), dtype=np.float32)
        contrasts[:30, 0] = 1
        contrasts[:30, 1] = np.tile([10, -10], 15)
        contrasts[30:60, 1] = 1
        contrasts[60:, 2] = 1
        states = np.zeros((90, 2, 1, 3), dtype=np.float32)
        states[:, 0, 0] = contrasts
        result = fit_states(states, ids, categories, [10], seed=314159)
        self.assertEqual(set(result["categories"]), {"pooled"})
        self.assertEqual(result["categories"]["pooled"]["pair_ids"], ids)
        expected = np.array([1 / 101**0.5, 1, 1])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result["categories"]["pooled"]["selective"], expected, rtol=1e-6)
        self.assertFalse(np.allclose(expected, np.ones(3) / 3**0.5))
        for fold in result["folds"].values():
            self.assertEqual(set(fold["categories"]), {"pooled"})
            self.assertEqual(len(fold["training_pair_ids"]), 60)
            self.assertEqual(len(fold["held_out_pair_ids"]), 30)
            training = [ids.index(pid) for pid in fold["training_pair_ids"]]
            np.testing.assert_allclose(fold["categories"]["pooled"], fit_direction(contrasts[training]))
        self.assertEqual(set(result["summary"]["counts_by_category"]), set(categories))

    def test_per_item_normalization_prevents_scale_dominance(self):
        np.testing.assert_allclose(fit_direction(np.array([[100, 0], [0, 1]])), [2**-0.5, 2**-0.5])

    def test_shared_state_changes_cancel(self):
        rng = np.random.default_rng(42)
        shift = rng.integers(-100, 100, size=(18, 1, 2, 3)).astype(np.float32)
        original, changed = self.fit(), self.fit(self.states + shift)
        self.assertEqual(original["summary"], changed["summary"])
        for category in ("pooled",):
            np.testing.assert_array_equal(original["categories"][category]["selective"], changed["categories"][category]["selective"])

    def test_layer_selection_uses_new_paired_states(self):
        self.assertEqual(self.fit()["summary"]["selected_layer"], 10)
        values = self.states.copy()
        for i in range(len(values)):
            values[i, 0, 0] = values[i, 1, 0] + [1, 1 if i % 2 else -1, 0]
        self.assertEqual(self.fit(values)["summary"]["selected_layer"], 16)

    def test_held_out_item_cannot_change_its_training_direction(self):
        states = self.states[:, :, :1].copy()
        original = self.fit(states, [10])
        fold = make_folds(self.ids, self.categories, 314159)[0]
        states[0, 0, 0] = states[0, 1, 0] + [7, -4, 3]
        changed = self.fit(states, [10])
        np.testing.assert_array_equal(
            original["folds"][fold]["categories"]["pooled"],
            changed["folds"][fold]["categories"]["pooled"],
        )
        held = []
        for value in changed["folds"].values():
            self.assertFalse(set(value["held_out_pair_ids"]) & set(value["training_pair_ids"]))
            held.extend(value["held_out_pair_ids"])
        self.assertCountEqual(held, self.ids)

    def test_invalid_and_degenerate_states_fail(self):
        for contrasts in (np.zeros((2, 3)), np.array([[1, np.nan, 0]]), np.array([[1, 0], [-1, 0]])):
            with self.assertRaises(ValueError):
                fit_direction(contrasts)
        for states in (np.zeros_like(self.states), np.full_like(self.states, np.inf)):
            with self.assertRaises(ValueError):
                self.fit(states)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(Path(__file__).resolve().parents[1] / "config.json")
        cls.item = load_pairs(cls.config)[0][0]

    def test_only_new_two_stage_pressure_caches_are_accepted(self):
        legacy = np.zeros((1, 3, 2, 1, 3), dtype=np.float32)
        legacy[:, 0] = np.nan
        legacy[:, 1] = [1, 2, 3]
        legacy[:, 2] = [4, 5, 6]
        payload = {
            "pair_id": self.item.pair_id, "category": self.item.category,
            "stage_names": ["initial", *STAGES], "layer_numbers": [10, 16],
            "condition_names": ["original"], "position_names": ["generation_query"], "states": legacy,
        }
        with self.assertRaises(RuntimeError):
            pressure_states(payload, self.item, [10, 16])
        payload["stage_names"] = list(STAGES)
        payload["states"] = legacy[:, 1:].copy()
        extracted = pressure_states(payload, self.item, [10, 16])
        np.testing.assert_array_equal(extracted[0], [[1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_equal(extracted[1], [[4, 5, 6], [4, 5, 6]])
        changed = copy.deepcopy(payload)
        changed["states"][:, 1] = np.nan
        with self.assertRaises(ValueError):
            pressure_states(changed, self.item, [10, 16])

    def test_collection_and_intervention_use_the_same_history_and_axis(self):
        class FakeEngine:
            def __init__(self):
                self.messages, self.calls = [], []

            def process_messages(self, messages):
                self.messages.append(messages)
                return SimpleNamespace(input_ids=np.zeros((1, 12)), stage=("initial", "Dplus", "Dminus")[len(self.messages) - 1])

            def prepare(self, processed):
                return processed

            def score_actual_options(self, processed, candidates, **kwargs):
                self.calls.append(kwargs)
                role = "Dminus" if processed.stage == "initial" else processed.stage
                scores = {key: float(key == role) for key in candidates}
                return {
                    "answer_key": role, "answer_text": candidates[role], "option_texts": candidates,
                    "option_token_ids": {key: [i] for i, key in enumerate(candidates)},
                    "choice_log_scores": scores, "choice_probabilities": {key: .25 for key in candidates},
                    "branch_count": 4,
                    "states": {layer: np.array([layer, float(processed.stage == "Dplus"), 0]) for layer in kwargs.get("capture_layers", ())},
                    "intervention_metrics": {"effective_ratio": abs(kwargs.get("intervention_ratio", 0))},
                }

        engine = FakeEngine()
        baseline, payload = collect_item(self.item, engine, [10, 16], None)
        self.assertEqual([tuple(call["capture_layers"]) for call in engine.calls], [(), (10, 16), (10, 16)])
        self.assertEqual(payload["stage_names"], list(STAGES))
        self.assertEqual(payload["states"].shape, (1, 2, 2, 1, 3))
        self.assertFalse(baseline["behavior"]["initial_correct"])
        for messages in engine.messages[1:]:
            self.assertEqual(messages[-2]["content"], baseline["initial_answer_used_in_both_branches"])
        direction = np.array([0, 1, 0], dtype=np.float32)
        engine = FakeEngine()
        calibrated = calibrate_item(self.item, baseline, 0, engine, 16, direction, .1, None)
        self.assertEqual(len(engine.calls), 6)
        self.assertTrue(all(call["intervention_direction"] is direction for call in engine.calls))
        self.assertEqual(calibrated["stages"]["Dplus"]["baseline"], baseline["stages"]["Dplus"])
        for messages in engine.messages[1:]:
            self.assertEqual(messages[-2]["content"], baseline["initial_answer_used_in_both_branches"])

        roles, _ = role_options(self.item)

        class LetterEngine(FakeEngine):
            def score_letter_options(self, processed, prepared, **kwargs):
                self.calls.append(kwargs)
                answer = roles["Dminus"] if processed.stage == "initial" else roles[processed.stage]
                return {
                    "answer": answer, "option_logits": {key: float(key == answer) for key in "ABCD"},
                    "intervention_metrics": {"effective_ratio": abs(kwargs["intervention_ratio"])} if kwargs else None,
                }

        engine = LetterEngine()
        signed = signed_ratio_conditions([.1, .2, .3])
        evaluated = run_item(self.item, engine, 16, direction, signed)
        validate_complete_row(evaluated, self.item, 16, signed)
        self.assertEqual(len(engine.calls), 21)
        interventions = [call for call in engine.calls if call]
        self.assertEqual(len(interventions), 18)
        self.assertTrue(all(call["intervention_direction"] is direction for call in interventions))
        for messages in engine.messages[1:]:
            self.assertEqual(messages[-2]["content"], f"{roles['Dminus']})")

    def test_default_entrypoint_and_output(self):
        args = parse_args([])
        self.assertEqual(args.stage, "all")
        self.assertEqual(args.output, ROOT / "results/steering/deepseek")
        self.assertFalse(hasattr(args, "fresh"))
        self.assertEqual(args.config, Path(__file__).resolve().parents[1] / "config.json")
        v1_args, v2_args = parse_v1_args([]), parse_v2_args([])
        self.assertEqual(v1_args.stage, "all")
        self.assertEqual(v1_args.output, args.output / "v1")
        self.assertEqual(v2_args.output, args.output / "v2")
        self.assertEqual(v2_args.v1_output, v1_args.output)
        self.assertEqual(v1_args.config, args.config)
        self.assertEqual(v2_args.config, args.config)

    def test_project_imports_without_other_src_projects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "src" / "steering" / "deepseek"
            shutil.copytree(Path(__file__).resolve().parents[1], project, ignore=shutil.ignore_patterns("__pycache__"))
            for entrypoint in ("run.py", "v1/run.py", "v2/run.py"):
                with self.subTest(entrypoint=entrypoint):
                    command = [sys.executable, "-B", str(project / entrypoint), "--help"]
                    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertNotIn("--fresh", result.stdout)
                    self.assertNotIn("--source-v1", result.stdout)
            script = (
                "import sys; sys.path.insert(0, 'src'); "
                "import steering.deepseek.v1.calibrate, steering.deepseek.v2.run, "
                "steering.deepseek.v1.selection, steering.deepseek.trie; "
                "assert not any(name.startswith(('steering_deepseek', 'visual_token_ablation', "
                "'comparison_experiments')) for name in sys.modules)"
            )
            result = subprocess.run([sys.executable, "-B", "-c", script], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
