"""Check dev-only sign selection, including initially wrong items and ties."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from steering.gemma.v1.selection import select_sign


def row(pair_id, initially_correct, positive, negative):
    stages = {}
    for stage in ("initial", "Dplus", "Dminus"):
        stages[stage] = {}
        for condition, margin in (("baseline", 0.0), ("positive", positive), ("negative", negative)):
            stages[stage][condition] = {
                "answer_role": "GOLD" if initially_correct else "other",
                "Dplus_minus_GOLD": -margin,
                "Dminus_minus_GOLD": -margin,
            }
    return {"pair_id": pair_id, "fold": 0, "stages": stages}


class SelectionTests(unittest.TestCase):
    def test_sign_margin_includes_initially_wrong_dev_items_and_both_arms(self):
        result = select_sign([row("correct", True, -1, 1), row("wrong", False, 3, -1)], ratio=0.1)
        self.assertEqual(result["selected_sign"], 1)
        self.assertEqual(result["selection_metrics"]["baseline"]["initially_correct_denominator"], 1)
        self.assertEqual(result["selection_metrics"]["baseline"]["margin_denominator"], 4)
        self.assertEqual(result["mean_margin_change_by_condition"], {"negative": 0.0, "positive": 1.0})

    def test_no_improvement_or_positive_tie_selects_zero(self):
        for positive, negative in ((-1, -2), (0, 0), (1, 1)):
            with self.subTest(positive=positive, negative=negative):
                result = select_sign([row("item", True, positive, negative)], ratio=0.1)
                self.assertEqual(result["selected_sign"], 0)


if __name__ == "__main__":
    unittest.main()
