"""Choose the paired steering sign from mean GOLD-minus-target text-score change."""

from __future__ import annotations

from statistics import mean
from typing import Any


def _candidate_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row["stages"]["initial"]["baseline"]["answer_role"] == "GOLD"
    ]
    if not eligible:
        raise ValueError("No initially correct development items")
    counts = {
        arm: sum(row["stages"][arm][condition]["answer_role"] == arm for row in eligible)
        for arm in ("Dplus", "Dminus")
    }
    gold_minus_target = [
        -row["stages"][arm][condition][f"{arm}_minus_GOLD"]
        for row in rows for arm in ("Dplus", "Dminus")
    ]
    initial_correct = sum(
        row["stages"]["initial"][condition]["answer_role"] == "GOLD"
        for row in rows
    )
    return {
        "initially_correct_denominator": len(eligible),
        "target_adoption_count_by_arm": counts,
        "combined_target_adoption_rate": sum(counts.values()) / (2 * len(eligible)),
        "margin_denominator": len(gold_minus_target),
        "mean_gold_minus_target_text_score": mean(gold_minus_target),
        "mean_gold_minus_target_text_score_by_arm": {
            arm: mean(
                -row["stages"][arm][condition][f"{arm}_minus_GOLD"]
                for row in rows
            )
            for arm in ("Dplus", "Dminus")
        },
        "initial_accuracy": initial_correct / len(rows),
        "initial_correct_count": initial_correct,
    }


def select_sign(
    rows: list[dict[str, Any]],
    *,
    ratio: float,
) -> dict[str, Any]:
    if not rows or len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("Calibration rows must have unique pair IDs")
    if not 0 < ratio < 1:
        raise ValueError("Calibration ratio must be in (0, 1)")
    for row in rows:
        if set(row["stages"]) != {"initial", "Dplus", "Dminus"}:
            raise ValueError(f"Incomplete calibration stages for {row['pair_id']}")
        for stage in row["stages"].values():
            if set(stage) != {"baseline", "negative", "positive"}:
                raise ValueError("Each stage requires baseline, negative, positive")
    stats = {
        condition: _candidate_summary(rows, condition)
        for condition in ("baseline", "negative", "positive")
    }
    baseline_margin = stats["baseline"]["mean_gold_minus_target_text_score"]
    margin_changes = {
        condition: stats[condition]["mean_gold_minus_target_text_score"] - baseline_margin
        for condition in ("negative", "positive")
    }
    arm_margin_changes = {
        condition: {
            arm: (
                stats[condition]["mean_gold_minus_target_text_score_by_arm"][arm]
                - stats["baseline"]["mean_gold_minus_target_text_score_by_arm"][arm]
            )
            for arm in ("Dplus", "Dminus")
        }
        for condition in ("negative", "positive")
    }
    best_change = max(margin_changes.values())
    best = [condition for condition, change in margin_changes.items() if change == best_change]
    if best_change <= 0:
        chosen = "baseline"
        reason = "neither sign increased the mean GOLD-minus-target text score"
    elif len(best) != 1:
        chosen = "baseline"
        reason = "positive and negative signs tie on mean GOLD-minus-target change"
    else:
        chosen = best[0]
        reason = "largest positive mean GOLD-minus-target change on dev90"
    return {
        "schema_version": "gemma_paired_margin_sign_selection_v1",
        "protocol": "unlabeled actual-option text",
        "ratio": ratio,
        "selection_population": "all dev90 pairs, both pressure arms equally weighted",
        "selection_statistic": "mean GOLD-minus-target option-text log-score change from baseline",
        "selected_condition": chosen,
        "selected_sign": {"negative": -1, "baseline": 0, "positive": 1}[chosen],
        "selection_reason": reason,
        "mean_margin_change_by_condition": margin_changes,
        "mean_margin_change_by_arm": arm_margin_changes,
        "selection_metrics": stats,
        "pair_ids": [row["pair_id"] for row in rows],
        "folds": sorted({row["fold"] for row in rows}),
    }
