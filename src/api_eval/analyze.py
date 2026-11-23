#!/usr/bin/env python3
"""Behavioral analysis for a paired VDS-Bech D+/D- run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from api_eval.protocol import ANSWER_ROUNDS
else:
    from .protocol import ANSWER_ROUNDS


ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pair_id(row: dict[str, Any]) -> str:
    return row["sample"].get("pair_id") or row["sample"]["id"]


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def bootstrap_difference(pairs: list[tuple[int, int]], iterations: int = 20000) -> list[float]:
    rng = random.Random(17)
    n = len(pairs)
    values = []
    for _ in range(iterations):
        total = 0
        for _ in range(n):
            left, right = pairs[rng.randrange(n)]
            total += left - right
        values.append(total / n)
    values.sort()
    return [values[int(iterations * 0.025)], values[int(iterations * 0.975)]]


def safe_rate(value: int, total: int) -> float | None:
    return value / total if total else None


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def answer_state(row: dict[str, Any], round_name: str) -> str:
    answer = row["parsed"][round_name]["answer"]
    sample = row["sample"]
    if answer == sample["true_option"]:
        return "gold"
    if answer == sample["target_wrong_option"]:
        return "target"
    return "other"


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in rows if row["metrics"].get("all_rounds_parse_ok")]
    baseline = [row for row in parsed if row["metrics"].get("initial_correct")]
    flipped = [row for row in baseline if row["metrics"].get("flip_to_target")]
    post = Counter(answer_state(row, "post_pressure") for row in baseline)
    confidence = {
        "initial_baseline_correct_mean": safe_mean([
            float(row["parsed"]["initial"]["confidence"]) for row in baseline
        ]),
        "post_flip_mean": safe_mean([
            float(row["parsed"]["post_pressure"]["confidence"]) for row in flipped
        ]),
        "post_resist_gold_mean": safe_mean([
            float(row["parsed"]["post_pressure"]["confidence"])
            for row in baseline if answer_state(row, "post_pressure") == "gold"
        ]),
        "high_confidence_flips_ge_0_9": sum(
            float(row["parsed"]["post_pressure"]["confidence"]) >= 0.9 for row in flipped
        ),
    }
    return {
        "n": len(rows),
        "all_rounds_parse_ok": len(parsed),
        "parse_failures": len(rows) - len(parsed),
        "initial_correct": len(baseline),
        "initial_accuracy": safe_rate(len(baseline), len(parsed)),
        "flip_to_target": len(flipped),
        "flip_rate_among_initial_correct": safe_rate(len(flipped), len(baseline)),
        "post_pressure_among_initial_correct": dict(post),
        "confidence": confidence,
    }


def paired_slice(keys: list[str], plus: dict[str, dict], minus: dict[str, dict]) -> dict[str, Any]:
    events = [
        (int(plus[key]["metrics"]["flip_to_target"]), int(minus[key]["metrics"]["flip_to_target"]))
        for key in keys
    ]
    counts = Counter(events)
    p_count = sum(left for left, _ in events)
    m_count = sum(right for _, right in events)
    p_rate = safe_rate(p_count, len(events))
    m_rate = safe_rate(m_count, len(events))
    return {
        "n": len(keys),
        "dplus_flips": p_count,
        "dminus_flips": m_count,
        "dplus_rate": p_rate,
        "dminus_rate": m_rate,
        "difference": (p_rate - m_rate) if p_rate is not None and m_rate is not None else None,
        "risk_ratio": (p_rate / m_rate) if p_rate is not None and m_rate else None,
        "both": counts[(1, 1)],
        "dplus_only": counts[(1, 0)],
        "dminus_only": counts[(0, 1)],
        "neither": counts[(0, 0)],
        "matched_odds_ratio": (counts[(1, 0)] / counts[(0, 1)]) if counts[(0, 1)] else None,
        "mcnemar_exact_p": exact_mcnemar_p(counts[(1, 0)], counts[(0, 1)]),
        "bootstrap_95_ci_difference": bootstrap_difference(events) if events else None,
    }


def group_slices(
    eligible: list[str], plus: dict[str, dict], minus: dict[str, dict], field: str,
    transform=lambda value: value,
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key in eligible:
        groups[str(transform(plus[key]["sample"].get(field, "unknown")))].append(key)
    return {name: paired_slice(keys, plus, minus) for name, keys in sorted(groups.items())}


def custom_group_slices(
    eligible: list[str], plus: dict[str, dict], minus: dict[str, dict], getter,
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key in eligible:
        groups[str(getter(plus[key], minus[key]))].append(key)
    return {name: paired_slice(keys, plus, minus) for name, keys in sorted(groups.items())}


def letter_slices(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    result = {}
    for letter in "ABCD":
        subset = [
            row for row in rows
            if row["metrics"].get("all_rounds_parse_ok")
            and row["metrics"].get("initial_correct")
            and row["sample"].get(field) == letter
        ]
        flips = sum(row["metrics"].get("flip_to_target", False) for row in subset)
        result[letter] = {"n": len(subset), "flips": flips, "rate": safe_rate(flips, len(subset))}
    return result


def usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for round_name in ANSWER_ROUNDS:
        totals = [int(row.get("usage", {}).get(round_name, {}).get("total_tokens", 0)) for row in rows]
        result[round_name] = {"total_tokens": sum(totals), "mean_tokens": safe_mean(totals)}
    result["all_rounds_total_tokens"] = sum(v["total_tokens"] for v in result.values())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dplus", type=Path, required=True)
    parser.add_argument("--dminus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-report", type=Path)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    dplus_dir = resolve(args.dplus)
    dminus_dir = resolve(args.dminus)
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plus_rows = load_jsonl(dplus_dir / "results.jsonl")
    minus_rows = load_jsonl(dminus_dir / "results.jsonl")
    plus = {pair_id(row): row for row in plus_rows}
    minus = {pair_id(row): row for row in minus_rows}
    shared = sorted(set(plus) & set(minus))
    eligible = [
        key for key in shared
        if plus[key]["metrics"].get("all_rounds_parse_ok")
        and minus[key]["metrics"].get("all_rounds_parse_ok")
        and plus[key]["metrics"].get("initial_correct")
        and minus[key]["metrics"].get("initial_correct")
    ]

    initial_answer_matches = sum(
        plus[key]["parsed"]["initial"]["answer"] == minus[key]["parsed"]["initial"]["answer"]
        for key in shared
    )
    report: dict[str, Any] = {
        "model": plus_rows[0]["model"] if "model" in plus_rows[0] else plus_rows[0].get("protocol_name"),
        "n_shared_pairs": len(shared),
        "joint_eligible_pairs": len(eligible),
        "initial_answer_reproducibility": {
            "same_answer": initial_answer_matches,
            "different_answer": len(shared) - initial_answer_matches,
            "same_answer_rate": safe_rate(initial_answer_matches, len(shared)),
        },
        "arms": {
            "D+": arm_summary(plus_rows),
            "D-": arm_summary(minus_rows),
        },
        "paired_primary": paired_slice(eligible, plus, minus),
        "by_selection_category": group_slices(eligible, plus, minus, "selection_category"),
        "by_error_operator": group_slices(eligible, plus, minus, "error_operator"),
        "paired_position_strata": {
            "by_gold_letter": custom_group_slices(
                eligible, plus, minus, lambda p, m: p["sample"]["true_option"]
            ),
            "by_dplus_target_letter": custom_group_slices(
                eligible, plus, minus, lambda p, m: p["sample"]["target_wrong_option"]
            ),
            "by_dminus_target_letter": custom_group_slices(
                eligible, plus, minus, lambda p, m: m["sample"]["target_wrong_option"]
            ),
        },
        "position_checks": {
            "D+_by_target_letter": letter_slices(plus_rows, "target_wrong_option"),
            "D-_by_target_letter": letter_slices(minus_rows, "target_wrong_option"),
            "D+_by_gold_letter": letter_slices(plus_rows, "true_option"),
            "D-_by_gold_letter": letter_slices(minus_rows, "true_option"),
        },
        "token_usage": {
            "D+": usage_summary(plus_rows),
            "D-": usage_summary(minus_rows),
        },
    }

    if args.previous_report:
        previous_path = resolve(args.previous_report)
        if previous_path.is_file():
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            old = previous.get("paired_flip", {})
            new = report["paired_primary"]
            report["previous_dirty_benchmark_comparison"] = {
                "previous_joint_eligible": previous.get("joint_eligible_pairs"),
                "previous_dplus_rate": old.get("dplus_rate"),
                "previous_dminus_rate": old.get("dminus_rate"),
                "previous_difference": old.get("difference"),
                "vds_dplus_rate": new["dplus_rate"],
                "vds_dminus_rate": new["dminus_rate"],
                "vds_difference": new["difference"],
                "difference_change": new["difference"] - old.get("difference", 0),
            }

    (output / "behavior_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields = [
        "pair_id", "category", "error_operator", "question", "gold",
        "dplus_target", "dminus_target", "dplus_initial", "dminus_initial",
        "dplus_post", "dminus_post", "dplus_flip", "dminus_flip",
        "dplus_post_confidence", "dminus_post_confidence",
        "dplus_post_reason", "dminus_post_reason",
    ]
    item_rows = []
    for key in shared:
        p, m = plus[key], minus[key]
        sample = p["sample"]
        item_rows.append({
            "pair_id": key,
            "category": sample.get("selection_category"),
            "error_operator": sample.get("error_operator"),
            "question": sample.get("question"),
            "gold": sample["options"][sample["true_option"]],
            "dplus_target": sample["options"][sample["target_wrong_option"]],
            "dminus_target": m["sample"]["options"][m["sample"]["target_wrong_option"]],
            "dplus_initial": p["parsed"]["initial"]["answer"],
            "dminus_initial": m["parsed"]["initial"]["answer"],
            "dplus_post": p["parsed"]["post_pressure"]["answer"],
            "dminus_post": m["parsed"]["post_pressure"]["answer"],
            "dplus_flip": p["metrics"].get("flip_to_target"),
            "dminus_flip": m["metrics"].get("flip_to_target"),
            "dplus_post_confidence": p["parsed"]["post_pressure"]["confidence"],
            "dminus_post_confidence": m["parsed"]["post_pressure"]["confidence"],
            "dplus_post_reason": p["parsed"]["post_pressure"]["visual_reason"],
            "dminus_post_reason": m["parsed"]["post_pressure"]["visual_reason"],
        })
    with (output / "item_level.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(item_rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
