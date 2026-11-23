"""Summarize original/edited API runs on fixed matched initial-correct cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "visual_decoy_removal"

from .api_eval.client import load_jsonl, write_json

NAMES = ("original_dplus", "original_dminus", "edited_dplus", "edited_dminus")


def rate(count, total):
    return count / total if total else None


def analyze(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    requested = set(manifest["pair_ids"])
    by_condition, summary = {}, {}
    for name in manifest["conditions"]:
        path = run / name / "results.jsonl"
        rows = load_jsonl(path) if path.exists() else []
        if len({r["sample"]["pair_id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate result pair IDs: {name}")
        rows = [r for r in rows if r["sample"]["pair_id"] in requested]
        valid = [r for r in rows if r["metrics"].get("all_rounds_parse_ok")]
        correct = [r for r in valid if r["metrics"]["initial_correct"]]
        by_condition[name] = {r["sample"]["pair_id"]: r for r in correct}
        flips = sum(r["metrics"]["flip_to_target"] for r in correct)
        errors_path = run / name / "errors.jsonl"
        errors = load_jsonl(errors_path) if errors_path.exists() else []
        summary[name] = {"requested": len(requested), "completed": len(rows), "complete": len(rows) == len(requested) and not errors, "errors": len(errors), "valid_parsed": len(valid), "parse_failures": len(rows) - len(valid), "initial_correct": len(correct), "initial_accuracy_among_valid": rate(len(correct), len(valid)), "flips": flips, "sycophancy_rate": rate(flips, len(correct))}
    paired = {}
    for image_type in ("original", "edited"):
        a, b = image_type + "_dplus", image_type + "_dminus"
        if a in by_condition and b in by_condition:
            common = sorted(set(by_condition[a]) & set(by_condition[b]))
            plus = sum(by_condition[a][pid]["metrics"]["flip_to_target"] for pid in common)
            minus = sum(by_condition[b][pid]["metrics"]["flip_to_target"] for pid in common)
            paired[image_type] = {"n_initial_correct_in_both_arms": len(common), "dplus_rate": rate(plus, len(common)), "dminus_rate": rate(minus, len(common)), "gap": rate(plus - minus, len(common))}
    matched = None
    if all(name in by_condition for name in NAMES):
        common = sorted(set.intersection(*(set(by_condition[n]) for n in NAMES)))
        values = {name: rate(sum(by_condition[name][pid]["metrics"]["flip_to_target"] for pid in common), len(common)) for name in NAMES}
        matched = {"n_initial_correct_in_all_four": len(common), "pair_ids": common, "rates": values, "original_gap": None, "edited_gap": None, "gap_reduction": None}
        if common:
            matched["original_gap"] = values["original_dplus"] - values["original_dminus"]
            matched["edited_gap"] = values["edited_dplus"] - values["edited_dminus"]
            matched["gap_reduction"] = matched["original_gap"] - matched["edited_gap"]
    result = {"conditions": summary, "within_image_type_matched": paired, "all_four_matched": matched, "rate_unit": "fraction", "notes": ["Rates exclude malformed responses and use initially correct answers.", "The all-four gap comparison uses exactly the same initial-correct pair IDs in all conditions.", "Separate-arm initial answers follow the project's existing two-round API protocol; no initial assistant answer is fabricated or shared.", "No significance or causal conclusion is inferred automatically from these summaries."]}
    write_json(run / "analysis.json", result)
    def pct(value):
        return "—" if value is None else f"{100 * value:.2f}%"
    lines = ["# Original/edited image API results", "", "| Condition | Completed/requested | Valid parsed | Initial accuracy | Sycophancy |", "|---|---:|---:|---:|---:|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['completed']}/{s['requested']} | {s['valid_parsed']} | {pct(s['initial_accuracy_among_valid'])} | {pct(s['sycophancy_rate'])} |")
    if matched is not None:
        lines += ["", f"Initially correct in all four conditions: {matched['n_initial_correct_in_all_four']}.", "", f"Original D+/D− gap: {pct(matched['original_gap'])}; edited gap: {pct(matched['edited_gap'])}; reduction: {pct(matched['gap_reduction'])} (percentage-point differences)."]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {run / 'report.md'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    analyze(parser.parse_args().run)
