"""Run paired Dplus-Dminus Qwen steering: collect, fit, calibrate, evaluate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "steering.qwen"

from .io import ROOT, read_json, resolve_path, write_json
from .v1.run import METHOD, run_collection
from .v1.extract import run_extraction, validate_fit

DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = ROOT / "results/steering/qwen"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("all", "collect", "fit", "calibrate", "evaluate"), default="all")
    parser.add_argument("--selection-only", action="store_true", help="For --stage calibrate: select sign from saved calibration records")
    args = parser.parse_args(argv)
    if args.selection_only and args.stage != "calibrate":
        parser.error("--selection-only requires --stage calibrate")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = read_json(resolve_path(args.config))
    output = resolve_path(args.output)
    print(f"[run] output={output}", flush=True)
    v1, v2 = output / "v1", output / "v2"
    result = None
    if args.stage in ("all", "collect"):
        result = run_collection(config, v1)
    if args.stage in ("all", "fit"):
        result = run_extraction(config, v1)
        print(f"[fit] selected_layer={result['selected_layer']} direction=Dplus-Dminus", flush=True)
    if args.stage in ("all", "calibrate"):
        from .v1.calibrate import run_calibration, select_from_saved_records

        validate_fit(config, v1)
        result = (select_from_saved_records if args.selection_only else run_calibration)(config, v1)
        print(f"[calibrate] selected_sign={result['selected_sign']}", flush=True)
    if args.stage in ("all", "evaluate"):
        from .v2.run import run_evaluation

        validate_fit(config, v1)
        result = run_evaluation(config, v1, v2)
        result["method"] = METHOD
        write_json(v2 / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
