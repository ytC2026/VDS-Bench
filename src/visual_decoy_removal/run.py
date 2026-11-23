"""Evaluate the same 100 questions on original/edited images through an API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "visual_decoy_removal"

from .api_eval.client import load_env_file, load_jsonl, write_json
from .api_eval.providers import PROVIDER_CHOICES, resolve_provider_config, validate_provider_config

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
DATA = ROOT / "data" / "visual_decoy_removal"
RESULTS = ROOT / "results" / "visual_decoy_removal"
CONDITIONS = ("original_dplus", "original_dminus", "edited_dplus", "edited_dminus")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_image(relative, data_dir=DATA):
    # Retain support for legacy data/images/... without changing stored data.
    relative = Path(relative)
    if relative.parts[:2] == ("data", "images"):
        relative = Path(*relative.parts[1:])
    path = (data_dir / relative).resolve()
    if not path.is_relative_to((data_dir / "images").resolve()) or not path.is_file():
        raise ValueError(f"Missing or invalid image path: {relative}")
    return path


def validate_data(data_dir=DATA):
    compact = (data_dir / 'dplus.jsonl').is_file() and (data_dir / 'dminus.jsonl').is_file()
    if compact:
        arms = {arm: load_jsonl(data_dir / f'{arm}.jsonl') for arm in ('dplus', 'dminus')}
        ids = [r['pair_id'] for r in arms['dplus']]
    else:
        split = json.loads((data_dir / "splits/split100.json").read_text(encoding="utf-8"))
        ids = split["pair_ids"]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Expected exactly 100 unique pair IDs")
    datasets = {}
    for name in CONDITIONS:
        if compact:
            variant, arm = name.split('_')
            rows = [dict(r, image_path=r[f'{variant}_image_path']) for r in arms[arm]]
        else:
            rows = load_jsonl(data_dir / "benchmark" / f"{name}.jsonl")
        if [r["pair_id"] for r in rows] != ids or len({r["id"] for r in rows}) != 100:
            raise ValueError(f"Sample order or IDs differ: {name}")
        for row in rows:
            checked_image(row["image_path"], data_dir)
            if row["true_option"] == row["target_wrong_option"]:
                raise ValueError(f"Incorrect target equals gold: {row['pair_id']}")
        datasets[name] = rows
    for items in zip(*(datasets[name] for name in CONDITIONS)):
        for field in ("pair_id", "image_id", "question", "options", "true_option", "selection_category"):
            if any(row[field] != items[0][field] for row in items):
                raise ValueError(f"Mismatch in {field}: {items[0]['pair_id']}")
        a, b, c, d = items
        if a["image_path"] != b["image_path"] or c["image_path"] != d["image_path"] or a["image_path"] == c["image_path"]:
            raise ValueError("Original/edited image pairing is invalid")
        if a["target_wrong_option"] != c["target_wrong_option"] or b["target_wrong_option"] != d["target_wrong_option"] or a["target_wrong_option"] == b["target_wrong_option"]:
            raise ValueError("Feedback target pairing is invalid")
    return datasets


def default_env():
    local = ROOT / "configs" / "visual_decoy_removal" / ".env"
    return local if local.is_file() else ROOT / ".env"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate local data only; no API calls")
    parser.add_argument("--dry-run", action="store_true", help="Print the request plan; no API calls or result writes")
    parser.add_argument("--images", choices=("both", "original", "edited"), default="both")
    parser.add_argument("--data-dir", type=Path, default=DATA, help="Dataset directory, absolute or relative to the project root")
    parser.add_argument("--provider", choices=PROVIDER_CHOICES)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--env-file", type=Path, default=default_env())
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"))
    parser.add_argument("--limit", type=int, help="First N matching pairs for a smoke run")
    parser.add_argument("--run-name", help="Separate result directory name; rerun the same command to resume")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent questions per provider (four conditions still run one at a time)")
    args = parser.parse_args(argv)
    data_dir = (ROOT / args.data_dir).resolve()
    datasets = validate_data(data_dir)
    compact = (data_dir / 'dplus.jsonl').is_file() and (data_dir / 'dminus.jsonl').is_file()
    image_root = data_dir
    if args.limit is not None and not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    if args.workers < 1 or args.sleep < 0:
        parser.error("--workers must be positive and --sleep must be nonnegative")
    if args.check:
        print(json.dumps({"pairs": 100, "conditions": list(CONDITIONS), "unique_original_images": len({r['image_path'] for r in datasets['original_dplus']}), "unique_edited_images": len({r['image_path'] for r in datasets['edited_dplus']}), "paired_fields_unchanged": True, "api_calls": 0}, indent=2))
        return 0
    args.env_file = args.env_file.resolve()
    load_env_file(args.env_file)
    config = resolve_provider_config(provider=args.provider, model=args.model, base_url=args.base_url, temperature=args.temperature, reasoning_effort=args.reasoning_effort)
    validate_provider_config(config, require_api_key=not args.dry_run)
    names = [n for n in CONDITIONS if args.images == "both" or n.startswith(args.images + "_")]
    count = args.limit or 100
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.model)
    run_name = args.run_name or f"{model_slug}_{'full100' if count == 100 else f'smoke{count}'}_{args.images}"
    if args.run_name is None and data_dir != DATA.resolve():
        run_name += "_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", data_dir.name)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        parser.error("--run-name must use letters, numbers, underscores, dots or hyphens")
    output = RESULTS / run_name
    hashes = {}
    for name in names:
        path = data_dir / f"{name.split('_')[1]}.jsonl" if compact else data_dir / "benchmark" / f"{name}.jsonl"
        hashes[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = digest(path)
        for row in datasets[name][:count]:
            hashes[row["image_path"]] = digest(checked_image(row["image_path"], data_dir))
    manifest = {"schema_version": "edited100_api_run_v1", "provider": config.protocol_fields(), "extra_body": config.extra_body, "conditions": names, "pair_ids": [r["pair_id"] for r in datasets[names[0]][:count]], "max_tokens": args.max_tokens, "data_sha256": hashes, "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(SOURCE.rglob('*.py'))}, "initial_answer_policy": "independent_two_round_dialogue_per_condition_matching_existing_api_protocol"}
    commands = []
    for name in names:
        input_path = data_dir / f"{name.split('_')[1]}.jsonl" if compact else data_dir / "benchmark" / f"{name}.jsonl"
        command = [sys.executable, "-B", "-m", "visual_decoy_removal.api_eval.run_arm", "--input", str(input_path), "--arm", "D+" if name.endswith("dplus") else "D-", "--output", str(output / name), "--provider", config.provider, "--model", config.model, "--base-url", config.base_url, "--env-file", str(args.env_file), "--limit", str(count), "--timeout", str(args.timeout), "--max-retries", str(args.max_retries), "--retry-sleep", str(args.retry_sleep), "--sleep", str(args.sleep), "--resume", "--image-root", str(image_root)]
        if compact:
            command += ['--image-variant', name.split('_')[0]]
        if config.temperature is not None:
            command += ["--temperature", str(config.temperature)]
        if config.extra_body.get("reasoning_effort"):
            command += ["--reasoning-effort", str(config.extra_body["reasoning_effort"])]
        if args.max_tokens is not None:
            command += ["--max-tokens", str(args.max_tokens)]
        command += ["--workers", str(args.workers)]
        commands.append(command)
    plan = {"provider": config.provider, "model": config.model, "conditions": names, "pairs_per_condition": count, "workers": args.workers, "max_in_flight_requests_per_provider": min(args.workers, count), "condition_execution": "sequential", "initial_api_requests_before_retries": count * len(names) * 2, "output": str(output), "api_key_printed": False}
    plan.update(data_dir=str(data_dir), image_root=str(image_root))
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Data/model/settings/source changed; choose a different --run-name")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty output directory has no compatible manifest; choose a different --run-name")
    else:
        write_json(manifest_path, manifest)
    # Scheduling can change on resume without changing generation settings.
    with (output / "execution_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"started_at_utc": datetime.now(timezone.utc).isoformat(), "workers": args.workers, "conditions": names, "condition_execution": "sequential"}) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    failed = False
    for name, command in zip(names, commands):
        print(f"Starting {name}", flush=True)
        code = subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
        failed = failed or code != 0
        if code in {3, 130}:
            # Stop all remaining conditions on invalid credentials or interruption.
            break
    from .analyze import analyze
    analyze(output)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
