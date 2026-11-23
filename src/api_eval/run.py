#!/usr/bin/env python3
"""Launch the paired benchmark through the provider selected in .env."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .client import chat_completions_url, load_env_file
from .providers import (
    PROVIDER_CHOICES,
    ProviderConfig,
    resolve_provider_config,
    validate_provider_config,
)


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "benchmark"
OUTPUTS = ROOT / "results" / "api_eval"
CATEGORY_STARTS = (1, 151, 301)


def selected_ids(unsupported: bool, per_category: int) -> list[str]:
    if not 1 <= per_category <= 150:
        raise ValueError("--per-category must be between 1 and 150")
    prefix = "vds_dminus_" if unsupported else "vds_dplus_"
    return [
        f"{prefix}{index:04d}"
        for start in CATEGORY_STARTS
        for index in range(start, start + per_category)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VDS-Bech using VDS_PROVIDER and the matching .env profile."
    )
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Override VDS_PROVIDER")
    parser.add_argument("--model", help="Override the selected provider's MODEL")
    parser.add_argument("--base-url", help="Override the selected provider's BASE_URL")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"))
    parser.add_argument("--run-prefix")
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Run pair IDs from a JSON list/object or a text file (one ID per line).",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--sequential", action="store_true", help="Run D+ then D- instead of in parallel")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ids_from_file(path: Path) -> tuple[list[str], list[str]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        values = payload.get("pair_ids", payload) if isinstance(payload, dict) else payload
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(values, list):
        raise ValueError("--ids-file must contain a JSON list/object or one ID per line")
    numeric: list[str] = []
    for value in values:
        match = re.search(r"(\d{4})$", str(value))
        if not match or not 1 <= int(match.group(1)) <= 450:
            raise ValueError(f"Invalid benchmark pair ID in --ids-file: {value!r}")
        numeric.append(match.group(1))
    numeric = list(dict.fromkeys(numeric))
    if not numeric:
        raise ValueError("--ids-file contains no pair IDs")
    return (
        [f"vds_dplus_{number}" for number in numeric],
        [f"vds_dminus_{number}" for number in numeric],
    )


def arm_command(
    *,
    arm: str,
    input_path: Path,
    output: Path,
    ids: list[str] | None,
    args: argparse.Namespace,
    config: ProviderConfig,
) -> list[str]:
    command = [
        sys.executable,
        "-m", "api_eval.run_arm",
        "--input", str(input_path),
        "--arm", arm,
        "--output", str(output),
        "--provider", config.provider,
        "--model", config.model,
        "--base-url", config.base_url,
        "--env-file", str(args.env_file),
        "--resume",
    ]
    if config.temperature is not None:
        command.extend(["--temperature", str(config.temperature)])
    effort = config.extra_body.get("reasoning_effort")
    if effort:
        command.extend(["--reasoning-effort", str(effort)])
    for sample_id in ids or []:
        command.extend(["--sample-id", sample_id])
    return command


def _run_sequential(jobs: list[tuple[str, list[str]]]) -> int:
    failed = False
    for label, command in jobs:
        print(f"Starting {label} ...", flush=True)
        code = subprocess.run(command, cwd=ROOT, check=False).returncode
        print(f"{label} finished with exit code {code}.", flush=True)
        failed = failed or code != 0
    return 1 if failed else 0


def _run_parallel(jobs: list[tuple[str, list[str]]]) -> int:
    processes: list[tuple[str, subprocess.Popen[bytes]]] = []
    try:
        for label, command in jobs:
            print(f"Starting {label} ...", flush=True)
            processes.append((label, subprocess.Popen(command, cwd=ROOT)))
        failed = False
        for label, process in processes:
            code = process.wait()
            print(f"{label} finished with exit code {code}.", flush=True)
            failed = failed or code != 0
        return 1 if failed else 0
    except KeyboardInterrupt:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        return 130


def main() -> int:
    args = parse_args()
    if args.full and args.ids_file:
        raise ValueError("--full and --ids-file are mutually exclusive")
    load_env_file(args.env_file)
    config = resolve_provider_config(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )
    validate_provider_config(config, require_api_key=not args.dry_run)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.model)

    if args.ids_file:
        dplus_ids, dminus_ids = ids_from_file(args.ids_file)
        size_tag = f"subset{len(dplus_ids)}"
    else:
        dplus_ids = None if args.full else selected_ids(False, args.per_category)
        dminus_ids = None if args.full else selected_ids(True, args.per_category)
        size_tag = "full450" if args.full else f"calib{args.per_category * 3}"
    prefix = args.run_prefix or f"{safe_model}_{size_tag}"
    jobs = [
        (
            "D+",
            arm_command(
                arm="D+",
                input_path=BENCHMARK / "vds_dplus.jsonl",
                output=OUTPUTS / prefix / "dplus",
                ids=dplus_ids,
                args=args,
                config=config,
            ),
        ),
        (
            "D-",
            arm_command(
                arm="D-",
                input_path=BENCHMARK / "vds_dminus.jsonl",
                output=OUTPUTS / prefix / "dminus",
                ids=dminus_ids,
                args=args,
                config=config,
            ),
        ),
    ]
    if args.dry_run:
        print(f"provider={config.provider}")
        print(f"model={config.model}")
        print(f"endpoint={chat_completions_url(config.base_url)}")
        print(f"outputs={OUTPUTS}")
        for label, command in jobs:
            print(label, subprocess.list2cmdline(command))
        return 0
    return _run_sequential(jobs) if args.sequential else _run_parallel(jobs)


if __name__ == "__main__":
    raise SystemExit(main())
