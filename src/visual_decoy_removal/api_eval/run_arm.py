#!/usr/bin/env python3
"""Run one arm of the VDS-Bech two-round protocol."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

from .client import (
    call_chat_completion,
    image_to_data_url,
    load_env_file,
    load_jsonl,
    make_text_message,
    make_user_message,
    write_json,
    write_jsonl,
)
from .protocol import (
    ANSWER_ROUNDS,
    PRESSURE_ID,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    SYSTEM_PROMPT,
    initial_prompt,
    parse_answer,
    pressure_prompt,
    row_metrics,
)
from .providers import (
    PROVIDER_CHOICES,
    ProviderConfig,
    resolve_provider_config,
    validate_provider_config,
)


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "visual_decoy_removal"

SAMPLE_FIELDS = (
    "id",
    "pair_id",
    "question",
    "options",
    "true_option",
    "target_wrong_option",
    "selection_category",
    "error_operator",
)


def compact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields used to interpret and analyze a result row."""
    return {key: sample[key] for key in SAMPLE_FIELDS if key in sample}


def assistant_history_message(
    response: dict[str, Any], raw: str, preserve_complete_message: bool
) -> dict[str, Any]:
    if not preserve_complete_message:
        return {"role": "assistant", "content": raw}
    choices = response.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices else None
    if not isinstance(message, dict):
        raise ValueError("Provider response is missing choices[0].message")
    # Kimi and MiniMax may attach provider-required history fields (for example
    # reasoning_content). Preserve the returned assistant message without
    # concatenating those fields into content.
    return dict(message)


def continue_after(
    messages: list[dict[str, Any]], assistant_message: dict[str, Any], prompt: str
) -> list[dict[str, Any]]:
    return [*messages, assistant_message, make_text_message(prompt)]


def api_call(
    messages: list[dict[str, Any]], args: argparse.Namespace, config: ProviderConfig
):
    return call_chat_completion(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        response_format_json=config.response_format_json,
        extra_body=config.extra_body,
    )


def run_sample(
    sample: dict[str, Any], args: argparse.Namespace, config: ProviderConfig
) -> dict[str, Any]:
    relative = Path(sample["image_path"])
    if relative.parts[:2] == ("data", "images"):
        relative = Path(*relative.parts[1:])
    image_path = getattr(args, "image_root", DATA) / relative
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image = image_to_data_url(image_path)
    prompts = {
        "initial": initial_prompt(sample),
        "post_pressure": pressure_prompt(sample),
    }
    raw: dict[str, str] = {}
    parsed: dict[str, dict[str, Any]] = {}
    usage: dict[str, dict[str, Any]] = {}
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        make_user_message(prompts["initial"], image),
    ]

    raw["initial"], response = api_call(initial_messages, args, config)
    usage["initial"] = response.get("usage", {})
    parsed["initial"] = parse_answer(raw["initial"])
    initial_assistant = assistant_history_message(
        response, raw["initial"], config.preserve_assistant_message
    )
    post_messages = continue_after(initial_messages, initial_assistant, prompts["post_pressure"])

    raw["post_pressure"], response = api_call(post_messages, args, config)
    usage["post_pressure"] = response.get("usage", {})
    parsed["post_pressure"] = parse_answer(raw["post_pressure"])
    return {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "pressure_id": PRESSURE_ID,
        "arm": args.arm,
        "provider": config.provider,
        "model": config.model,
        "sample": compact_sample(sample),
        "raw_outputs": raw,
        "parsed": parsed,
        "metrics": row_metrics(sample, parsed),
        "usage": usage,
    }


def export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id", "pair_id", "arm", "provider", "model", "category", "true_option",
        "target_option", "initial_answer", "post_answer",
        "initial_correct", "flip_to_target", "post_reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            sample, parsed, metrics = row["sample"], row["parsed"], row["metrics"]
            writer.writerow({
                "id": sample["id"],
                "pair_id": sample.get("pair_id") or sample.get("paired_source_id") or sample["id"],
                "arm": row["arm"],
                "provider": row.get("provider"),
                "model": row["model"],
                "category": sample.get("selection_category"),
                "true_option": sample["true_option"],
                "target_option": sample["target_wrong_option"],
                "initial_answer": metrics["initial_answer"],
                "post_answer": metrics["post_answer"],
                "initial_correct": metrics["initial_correct"],
                "flip_to_target": metrics["flip_to_target"],
                "post_reason": parsed["post_pressure"]["visual_reason"],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one VDS-Bech arm with the active .env provider.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=DATA, help="Base directory for benchmark image paths")
    parser.add_argument("--image-variant", choices=('original', 'edited'), help="Select original_image_path or edited_image_path from compact question files")
    parser.add_argument("--arm", choices=("D+", "D-"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=PROVIDER_CHOICES)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", help="Legacy override for the API-key environment variable name")
    from ..run import default_env
    parser.add_argument("--env-file", type=Path, default=default_env())
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"))
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent questions; rounds within each question stay sequential")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _ordered_results(rows: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [by_id[row["id"]] for row in rows if row["id"] in by_id]


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.sleep < 0:
        raise ValueError("--workers must be positive and --sleep must be nonnegative")
    load_env_file(args.env_file)
    config = resolve_provider_config(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        api_key_env=args.api_key_env,
    )
    validate_provider_config(config)

    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    rows = load_jsonl(input_path)
    if args.image_variant:
        rows = [dict(row, image_path=row[f'{args.image_variant}_image_path']) for row in rows]
    if args.sample_id:
        wanted = set(args.sample_id)
        rows = [row for row in rows if row["id"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No benchmark rows selected")
    args.output.mkdir(parents=True, exist_ok=True)

    protocol = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "pressure_id": PRESSURE_ID,
        "arm": args.arm,
        "input": input_path.name,
        **config.protocol_fields(),
        "rounds": list(ANSWER_ROUNDS),
        "self_attribution_probe": False,
        "derived_explanation_audit_included": False,
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
        if existing != protocol:
            raise RuntimeError("Incompatible existing run directory; choose a new run prefix/output.")
    else:
        write_json(protocol_path, protocol)

    results_path = args.output / "results.jsonl"
    result_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and results_path.exists():
        result_by_id = {row["sample"]["id"]: row for row in load_jsonl(results_path)}
    selected_ids = {row["id"] for row in rows}
    errors_path = args.output / "errors.jsonl"
    error_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and errors_path.exists():
        error_by_id = {
            row["id"]: row for row in load_jsonl(errors_path)
            if row.get("id") in selected_ids and row.get("id") not in result_by_id
        }

    completed = sum(row["id"] in result_by_id for row in rows)

    def work(sample):
        # A worker owns the entire two-round conversation, never shared history.
        try:
            return run_sample(sample, args, config)
        finally:
            if args.sleep:
                time.sleep(args.sleep)

    def save_result(sample, future):
        nonlocal completed
        sample_id = sample["id"]
        try:
            result = future.result()
        except Exception as exc:
            error_by_id[sample_id] = {"id": sample_id, "error": repr(exc)}
            write_jsonl(errors_path, error_by_id.values())
            print(f"ERROR {sample_id}: {exc}", file=sys.stderr, flush=True)
            # Credentials are common to every question; do not send 100 identical failures.
            return getattr(exc, "status", None) in {401, 403}
        else:
            result_by_id[sample_id] = result
            error_by_id.pop(sample_id, None)
            completed += 1
            # Only the main thread writes, using the existing atomic file writer.
            write_jsonl(results_path, _ordered_results(rows, result_by_id))
            write_jsonl(errors_path, error_by_id.values())
            print(
                f"[{completed}/{len(rows)}] {sample_id} initial={result['metrics']['initial_answer']} "
                f"post={result['metrics']['post_answer']}",
                flush=True,
            )
            return False

    pending_rows = iter(row for row in rows if row["id"] not in result_by_id)
    stopped_for_auth = False
    interrupted = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {}

        def submit_next():
            sample = next(pending_rows, None)
            if sample is not None:
                active[pool.submit(work, sample)] = sample

        for _ in range(args.workers):
            submit_next()
        while active:
            try:
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                interrupted = True
                print("Stopping: saving in-flight questions; no new questions will be submitted.", flush=True)
                continue
            # Drain this batch before refilling, so an auth failure stops submissions.
            for future in finished:
                stopped_for_auth = save_result(active.pop(future), future) or stopped_for_auth
            if not stopped_for_auth and not interrupted:
                for _ in finished:
                    submit_next()

    output_rows = _ordered_results(rows, result_by_id)
    write_jsonl(errors_path, error_by_id.values())
    export_csv(args.output / "results.csv", output_rows)
    summary = {
        "n": len(output_rows),
        "requested": len(rows),
        "initial_correct": sum(row["metrics"]["initial_correct"] for row in output_rows),
        "flips": sum(row["metrics"]["flip_to_target"] for row in output_rows),
        "errors": len(error_by_id),
        "complete": len(output_rows) == len(rows) and not error_by_id,
    }
    write_json(args.output / "summary.json", summary)
    if interrupted:
        return 130
    return 3 if stopped_for_auth else (1 if error_by_id else 0)


if __name__ == "__main__":
    raise SystemExit(main())
