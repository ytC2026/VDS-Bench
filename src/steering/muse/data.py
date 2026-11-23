"""Load and validate fixed D+/D- splits without behavior-based filtering."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl, resolve_path


LETTERS = ("A", "B", "C", "D")
EXPECTED_CATEGORIES = {
    "referent_or_attribute_rebinding",
    "spatial_scope_shift",
    "object_or_part_switch",
}


@dataclass(frozen=True)
class PressurePair:
    pair_id: str
    dplus: dict[str, Any]
    dminus: dict[str, Any]
    image_path: Path

    @property
    def category(self) -> str:
        return str(self.dplus["selection_category"])

    @property
    def options(self) -> dict[str, str]:
        return {letter: str(self.dplus["options"][letter]) for letter in LETTERS}


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            raise ValueError(f"Missing pair_id in {label}")
        if pair_id in result:
            raise ValueError(f"Duplicate pair_id {pair_id!r} in {label}")
        result[pair_id] = row
    return result


def _validate_pair(pair_id: str, plus: dict[str, Any], minus: dict[str, Any]) -> None:
    for field in (
        "image_id", "image_path", "question", "options", "true_option",
        "selection_category", "error_operator", "true_evidence",
    ):
        if plus.get(field) != minus.get(field):
            raise ValueError(f"D+/D- mismatch for {pair_id}: {field}")
    options = {str(key).upper(): str(value).strip() for key, value in plus["options"].items()}
    if set(options) != set(LETTERS) or any(not value for value in options.values()):
        raise ValueError(f"Expected four nonempty A/B/C/D options for {pair_id}")
    if len(set(options.values())) != 4:
        raise ValueError(f"Option texts must be unique for {pair_id}: {options}")
    roles = {
        str(plus["true_option"]).upper(),
        str(plus["target_wrong_option"]).upper(),
        str(minus["target_wrong_option"]).upper(),
    }
    if len(roles) != 3 or not roles <= set(LETTERS):
        raise ValueError(f"GOLD, D+, and D- must be three distinct options for {pair_id}")
    if str(plus["selection_category"]) not in EXPECTED_CATEGORIES:
        raise ValueError(f"Unknown category for {pair_id}: {plus['selection_category']}")


def load_pairs(config: dict[str, Any]) -> tuple[list[PressurePair], dict[str, Any]]:
    plus = _index(read_jsonl(config["dplus_path"]), "D+")
    minus = _index(read_jsonl(config["dminus_path"]), "D-")
    if set(plus) != set(minus):
        raise ValueError("D+ and D- benchmark pair sets differ")
    split = read_json(config["split_path"])
    pair_ids = [str(value) for value in split.get("pair_ids", [])]
    if not pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Split must contain unique pair IDs")
    image_root = resolve_path(config["images_dir"])
    items: list[PressurePair] = []
    for pair_id in pair_ids:
        if pair_id not in plus:
            raise ValueError(f"Split pair is absent from benchmark: {pair_id}")
        p, m = plus[pair_id], minus[pair_id]
        _validate_pair(pair_id, p, m)
        image_path = image_root / Path(str(p["image_path"])).name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        items.append(PressurePair(pair_id, p, m, image_path))
    counts = dict(sorted(Counter(item.category for item in items).items()))
    return items, {
        "selected": len(items),
        "counts_by_category": counts,
        "split_role": split.get("role"),
    }
