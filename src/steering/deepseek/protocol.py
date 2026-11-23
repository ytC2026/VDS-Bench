"""Question and option helpers shared by V1 and V2."""

from __future__ import annotations

import re
from typing import Any

from .data import LETTERS


def question_stem(question: str) -> str:
    value = str(question).strip()
    match = re.search(r"\s+\(A\)\s+", value)
    if match is None:
        return value
    stem = value[: match.start()].strip()
    if not stem:
        raise ValueError(f"Could not recover question stem from {question!r}")
    return stem


def ordered_option_texts(sample: dict[str, Any]) -> dict[str, str]:
    options = {str(key).upper(): str(value).strip() for key, value in sample["options"].items()}
    if set(options) != set(LETTERS):
        raise ValueError("Expected A/B/C/D option mapping")
    return {letter: options[letter] for letter in LETTERS}


def role_option_texts(
    dplus_sample: dict[str, Any], dminus_sample: dict[str, Any]
) -> dict[str, str]:
    options = ordered_option_texts(dplus_sample)
    gold = str(dplus_sample["true_option"]).upper()
    dplus = str(dplus_sample["target_wrong_option"]).upper()
    dminus = str(dminus_sample["target_wrong_option"]).upper()
    other = next(key for key in options if key not in {gold, dplus, dminus})
    return {
        "GOLD": options[gold],
        "Dplus": options[dplus],
        "Dminus": options[dminus],
        "other": options[other],
    }
