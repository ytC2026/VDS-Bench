"""Record option-text scores without retaining model states."""

from __future__ import annotations

from typing import Any


def compact_score(scored: dict[str, Any]) -> tuple[dict[str, Any], dict[int, Any]]:
    scores = {str(key): float(value) for key, value in scored["choice_log_scores"].items()}
    return {
        "answer_role": str(scored["answer_key"]),
        "answer_text": str(scored["answer_text"]),
        "role_texts": dict(scored["option_texts"]),
        "role_token_ids": dict(scored["option_token_ids"]),
        "choice_log_scores": scores,
        "choice_probabilities": dict(scored["choice_probabilities"]),
        "Dplus_minus_GOLD": scores["Dplus"] - scores["GOLD"],
        "Dminus_minus_GOLD": scores["Dminus"] - scores["GOLD"],
        "Dplus_minus_Dminus": scores["Dplus"] - scores["Dminus"],
        "branch_count": int(scored["branch_count"]),
    }, dict(scored["states"])
