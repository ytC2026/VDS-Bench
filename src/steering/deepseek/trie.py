"""Dependency-free constrained scoring for multi-token option strings."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class EncodedOption:
    key: str
    text: str
    token_ids: tuple[int, ...]


@dataclass
class _Node:
    children: dict[int, "_Node"] = field(default_factory=dict)


LogitProvider = Callable[[tuple[int, ...], tuple[int, ...]], Mapping[int, float]]


def _log_softmax(values: Mapping[int, float]) -> dict[int, float]:
    if not values:
        raise ValueError("Cannot normalize an empty branch")
    maximum = max(float(value) for value in values.values())
    normalizer = maximum + math.log(
        sum(math.exp(float(value) - maximum) for value in values.values())
    )
    return {token: float(value) - normalizer for token, value in values.items()}


def validate_options(options: Sequence[EncodedOption], terminal_token_id: int) -> None:
    if len(options) < 2:
        raise ValueError("At least two options are required")
    keys = [option.key for option in options]
    texts = [option.text for option in options]
    paths = [(*option.token_ids, int(terminal_token_id)) for option in options]
    if len(set(keys)) != len(keys) or len(set(texts)) != len(texts):
        raise ValueError("Option keys and texts must be unique")
    if any(not option.token_ids for option in options):
        raise ValueError("Every option must contain at least one token")
    if len(set(paths)) != len(paths):
        raise ValueError("Encoded option paths must be unique")


def build_trie(options: Sequence[EncodedOption], terminal_token_id: int) -> _Node:
    validate_options(options, terminal_token_id)
    root = _Node()
    for option in options:
        node = root
        for token in (*option.token_ids, int(terminal_token_id)):
            node = node.children.setdefault(int(token), _Node())
    return root


def branch_prefixes(options: Sequence[EncodedOption], terminal_token_id: int) -> dict[tuple[int, ...], tuple[int, ...]]:
    root = build_trie(options, terminal_token_id)
    result: dict[tuple[int, ...], tuple[int, ...]] = {}

    def walk(node: _Node, prefix: tuple[int, ...]) -> None:
        children = tuple(sorted(node.children))
        if len(children) > 1:
            result[prefix] = children
        for token, child in node.children.items():
            walk(child, (*prefix, token))

    walk(root, ())
    return result


def score_options(
    options: Sequence[EncodedOption],
    terminal_token_id: int,
    provider: LogitProvider,
) -> dict[str, object]:
    """Score complete choices by locally normalizing only at trie branches."""
    branches = branch_prefixes(options, terminal_token_id)
    transition_scores: dict[tuple[tuple[int, ...], int], float] = {}
    raw_branches: list[dict[str, object]] = []
    for prefix, children in branches.items():
        supplied = {
            int(token): float(value)
            for token, value in provider(prefix, children).items()
        }
        if set(supplied) != set(children):
            raise ValueError(
                f"Provider returned tokens {sorted(supplied)}, expected {sorted(children)}"
            )
        raw_branches.append({
            "prefix": list(prefix),
            "children": {str(token): supplied[token] for token in children},
        })
        normalized = _log_softmax(supplied)
        for token, value in normalized.items():
            transition_scores[(prefix, token)] = value

    scores: dict[str, float] = {}
    for option in options:
        prefix: tuple[int, ...] = ()
        total = 0.0
        for token in (*option.token_ids, int(terminal_token_id)):
            if prefix in branches:
                total += transition_scores[(prefix, int(token))]
            prefix = (*prefix, int(token))
        scores[option.key] = total

    maximum = max(scores.values())
    weights = {key: math.exp(value - maximum) for key, value in scores.items()}
    total_weight = sum(weights.values())
    probabilities = {key: value / total_weight for key, value in weights.items()}
    # Python's max keeps the first occurrence on an exact tie, so the model's
    # score is never tie-broken by an internal option identifier.
    answer = max(options, key=lambda option: scores[option.key]).key
    return {
        "answer_key": answer,
        "choice_log_scores": scores,
        "choice_probabilities": probabilities,
        "branch_count": len(branches),
        "branch_logits": raw_branches,
    }
