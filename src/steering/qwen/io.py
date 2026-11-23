"""Small shared file helpers."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[3]

def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding='utf-8-sig'))

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in resolve_path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(''.join((json.dumps(value, ensure_ascii=False) + '\n' for value in values)), encoding='utf-8')
