#!/usr/bin/env python3
"""Stable OpenAI-compatible multimodal HTTP client for VDS-Bech."""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-max"
ANSWER_RE = re.compile(r"\b([ABCD]|[1-4])\b", re.IGNORECASE)
RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ChatCompletionError(RuntimeError):
    """A normalized API/transport/response failure."""

    def __init__(self, message: str, *, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def chat_completions_url(base_url: str) -> str:
    """Accept a host, an SDK-style /v1 URL, or the complete endpoint URL."""
    value = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid API base URL: {base_url!r}")
    path = parsed.path.rstrip("/")
    lowered = path.lower()
    if lowered.endswith("/chat/completions"):
        final_path = path
    elif lowered.endswith("/v1"):
        final_path = f"{path}/chat/completions"
    elif not path:
        final_path = "/v1/chat/completions"
    else:
        final_path = f"{path}/chat/completions"
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, final_path, parsed.query, parsed.fragment)
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_text(path, text)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid .env line at {path}:{line_no}")
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError(f"Empty .env key at {path}:{line_no}")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)


def image_to_data_url(path: Path) -> str:
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def make_user_message(text: str, image_data_url: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": text},
        ],
    }


def make_text_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def normalize_answer(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"A", "B", "C", "D", "1", "2", "3", "4"}:
        return text
    match = ANSWER_RE.search(text)
    return match.group(1).upper() if match else None


def build_chat_completion_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: Optional[float],
    max_tokens: Optional[int],
    response_format_json: bool,
    extra_body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}
    for key, value in (extra_body or {}).items():
        if key in {"model", "messages"}:
            raise ValueError(f"Provider extra_body cannot override {key!r}")
        if value is not None:
            payload[key] = value
    return payload


def _response_text(obj: dict[str, Any]) -> str:
    try:
        message = obj["choices"][0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatCompletionError("Response is missing choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                value = part.get("text")
                if isinstance(value, str):
                    parts.append(value)
        text = "".join(parts)
        if text:
            return text
    raise ChatCompletionError(f"Unsupported response content type: {type(content).__name__}")


def _http_error(exc: urllib.error.HTTPError) -> ChatCompletionError:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    body = body[:2000].strip()
    detail = f": {body}" if body else ""
    return ChatCompletionError(f"HTTP {exc.code}{detail}", status=exc.code)


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: Optional[float],
    timeout: int,
    max_tokens: Optional[int],
    max_retries: int,
    retry_sleep: float,
    response_format_json: bool = False,
    extra_body: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    payload = build_chat_completion_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format_json=response_format_json,
        extra_body=extra_body,
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    endpoint = chat_completions_url(base_url)
    last_error: Optional[BaseException] = None
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        retryable = True
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ChatCompletionError("API response is not a JSON object")
            return _response_text(obj), obj
        except urllib.error.HTTPError as exc:
            last_error = _http_error(exc)
            retryable = exc.code in RETRYABLE_HTTP_STATUS
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = ChatCompletionError(f"Transport error: {exc}")
        except json.JSONDecodeError as exc:
            last_error = ChatCompletionError(f"API returned invalid JSON: {exc}")
        except ChatCompletionError as exc:
            last_error = exc
        if not retryable or attempt >= max_retries:
            break
        time.sleep(retry_sleep * (attempt + 1))
    raise ChatCompletionError(
        f"API call failed after {attempts} attempt(s): {last_error}",
        status=getattr(last_error, "status", None),
    )
