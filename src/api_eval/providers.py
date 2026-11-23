#!/usr/bin/env python3
"""Provider profiles for the unified VDS-Bech API entry point.

The benchmark protocol is provider-neutral.  This module contains the small
set of request-shape differences required by the supported OpenAI-compatible
endpoints.  Switching providers is therefore an environment/configuration
operation rather than a source-code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


PROVIDER_ALIASES = {
    "qwen": "qwen",
    "dashscope": "qwen",
    "aliyun": "qwen",
    "kimi": "kimi",
    "moonshot": "kimi",
    "mimo": "mimo",
    "xiaomi": "mimo",
    "xiaomi_mimo": "mimo",
    "minimax": "minimax",
    "minmax": "minimax",
    "generic": "generic",
    "openai": "generic",
    "openai_compatible": "generic",
}
PROVIDER_CHOICES = ("qwen", "kimi", "mimo", "minimax", "generic")


@dataclass(frozen=True)
class ProviderDefaults:
    base_url: str
    model: str
    temperature: Optional[float]
    effective_temperature: Optional[float]
    response_format_json: bool = True
    preserve_assistant_message: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


DEFAULTS: dict[str, ProviderDefaults] = {
    "qwen": ProviderDefaults(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.8-max",
        temperature=0.0,
        effective_temperature=0.0,
        extra_body={"enable_thinking": False},
        notes=("Qwen non-thinking mode via enable_thinking=false",),
    ),
    "kimi": ProviderDefaults(
        base_url="https://api.moonshot.cn/v1",
        model="kimi-k3",
        temperature=None,
        effective_temperature=1.0,
        preserve_assistant_message=True,
        extra_body={"reasoning_effort": "low"},
        notes=(
            "temperature omitted because Kimi-K3 fixes it provider-side",
            "complete assistant message is preserved between protocol rounds",
        ),
    ),
    "mimo": ProviderDefaults(
        base_url="https://api.xiaomimimo.com/v1",
        model="mimo-v2.5",
        temperature=0.0,
        effective_temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
        notes=("MiMo thinking disabled for protocol comparability",),
    ),
    "minimax": ProviderDefaults(
        base_url="https://api.minimax.io/v1",
        model="MiniMax-M3",
        temperature=1.0,
        effective_temperature=1.0,
        preserve_assistant_message=True,
        extra_body={"thinking": {"type": "disabled"}},
        notes=(
            "MiniMax temperature 1.0 avoids the provider's rejection of 0.0",
            "complete assistant message is preserved between protocol rounds",
        ),
    ),
    "generic": ProviderDefaults(
        base_url="https://api.openai.com/v1",
        model="replace-with-multimodal-model-id",
        temperature=0.0,
        effective_temperature=0.0,
        notes=("plain OpenAI-compatible request without vendor-only fields",),
    ),
}


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    base_url: str
    model: str
    api_key: str
    temperature: Optional[float]
    effective_temperature: Optional[float]
    response_format_json: bool
    preserve_assistant_message: bool
    extra_body: dict[str, Any]
    notes: tuple[str, ...]

    def protocol_fields(self) -> dict[str, Any]:
        """Return a secret-free record of the resolved request contract."""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "temperature_sent": self.temperature,
            "temperature_effective": self.effective_temperature,
            "response_format_json": self.response_format_json,
            "preserve_assistant_message": self.preserve_assistant_message,
            "provider_request_fields": sorted(self.extra_body),
            "provider_notes": list(self.notes),
        }


def canonical_provider(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    try:
        return PROVIDER_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(PROVIDER_CHOICES)
        raise ValueError(f"Unsupported provider {value!r}; choose one of: {choices}") from exc


def infer_provider(model: str, base_url: str = "") -> str:
    value = model.strip().lower().split("/")[-1]
    host = base_url.lower()
    if "qwen" in value or "dashscope" in host or "aliyuncs.com" in host:
        return "qwen"
    if "kimi" in value or "moonshot" in host:
        return "kimi"
    if value.startswith("mimo-") or "xiaomimimo" in host:
        return "mimo"
    if value.startswith("minimax-") or "minimax" in host or "minimaxi" in host:
        return "minimax"
    return "generic"


def _first(env: Mapping[str, str], *names: str) -> Optional[str]:
    for name in names:
        value = env.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _float(value: Optional[str], default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    if value.strip().lower() in {"omit", "none", "null"}:
        return None
    return float(value)


def _provider_env_name(provider: str, suffix: str) -> str:
    return f"VDS_{provider.upper()}_{suffix}"


def resolve_provider_config(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    api_key_env: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> ProviderConfig:
    """Resolve the active provider from CLI overrides and one unified .env.

    Provider-specific variables take precedence over the legacy VDS_TARGET_*
    variables.  CLI values take precedence over both.
    """
    values = os.environ if env is None else env
    legacy_model = _first(values, "VDS_TARGET_MODEL") or ""
    legacy_url = _first(values, "VDS_TARGET_BASE_URL") or ""
    selected = provider or _first(values, "VDS_PROVIDER")
    if selected:
        selected = canonical_provider(selected)
    else:
        selected = infer_provider(model or legacy_model, base_url or legacy_url)
    defaults = DEFAULTS[selected]
    prefix_model = _provider_env_name(selected, "MODEL")
    prefix_url = _provider_env_name(selected, "BASE_URL")
    prefix_key = _provider_env_name(selected, "API_KEY")
    prefix_temperature = _provider_env_name(selected, "TEMPERATURE")
    prefix_json = _provider_env_name(selected, "RESPONSE_FORMAT_JSON")
    prefix_preserve = _provider_env_name(selected, "PRESERVE_ASSISTANT_MESSAGE")

    resolved_model = model or _first(values, prefix_model, "VDS_TARGET_MODEL") or defaults.model
    resolved_url = base_url or _first(values, prefix_url, "VDS_TARGET_BASE_URL") or defaults.base_url
    key_name = api_key_env or _first(values, "VDS_API_KEY_ENV")
    resolved_key = _first(values, key_name) if key_name else None
    resolved_key = resolved_key or _first(values, prefix_key, "VDS_TARGET_API_KEY") or ""
    configured_temperature = _float(_first(values, prefix_temperature), defaults.temperature)
    if temperature is not None:
        configured_temperature = temperature

    extra = dict(defaults.extra_body)
    if selected == "kimi":
        # Kimi-K3 rejects a client-provided temperature.  Keep the CLI/env
        # override only in the protocol record; never send it on the wire.
        configured_temperature = None
        effort = reasoning_effort or _first(values, _provider_env_name(selected, "REASONING_EFFORT"))
        extra["reasoning_effort"] = effort or "low"
    elif reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort

    response_format_json = _bool(_first(values, prefix_json), defaults.response_format_json)
    preserve = _bool(_first(values, prefix_preserve), defaults.preserve_assistant_message)
    effective_temperature = defaults.effective_temperature
    if selected not in {"kimi", "minimax"}:
        effective_temperature = configured_temperature

    return ProviderConfig(
        provider=selected,
        base_url=resolved_url,
        model=resolved_model,
        api_key=resolved_key,
        temperature=configured_temperature,
        effective_temperature=effective_temperature,
        response_format_json=response_format_json,
        preserve_assistant_message=preserve,
        extra_body=extra,
        notes=defaults.notes,
    )


def validate_provider_config(config: ProviderConfig, *, require_api_key: bool = True) -> None:
    if not config.base_url.startswith(("http://", "https://")):
        raise ValueError(f"Provider {config.provider}: BASE_URL must use http:// or https://")
    if not config.model or config.model.startswith("replace-with"):
        raise ValueError(f"Provider {config.provider}: MODEL is not configured")
    inferred = infer_provider(config.model, config.base_url)
    if inferred != "generic" and inferred != config.provider:
        raise ValueError(
            f"Provider/model mismatch: VDS_PROVIDER={config.provider!r}, but model/URL look like {inferred!r}"
        )
    if require_api_key and not config.api_key:
        raise RuntimeError(f"Provider {config.provider}: API_KEY is not configured in .env")
