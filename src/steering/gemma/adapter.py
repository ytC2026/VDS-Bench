"""Gemma 4 image/text scoring and query-state relative-norm steering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .io import resolve_path
from .trie import EncodedOption, score_options


LETTERS = ("A", "B", "C", "D")


def _hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unexpected decoder layer output: {type(output).__name__}")


def _replace(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"Unexpected decoder layer output: {type(output).__name__}")


@dataclass(frozen=True)
class FullVisualCondition:
    image_features: torch.Tensor


class Gemma431BEngine:
    def __init__(self, model: Any, processor: Any):
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.option_token_ids: dict[str, int] = {}
        for letter in LETTERS:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Gemma 4 does not encode {letter!r} as one token: {ids}")
            self.option_token_ids[letter] = int(ids[0])
        if len(set(self.option_token_ids.values())) != len(LETTERS):
            raise RuntimeError(f"A/B/C/D token IDs are not distinct: {self.option_token_ids}")

    @classmethod
    def load(
        cls, model_path: str | Path, *, dtype: str = "bfloat16",
        device_map: str = "auto", attn_implementation: str = "sdpa",
    ) -> "Gemma431BEngine":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Missing local Gemma 4 checkpoint: {path}")
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            path, dtype=getattr(torch, dtype), device_map=device_map,
            attn_implementation=attn_implementation, low_cpu_mem_usage=True,
            local_files_only=True,
        )
        if str(model.config.model_type) != "gemma4":
            raise RuntimeError(f"Expected gemma4, got {model.config.model_type}")
        engine = cls(model, processor)
        if engine.decoder_depth != 60:
            raise RuntimeError(
                f"Expected the 60-layer Gemma 4 31B checkpoint, got {engine.decoder_depth}"
            )
        return engine

    @property
    def decoder_layers(self) -> Any:
        return self.model.model.language_model.layers

    @property
    def decoder_depth(self) -> int:
        return len(self.decoder_layers)

    def metadata(self) -> dict[str, Any]:
        config = self.model.config
        return {
            "model_type": str(config.model_type),
            "architectures": list(config.architectures or []),
            "decoder_depth": self.decoder_depth,
            "hidden_size": int(config.text_config.hidden_size),
            "option_token_ids": dict(self.option_token_ids),
            "thinking_enabled": False,
            "final_logit_softcapping": config.text_config.final_logit_softcapping,
            "device_map": getattr(self.model, "hf_device_map", None),
        }

    def _move_inputs(self, values: Mapping[str, Any]) -> dict[str, Any]:
        embedding = self.model.model.language_model.embed_tokens.weight
        model_dtype = next(self.model.parameters()).dtype
        result: dict[str, Any] = {}
        for key, value in values.items():
            if not torch.is_tensor(value):
                result[key] = value
            elif torch.is_floating_point(value):
                result[key] = value.to(device=embedding.device, dtype=model_dtype)
            else:
                result[key] = value.to(device=embedding.device)
        return result

    def process_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        processed = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        result = self._move_inputs(processed)
        result["generation_query_index"] = int(result["input_ids"].shape[1]) - 1
        return result

    @torch.inference_mode()
    def full_visual_condition(self, processed: dict[str, Any]) -> FullVisualCondition:
        features = self.model.get_image_features(
            pixel_values=processed["pixel_values"],
            image_position_ids=processed["image_position_ids"],
            return_dict=True,
        ).pooler_output
        if not isinstance(features, (tuple, list)) or len(features) != 1:
            raise RuntimeError("Paired Gemma steering requires exactly one image")
        return FullVisualCondition(features[0].detach())

    def prepare(self, processed: dict[str, Any]) -> FullVisualCondition:
        """Prepare the complete image once for text prefixes or letter conditions."""
        return self.full_visual_condition(processed)

    def _prepare_language_inputs(
        self, processed: dict[str, Any], condition: FullVisualCondition,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_ids = processed["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Paired Gemma steering expects one example per forward")
        image_mask = input_ids == self.model.config.image_token_id
        pad_id = self.model.config.text_config.pad_token_id
        llm_ids = torch.where(image_mask, pad_id, input_ids)
        language_model = self.model.model.language_model
        embeddings = language_model.embed_tokens(llm_ids)
        per_layer_inputs = None
        if language_model.hidden_size_per_layer_input:
            per_layer_inputs = language_model.get_per_layer_inputs(llm_ids, None)
        image_features = condition.image_features.to(
            device=embeddings.device, dtype=embeddings.dtype,
        )
        n_image_tokens = int(image_mask.sum().item())
        if (
            image_features.ndim != 2
            or tuple(image_features.shape) != (n_image_tokens, embeddings.shape[-1])
            or n_image_tokens == 0
        ):
            raise RuntimeError(
                f"Image tokens/features differ: tokens={n_image_tokens}, "
                f"features={tuple(image_features.shape)}, hidden={embeddings.shape[-1]}"
            )
        expanded_mask = image_mask.unsqueeze(-1).expand_as(embeddings)
        return embeddings.masked_scatter(expanded_mask, image_features), per_layer_inputs

    @staticmethod
    def _with_prefix(processed: dict[str, Any], prefix: tuple[int, ...]) -> dict[str, Any]:
        if not prefix:
            return processed
        result = dict(processed)
        ids = processed["input_ids"]
        suffix = torch.tensor(prefix, dtype=ids.dtype, device=ids.device).unsqueeze(0)
        result["input_ids"] = torch.cat((ids, suffix), dim=1)
        mask = processed["attention_mask"]
        result["attention_mask"] = torch.cat((
            mask, torch.ones_like(suffix, dtype=mask.dtype, device=mask.device)
        ), dim=1)
        if "position_ids" in processed:
            positions = processed["position_ids"]
            start = int(positions[0, -1].item()) + 1
            extra = torch.arange(
                start, start + len(prefix), dtype=positions.dtype,
                device=positions.device,
            ).unsqueeze(0)
            result["position_ids"] = torch.cat((positions, extra), dim=1)
        if "mm_token_type_ids" in processed:
            types = processed["mm_token_type_ids"]
            text_types = torch.zeros(
                (types.shape[0], len(prefix)), dtype=types.dtype, device=types.device,
            )
            result["mm_token_type_ids"] = torch.cat((types, text_types), dim=1)
        result["generation_query_index"] = int(result["input_ids"].shape[1]) - 1
        return result

    @torch.inference_mode()
    def _forward_once(
        self, processed: dict[str, Any], condition: FullVisualCondition, *,
        capture_layers: Iterable[int] = (), intervention_layer: int | None = None,
        intervention_direction: torch.Tensor | None = None,
        intervention_ratio: float | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[str, float] | None]:
        wanted = sorted(set(int(layer) for layer in capture_layers))
        if wanted and (wanted[0] < 1 or wanted[-1] > self.decoder_depth):
            raise ValueError(f"Capture layers outside 1..{self.decoder_depth}: {wanted}")
        values = (intervention_layer, intervention_direction, intervention_ratio)
        if any(value is None for value in values) and not all(value is None for value in values):
            raise ValueError("Intervention layer, direction and ratio must be provided together")
        if intervention_layer is not None and not 1 <= intervention_layer <= self.decoder_depth:
            raise ValueError(f"Intervention layer outside 1..{self.decoder_depth}")
        if intervention_ratio is not None and not 0 < abs(float(intervention_ratio)) < 1:
            raise ValueError("Intervention ratio must be signed and strictly between -1 and 1")

        states: dict[int, torch.Tensor] = {}
        handles = []
        intervention_metrics: dict[str, float] | None = None

        def capture(layer: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                states[layer] = _hidden(output)[0, -1].detach().float().cpu().clone()
            return hook

        def intervene(_module: Any, _inputs: Any, output: Any) -> Any:
            nonlocal intervention_metrics
            hidden = _hidden(output)
            query = hidden[0, -1]
            vector = intervention_direction.to(device=hidden.device, dtype=torch.float32)
            if tuple(vector.shape) != tuple(hidden.shape[-1:]):
                raise ValueError("Intervention direction does not match hidden size")
            vector_norm = torch.linalg.vector_norm(vector)
            hidden_norm = torch.linalg.vector_norm(query.float())
            if not bool(torch.isfinite(vector_norm).item()) or float(vector_norm.item()) <= 0:
                raise ValueError("Intervention direction has invalid norm")
            if not bool(torch.isfinite(hidden_norm).item()) or float(hidden_norm.item()) <= 0:
                raise ValueError("Intervention hidden state has invalid norm")
            delta = (
                float(intervention_ratio) * hidden_norm * vector / vector_norm
            ).to(query.dtype)
            patched = hidden.clone()
            patched[0, -1] = query + delta
            effective_delta_norm = float(torch.linalg.vector_norm(
                patched[0, -1].float() - query.float()
            ).item())
            reference_norm = float(hidden_norm.item())
            intervention_metrics = {
                "requested_signed_ratio": float(intervention_ratio),
                "pre_hidden_norm": reference_norm,
                "requested_delta_norm": abs(float(intervention_ratio)) * reference_norm,
                "effective_delta_norm": effective_delta_norm,
                "effective_ratio": effective_delta_norm / reference_norm,
            }
            return _replace(output, patched)

        embeddings, per_layer_inputs = self._prepare_language_inputs(processed, condition)
        try:
            for layer in wanted:
                handles.append(self.decoder_layers[layer - 1].register_forward_hook(capture(layer)))
            if intervention_layer is not None:
                handles.append(self.decoder_layers[intervention_layer - 1].register_forward_hook(intervene))
            outputs = self.model.model(
                inputs_embeds=embeddings, per_layer_inputs=per_layer_inputs,
                attention_mask=processed["attention_mask"],
                mm_token_type_ids=processed["mm_token_type_ids"],
                position_ids=processed.get("position_ids"),
                use_cache=False, return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        if set(states) != set(wanted):
            raise RuntimeError(f"Captured {sorted(states)}; expected {wanted}")
        hidden = outputs.last_hidden_state[:, -1, :]
        parameter = self.model.lm_head.weight
        logits = self.model.lm_head(
            hidden.to(device=parameter.device, dtype=parameter.dtype)
        )[0]
        cap = self.model.config.text_config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        if intervention_layer is not None and intervention_metrics is None:
            raise RuntimeError("Intervention hook did not run")
        return logits.float(), states, intervention_metrics

    @torch.inference_mode()
    def score_actual_options(
        self, processed: dict[str, Any], option_texts: Mapping[str, str], *,
        prepared: FullVisualCondition | None = None, capture_layers: Iterable[int] = (),
        terminal_token_id: int | None = None,
        intervention_layer: int | None = None,
        intervention_direction: torch.Tensor | None = None,
        intervention_ratio: float | None = None,
    ) -> dict[str, Any]:
        condition = self.prepare(processed) if prepared is None else prepared
        encoded = []
        for key, raw_text in option_texts.items():
            answer = str(raw_text).strip()
            ids = tuple(int(value) for value in self.tokenizer.encode(
                answer, add_special_tokens=False,
            ))
            if not ids:
                raise RuntimeError(f"Option {key!r} has no tokens")
            encoded.append(EncodedOption(str(key), answer, ids))
        terminal = terminal_token_id
        if terminal is None:
            terminal = self.tokenizer.eos_token_id
        if isinstance(terminal, (tuple, list)):
            terminal = terminal[0] if terminal else None
        if terminal is None:
            raise RuntimeError("Tokenizer has no EOS token; set terminal_token_id")
        terminal = int(terminal)
        root_logits, states, root_metrics = self._forward_once(
            processed, condition, capture_layers=capture_layers,
            intervention_layer=intervention_layer,
            intervention_direction=intervention_direction,
            intervention_ratio=intervention_ratio,
        )
        cache: dict[tuple[int, ...], torch.Tensor] = {(): root_logits}
        prefix_metrics: dict[tuple[int, ...], dict[str, float]] = {}

        def provider(prefix: tuple[int, ...], children: tuple[int, ...]) -> dict[int, float]:
            if prefix not in cache:
                prefixed = self._with_prefix(processed, prefix)
                cache[prefix], _unused, metric = self._forward_once(
                    prefixed, condition,
                    intervention_layer=intervention_layer,
                    intervention_direction=intervention_direction,
                    intervention_ratio=intervention_ratio,
                )
                if metric is not None:
                    prefix_metrics[prefix] = metric
            return {token: float(cache[prefix][token].item()) for token in children}

        scored = score_options(encoded, terminal, provider)
        by_key = {option.key: option.text for option in encoded}
        return {
            **scored,
            "answer_text": by_key[scored["answer_key"]],
            "option_texts": by_key,
            "option_token_ids": {option.key: list(option.token_ids) for option in encoded},
            "terminal_token_id": terminal,
            "scoring_method": "locally_normalized_valid_continuation_trie",
            "states": states,
            "intervention_metrics": {
                "root": root_metrics,
                "prefixes": [
                    {"prefix_token_ids": list(prefix), **metric}
                    for prefix, metric in sorted(prefix_metrics.items())
                ],
            },
        }

    @torch.inference_mode()
    def score_letter_options(
        self, processed: dict[str, Any], condition: FullVisualCondition, *,
        intervention_layer: int | None = None,
        intervention_direction: torch.Tensor | None = None,
        intervention_ratio: float | None = None,
    ) -> dict[str, Any]:
        logits, _states, metrics = self._forward_once(
            processed, condition,
            intervention_layer=intervention_layer,
            intervention_direction=intervention_direction,
            intervention_ratio=intervention_ratio,
        )
        option_logits = {
            letter: float(logits[token].item())
            for letter, token in self.option_token_ids.items()
        }
        return {
            "answer": max(LETTERS, key=option_logits.get),
            "option_logits": option_logits,
            "option_token_ids": dict(self.option_token_ids),
            "intervention_metrics": metrics,
        }


def load_engine(config: dict[str, Any]) -> Gemma431BEngine:
    return Gemma431BEngine.load(
        resolve_path(config["model_path"]), dtype=str(config["dtype"]),
        device_map=str(config["device_map"]),
        attn_implementation=str(config.get("attn_implementation", "sdpa")),
    )
