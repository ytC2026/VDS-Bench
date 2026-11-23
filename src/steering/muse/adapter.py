"""Muse Glimmer 30B image/text scoring and query-state relative-norm steering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .trie import EncodedOption, score_options
from .io import resolve_path


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


class MuseGlimmerEngine:
    # The checkpoint template ends at "<|start|>assistant". Complete the
    # recipient and message delimiter before scoring answer tokens.
    ANSWER_PREFIX = " to=user<|message|>"

    def __init__(self, model: Any, processor: Any, *, reasoning_strength: str = "low"):
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.reasoning_strength = reasoning_strength
        self.generation_header_ids = tuple(int(value) for value in self.tokenizer.encode(
            "<|start|>assistant", add_special_tokens=False,
        ))
        self.answer_header_ids = tuple(int(value) for value in self.tokenizer.encode(
            "<|start|>assistant" + self.ANSWER_PREFIX, add_special_tokens=False,
        ))
        if not self.generation_header_ids or not self.answer_header_ids:
            raise RuntimeError("Muse Glimmer answer header has no tokens")
        self.turn_end_token_id = int(self.tokenizer.convert_tokens_to_ids("<|eot|>"))
        if self.tokenizer.convert_ids_to_tokens(self.turn_end_token_id) != "<|eot|>":
            raise RuntimeError("Muse Glimmer tokenizer has no <|eot|> token")
        self.option_token_ids: dict[str, int] = {}
        for letter in LETTERS:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Muse Glimmer does not encode {letter!r} as one token: {ids}")
            self.option_token_ids[letter] = int(ids[0])
        if len(set(self.option_token_ids.values())) != len(LETTERS):
            raise RuntimeError(f"A/B/C/D token IDs are not distinct: {self.option_token_ids}")

    @classmethod
    def load(
        cls, model_path: str | Path, *, dtype: str = "bfloat16",
        device_map: str = "auto", attn_implementation: str = "sdpa",
        reasoning_strength: str = "low",
    ) -> "MuseGlimmerEngine":
        from transformers import AutoProcessor, MuseGlimmerForConditionalGeneration

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Missing local Muse Glimmer 30B checkpoint: {path}")
        processor = AutoProcessor.from_pretrained(
            path, local_files_only=True,
        )
        model = MuseGlimmerForConditionalGeneration.from_pretrained(
            path, dtype=getattr(torch, dtype), device_map=device_map,
            attn_implementation=attn_implementation, low_cpu_mem_usage=True,
            local_files_only=True,
        )
        if str(model.config.model_type) != "muse_glimmer":
            raise RuntimeError(f"Expected muse_glimmer, got {model.config.model_type}")
        if str(model.config.text_config.model_type) != "muse_glimmer_text":
            raise RuntimeError(f"Unexpected text decoder: {model.config.text_config.model_type}")
        if str(model.config.vision_config.model_type) != "muse_glimmer_vision":
            raise RuntimeError(f"Unexpected vision encoder: {model.config.vision_config.model_type}")
        engine = cls(model, processor, reasoning_strength=reasoning_strength)
        if engine.decoder_depth != 52:
            raise RuntimeError(
                f"Expected the 52-layer Muse Glimmer decoder, got {engine.decoder_depth}"
            )
        if int(model.config.text_config.hidden_size) != 6656:
            raise RuntimeError(
                "Expected the 6656-wide Muse Glimmer decoder, "
                f"got {model.config.text_config.hidden_size}"
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
            "language_model_type": str(config.text_config.model_type),
            "vision_model_type": str(config.vision_config.model_type),
            "image_token_id": int(config.image_token_id),
            "turn_end_token_id": self.turn_end_token_id,
            "reasoning_strength": self.reasoning_strength,
            "generation_header_token_ids": list(self.generation_header_ids),
            "answer_header_token_ids": list(self.answer_header_ids),
            "logit_output_multiplier": float(config.text_config.output_multiplier),
            "final_logit_softcapping": float(config.text_config.final_logit_softcapping),
            "device_map": getattr(self.model, "hf_device_map", None),
        }

    def _move_inputs(self, values: Mapping[str, Any]) -> dict[str, Any]:
        embedding = self.model.model.language_model.embed_tokens.weight
        result: dict[str, Any] = {}
        for key, value in values.items():
            if not torch.is_tensor(value):
                result[key] = value
            elif key in {"pixel_values", "image_grid_thw"}:
                result[key] = value
            elif torch.is_floating_point(value):
                result[key] = value.to(device=embedding.device, dtype=embedding.dtype)
            else:
                result[key] = value.to(device=embedding.device)
        return result

    def process_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        processed = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
            reasoning_strength=self.reasoning_strength,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        result = self._move_inputs(processed)
        if "pixel_values" not in result or "image_grid_thw" not in result:
            raise RuntimeError("Muse Glimmer processor did not return image pixels and grid")
        if result["input_ids"].shape[0] != 1:
            raise RuntimeError("Muse Glimmer steering requires batch size one")
        prompt_end = self.tokenizer.decode(
            result["input_ids"][0, -16:].tolist(), skip_special_tokens=False,
        )
        if not prompt_end.endswith("<|start|>assistant"):
            raise RuntimeError(f"Unexpected Muse Glimmer generation prefix: {prompt_end!r}")
        ids = result["input_ids"]
        header_length = len(self.generation_header_ids)
        if tuple(ids[0, -header_length:].tolist()) != self.generation_header_ids:
            raise RuntimeError("Processor generation header differs from tokenizer encoding")
        base_length = ids.shape[1] - header_length
        completed = torch.tensor(
            self.answer_header_ids, dtype=ids.dtype, device=ids.device,
        ).unsqueeze(0)
        result["input_ids"] = torch.cat((ids[:, :base_length], completed), dim=1)
        mask = result["attention_mask"]
        result["attention_mask"] = torch.cat((
            mask[:, :base_length],
            torch.ones_like(completed, dtype=mask.dtype, device=mask.device),
        ), dim=1)
        if "position_ids" in result:
            positions = result["position_ids"]
            if positions.ndim != 2 or base_length == 0:
                raise RuntimeError("Unexpected Muse Glimmer position IDs")
            replacement = positions[:, base_length - 1:base_length] + 1 + torch.arange(
                len(self.answer_header_ids), dtype=positions.dtype, device=positions.device,
            ).unsqueeze(0)
            result["position_ids"] = torch.cat((positions[:, :base_length], replacement), dim=1)
        result["generation_query_index"] = int(result["input_ids"].shape[1]) - 1
        return result

    @torch.inference_mode()
    def full_visual_condition(self, processed: dict[str, Any]) -> FullVisualCondition:
        vision_parameter = next(self.model.model.vision_tower.parameters())
        pixel_values = processed["pixel_values"].to(
            device=vision_parameter.device, dtype=vision_parameter.dtype,
        )
        image_grid_thw = processed["image_grid_thw"].to(device=vision_parameter.device)
        features = self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).pooler_output
        if not isinstance(features, (tuple, list)) or not features:
            raise RuntimeError("Muse Glimmer image features must be a nonempty tuple")
        if any(feature.ndim != 2 for feature in features):
            raise RuntimeError("Muse Glimmer image features must have token and hidden dimensions")
        return FullVisualCondition(torch.cat(features, dim=0).detach())

    def prepare(self, processed: dict[str, Any]) -> FullVisualCondition:
        return self.full_visual_condition(processed)

    def _prepare_language_inputs(
        self, processed: dict[str, Any], condition: FullVisualCondition,
    ) -> torch.Tensor:
        input_ids = processed["input_ids"]
        image_mask = input_ids == self.model.config.image_token_id
        video_mask = input_ids == self.model.config.video_token_id
        if bool(video_mask.any().item()):
            raise RuntimeError("This experiment accepts still images only")
        language_model = self.model.model.language_model
        safe_ids = input_ids.masked_fill(image_mask, 0)
        embeddings = language_model.embed_tokens(safe_ids)
        image_features = condition.image_features
        token_count = int(image_mask.sum().item())
        if token_count != image_features.shape[0]:
            raise RuntimeError(
                f"Muse Glimmer image token/feature mismatch: {token_count} != "
                f"{image_features.shape[0]}"
            )
        image_features = image_features.to(device=embeddings.device, dtype=embeddings.dtype)
        expanded_mask = image_mask.unsqueeze(-1).expand_as(embeddings)
        return embeddings.masked_scatter(expanded_mask, image_features)

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

        embeddings = self._prepare_language_inputs(processed, condition)
        try:
            for layer in wanted:
                handles.append(self.decoder_layers[layer - 1].register_forward_hook(capture(layer)))
            if intervention_layer is not None:
                handles.append(self.decoder_layers[intervention_layer - 1].register_forward_hook(intervene))
            outputs = self.model.model.language_model(
                inputs_embeds=embeddings,
                attention_mask=processed["attention_mask"],
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
        text_config = self.model.config.text_config
        cap = float(text_config.final_logit_softcapping)
        logits = logits * float(text_config.output_multiplier)
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
        condition = prepared if prepared is not None else self.prepare(processed)
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
            terminal = self.turn_end_token_id
        if isinstance(terminal, (tuple, list)):
            terminal = terminal[0] if terminal else None
        if terminal is None:
            raise RuntimeError("No assistant turn-end token; set terminal_token_id")
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

def load_engine(config: dict[str, Any]) -> MuseGlimmerEngine:
    return MuseGlimmerEngine.load(
        resolve_path(config["model_path"]), dtype=str(config["dtype"]),
        device_map=str(config["device_map"]),
        attn_implementation=str(config.get("attn_implementation", "sdpa")),
        reasoning_strength=str(config.get("reasoning_strength", "low")),
    )
