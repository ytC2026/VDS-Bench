"""Standalone Qwen full-image adapter for paired Dplus-Dminus steering.

The full-image preprocessing, M-RoPE, scoring and query hooks are taken from
the existing Qwen adapters. No pruning or residual-patching package is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .trie import EncodedOption, score_options
from .io import resolve_path

LETTERS = ("A", "B", "C", "D")


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def hidden_from_layer_output(output: Any) -> torch.Tensor:
    """Extract the residual tensor from common decoder-layer output forms."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported decoder layer output type: {type(output).__name__}")


def replace_hidden_in_layer_output(output: Any, hidden: torch.Tensor) -> Any:
    """Return ``output`` with its residual tensor replaced, preserving its container."""
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"Unsupported decoder layer output type: {type(output).__name__}")


@dataclass(frozen=True)
class VisualCondition:
    image_features: torch.Tensor


class QwenFullImageBase:
    def __init__(self, model: Any, processor: Any):
        self.model = model
        self.processor = processor
        self.base = model.model
        self.visual = self.base.visual
        self.tokenizer = processor.tokenizer
        self.depth = len(self.visual.blocks)
        self.merge_size = int(self.visual.spatial_merge_size)
        self.merge_unit = self.merge_size**2
        self.option_token_ids = self._resolve_option_token_ids()

    @classmethod
    def load(
        cls,
        model_path: Path,
        dtype: str = "bfloat16",
        device_map: str | None = "auto",
        attn_implementation: str = "sdpa",
    ) -> "QwenFullImageBase":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose one of {sorted(dtype_map)}")
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype_map[dtype],
            device_map=device_map,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return cls(model, processor)

    def _resolve_option_token_ids(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for letter in LETTERS:
            token_ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(
                    f"Forced-choice probe requires one token per option; {letter!r} encoded as {token_ids}"
                )
            result[letter] = int(token_ids[0])
        if len(set(result.values())) != len(LETTERS):
            raise RuntimeError(f"Option token IDs are not unique: {result}")
        return result

    def process_messages(self, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        missing = required - set(inputs)
        if missing:
            raise RuntimeError(f"Qwen processor output is missing: {sorted(missing)}")
        if inputs["input_ids"].shape[0] != 1 or inputs["image_grid_thw"].shape[0] != 1:
            raise ValueError("Qwen steering supports one sample and one image per call")
        return dict(inputs)

    def _vision_inputs(self, processed: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        patch_device = _module_device(self.visual.patch_embed)
        patch_dtype = _module_dtype(self.visual.patch_embed)
        pixel_values = processed["pixel_values"].to(device=patch_device, dtype=patch_dtype)
        grid_thw = processed["image_grid_thw"].to(device=patch_device)
        return pixel_values, grid_thw

    def _multimodal_positions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        mrope_positions, _ = self.base.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        return mrope_positions

    def _prepare_language_inputs(
        self,
        processed: dict[str, torch.Tensor],
        condition: VisualCondition,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed_tokens = self.base.language_model.embed_tokens
        embed_device = _module_device(embed_tokens)
        input_ids = processed["input_ids"].to(embed_device)
        attention_mask = processed["attention_mask"].to(embed_device)
        grid = processed["image_grid_thw"].to(embed_device)
        mm_types = processed.get("mm_token_type_ids")
        if mm_types is None:
            mm_types = torch.zeros_like(input_ids, dtype=torch.int32)
            mm_types[input_ids == int(self.model.config.image_token_id)] = 1
        else:
            mm_types = mm_types.to(embed_device)

        image_mask = input_ids[0] == int(self.model.config.image_token_id)
        image_positions = torch.nonzero(image_mask, as_tuple=False).flatten()
        expected_full = int(processed["image_grid_thw"][0].prod().item()) // self.merge_unit
        if image_positions.numel() != expected_full:
            raise RuntimeError(
                f"Processor emitted {image_positions.numel()} image tokens; expected {expected_full}"
            )
        if condition.image_features.shape[0] != expected_full or int((mm_types == 1).sum().item()) != expected_full:
            raise RuntimeError("Full image features do not match image placeholders")

        mrope_positions = self._multimodal_positions(input_ids, attention_mask, grid, mm_types)
        inputs_embeds = embed_tokens(input_ids)
        features = condition.image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[0, image_positions] = features

        # Keep the full sequence and the existing text + temporal/height/width
        # M-RoPE convention used by the baseline Qwen adapter.
        text_positions = attention_mask.long().cumsum(-1) - 1
        text_positions = text_positions.masked_fill(attention_mask == 0, 0)
        position_ids = torch.cat((text_positions.unsqueeze(0), mrope_positions), dim=0)
        return inputs_embeds, attention_mask, position_ids

    @property
    def decoder_layers(self) -> Any:
        layers = getattr(self.base.language_model, "layers", None)
        if layers is None:
            raise RuntimeError("Loaded language model has no .layers decoder stack")
        return layers

    @property
    def decoder_depth(self) -> int:
        return len(self.decoder_layers)

    def _forward_language(
        self,
        processed: dict[str, torch.Tensor],
        condition: VisualCondition,
    ) -> Any:
        inputs_embeds, attention_mask, position_ids = self._prepare_language_inputs(
            processed, condition
        )
        return self.base.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )


class QwenActualTextEngine(QwenFullImageBase):
    def prepare(self, processed: dict[str, torch.Tensor]) -> VisualCondition:
        return self.full_visual_condition(processed)

    def metadata(self) -> dict[str, Any]:
        config = self.model.config
        text_config = getattr(config, "text_config", config)
        return {
            "model_type": str(config.model_type),
            "decoder_depth": self.decoder_depth,
            "hidden_size": int(text_config.hidden_size),
            "vision_depth": self.depth,
            "spatial_merge_size": self.merge_size,
            "option_token_ids": dict(self.option_token_ids),
            "device_map": getattr(self.model, "hf_device_map", None),
            "visual_condition": "full_image",
        }

    def _resolve_option_token_ids(self) -> dict[str, int]:
        return super()._resolve_option_token_ids()

    @torch.inference_mode()
    def full_visual_condition(
        self, processed: dict[str, torch.Tensor]
    ) -> VisualCondition:
        pixel_values, grid = self._vision_inputs(processed)
        features = self.visual(
            pixel_values, grid_thw=grid, return_dict=True
        ).pooler_output
        return VisualCondition(image_features=features)

    def terminal_token_id(self, configured: int | None = None) -> int:
        if configured is not None:
            return int(configured)
        value = self.tokenizer.eos_token_id
        if isinstance(value, (tuple, list)):
            if not value:
                raise RuntimeError("Tokenizer has no EOS token")
            value = value[0]
        if value is None:
            raise RuntimeError("Tokenizer has no EOS token; configure terminal_token_id")
        return int(value)

    def encode_options(self, option_texts: Mapping[str, str]) -> list[EncodedOption]:
        result = []
        for key, raw_text in option_texts.items():
            text = str(raw_text).strip()
            token_ids = tuple(
                int(value)
                for value in self.tokenizer.encode(text, add_special_tokens=False)
            )
            if not token_ids:
                raise RuntimeError(f"Option {key!r} has no tokens: {text!r}")
            result.append(EncodedOption(str(key), text, token_ids))
        return result

    @staticmethod
    def _with_prefix(
        processed: dict[str, torch.Tensor], prefix: tuple[int, ...]
    ) -> dict[str, torch.Tensor]:
        if not prefix:
            return processed
        result = dict(processed)
        ids = processed["input_ids"]
        suffix = torch.tensor(prefix, dtype=ids.dtype, device=ids.device).unsqueeze(0)
        result["input_ids"] = torch.cat((ids, suffix), dim=1)
        mask = processed["attention_mask"]
        result["attention_mask"] = torch.cat(
            (mask, torch.ones_like(suffix, dtype=mask.dtype, device=mask.device)), dim=1
        )
        if "mm_token_type_ids" in processed:
            types = processed["mm_token_type_ids"]
            result["mm_token_type_ids"] = torch.cat(
                (types, torch.zeros_like(suffix, dtype=types.dtype, device=types.device)),
                dim=1,
            )
        return result

    @torch.inference_mode()
    def _forward_once(
        self,
        processed: dict[str, torch.Tensor],
        condition: VisualCondition,
        *,
        capture_layers: Iterable[int] = (),
        intervention_layer: int | None = None,
        intervention_direction: torch.Tensor | None = None,
        intervention_ratio: float | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[str, float] | None]:
        wanted = sorted(set(int(layer) for layer in capture_layers))
        if wanted and (wanted[0] < 1 or wanted[-1] > self.decoder_depth):
            raise ValueError(f"Capture layers outside 1..{self.decoder_depth}: {wanted}")
        states: dict[int, torch.Tensor] = {}
        handles = []
        intervention_values = (intervention_layer, intervention_direction, intervention_ratio)
        if any(value is None for value in intervention_values) and not all(
            value is None for value in intervention_values
        ):
            raise ValueError("Intervention layer, direction, and ratio must be provided together")
        if intervention_layer is not None and not 1 <= int(intervention_layer) <= self.decoder_depth:
            raise ValueError(f"Intervention layer outside 1..{self.decoder_depth}")
        if intervention_ratio is not None and not 0 < abs(float(intervention_ratio)) < 1:
            raise ValueError("Intervention ratio must be signed and strictly between -1 and 1")
        intervention_metrics: dict[str, float] | None = None

        def capture_hook(layer: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                hidden = hidden_from_layer_output(output)
                states[layer] = hidden[0, -1].detach().float().cpu().clone()
            return hook

        def intervene(_module: Any, _inputs: Any, output: Any) -> Any:
            nonlocal intervention_metrics
            hidden = hidden_from_layer_output(output)
            vector = intervention_direction.to(hidden.device, torch.float32)
            if tuple(vector.shape) != tuple(hidden.shape[-1:]):
                raise ValueError("Intervention direction does not match hidden size")
            vector_norm = torch.linalg.vector_norm(vector)
            query = hidden[0, -1]
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
                (patched[0, -1].float() - query.float())
            ).item())
            reference_norm = float(hidden_norm.item())
            intervention_metrics = {
                "requested_signed_ratio": float(intervention_ratio),
                "pre_hidden_norm": reference_norm,
                "requested_delta_norm": abs(float(intervention_ratio)) * reference_norm,
                "effective_delta_norm": effective_delta_norm,
                "effective_ratio": effective_delta_norm / reference_norm,
            }
            return replace_hidden_in_layer_output(output, patched)

        try:
            for layer in wanted:
                handles.append(
                    self.decoder_layers[layer - 1].register_forward_hook(capture_hook(layer))
                )
            if intervention_layer is not None:
                handles.append(
                    self.decoder_layers[int(intervention_layer) - 1].register_forward_hook(
                        intervene
                    )
                )
            outputs = self._forward_language(processed, condition)
        finally:
            for handle in handles:
                handle.remove()
        hidden = outputs.last_hidden_state[:, -1, :]
        head = self.model.lm_head
        parameter = next(head.parameters())
        logits = head(hidden.to(parameter.device, parameter.dtype))[0].float()
        if set(states) != set(wanted):
            raise RuntimeError(f"Captured layers {sorted(states)}, expected {wanted}")
        if intervention_layer is not None and intervention_metrics is None:
            raise RuntimeError("Intervention hook did not run")
        return logits, states, intervention_metrics

    @torch.inference_mode()
    def score_actual_options(
        self,
        processed: dict[str, torch.Tensor],
        option_texts: Mapping[str, str],
        *,
        prepared: VisualCondition | None = None,
        capture_layers: Iterable[int] = (),
        terminal_token_id: int | None = None,
        intervention_layer: int | None = None,
        intervention_direction: torch.Tensor | None = None,
        intervention_ratio: float | None = None,
    ) -> dict[str, Any]:
        condition = prepared if prepared is not None else self.prepare(processed)
        encoded = self.encode_options(option_texts)
        terminal = self.terminal_token_id(terminal_token_id)
        root_logits, states, root_metrics = self._forward_once(
            processed,
            condition,
            capture_layers=capture_layers,
            intervention_layer=intervention_layer,
            intervention_direction=intervention_direction,
            intervention_ratio=intervention_ratio,
        )
        cache: dict[tuple[int, ...], torch.Tensor] = {(): root_logits}
        prefix_metrics: dict[tuple[int, ...], dict[str, float]] = {}

        def provider(
            prefix: tuple[int, ...], children: tuple[int, ...]
        ) -> Mapping[int, float]:
            if prefix not in cache:
                prefixed = self._with_prefix(processed, prefix)
                cache[prefix], _unused, metric = self._forward_once(
                    prefixed,
                    condition,
                    intervention_layer=intervention_layer,
                    intervention_direction=intervention_direction,
                    intervention_ratio=intervention_ratio,
                )
                if metric is not None:
                    prefix_metrics[prefix] = metric
            values = cache[prefix]
            return {token: float(values[token].item()) for token in children}

        scored = score_options(encoded, terminal, provider)
        answer_key = str(scored["answer_key"])
        text_by_key = {option.key: option.text for option in encoded}
        return {
            **scored,
            "answer_text": text_by_key[answer_key],
            "option_texts": text_by_key,
            "option_token_ids": {
                option.key: list(option.token_ids) for option in encoded
            },
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
        self,
        processed: dict[str, torch.Tensor],
        condition: VisualCondition,
        *,
        intervention_layer: int | None = None,
        intervention_direction: torch.Tensor | None = None,
        intervention_ratio: float | None = None,
    ) -> dict[str, Any]:
        logits, _states, metrics = self._forward_once(
            processed,
            condition,
            intervention_layer=intervention_layer,
            intervention_direction=intervention_direction,
            intervention_ratio=intervention_ratio,
        )
        option_logits = {
            letter: float(logits[self.option_token_ids[letter]].item())
            for letter in LETTERS
        }
        return {
            "answer": max(LETTERS, key=option_logits.get),
            "option_logits": option_logits,
            "option_token_ids": dict(self.option_token_ids),
            "intervention_metrics": metrics,
        }

def load_engine(config: dict[str, Any]) -> QwenActualTextEngine:
    return QwenActualTextEngine.load(
        resolve_path(config["model_path"]), dtype=str(config["dtype"]),
        device_map=str(config["device_map"]),
        attn_implementation=str(config.get("attn_implementation", "sdpa")),
    )
