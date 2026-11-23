"""DeepSeek-VL2 preprocessing, scoring, and relative-norm steering."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .trie import EncodedOption, score_options


LETTERS = ("A", "B", "C", "D")


def _hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unexpected decoder layer output: {type(output).__name__}")


def _replace(output: Any, value: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return value
    if isinstance(output, tuple):
        return (value, *output[1:])
    if isinstance(output, list):
        return [value, *output[1:]]
    raise TypeError(type(output).__name__)


class DeepSeekVL2Engine:
    def __init__(self, model: Any, processor: Any):
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.layers = model.language.model.layers
        self.option_token_ids = {}
        for letter in LETTERS:
            ids = self._answer_ids(letter)
            if len(ids) != 1:
                raise RuntimeError(f"A/B/C/D answer is not one token: {letter} -> {ids}")
            self.option_token_ids[letter] = ids[0]
        if len(set(self.option_token_ids.values())) != 4:
            raise RuntimeError(f"Letter tokens are not distinct: {self.option_token_ids}")

    @classmethod
    def load(cls, model_path: str | Path, *, dtype: str = "bfloat16", device_map: str = "auto"):
        from transformers import AutoModelForCausalLM
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Missing local DeepSeek-VL2 checkpoint: {path}")
        tensor_dtype = getattr(torch, dtype)
        # Place the complete checkpoint on the configured GPU. CPU offload
        # cannot handle MLA's direct kv_b_proj.weight access.
        runtime_device_map = os.environ.get("VDS_DEEPSEEK_DEVICE_MAP", device_map)
        if runtime_device_map == "cuda":
            runtime_device_map = {"": 0}
        elif isinstance(runtime_device_map, str) and runtime_device_map.startswith("cuda:"):
            runtime_device_map = {"": int(runtime_device_map.split(":", 1)[1])}
        # The official outer VL2 class leaves this list empty; keep each MoE
        # decoder block and the image encoder intact when Accelerate shards it.
        DeepseekVLV2ForCausalLM._no_split_modules = [
            "DeepseekV2DecoderLayer", "VisionTransformer",
        ]
        processor = DeepseekVLV2Processor.from_pretrained(str(path), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True,
            torch_dtype=tensor_dtype, device_map=runtime_device_map, low_cpu_mem_usage=True,
        )
        if str(model.config.model_type) != "deepseek_vl_v2":
            raise RuntimeError(f"Expected deepseek_vl_v2, got {model.config.model_type}")
        print(f"[DeepSeek] device_map={getattr(model, 'hf_device_map', None)}", flush=True)
        return cls(model, processor)

    @property
    def decoder_depth(self) -> int:
        return len(self.layers)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": str(self.model.config.model_type),
            "decoder_depth": self.decoder_depth,
            "hidden_size": int(self.model.config.language_config.hidden_size),
            "option_token_ids": self.option_token_ids,
            "device_map": getattr(self.model, "hf_device_map", None),
        }

    def _answer_ids(self, answer: str) -> tuple[int, ...]:
        # SentencePiece may retokenize the final ':' when answer text is
        # concatenated to the prompt. Score answer tokens as a continuation
        # instead, using the same forced-choice convention as the other models.
        ids = tuple(int(x) for x in self.tokenizer.encode(answer, add_special_tokens=False))
        if not ids:
            raise RuntimeError(f"Tokenizer produced no tokens for answer {answer!r}")
        return ids

    def process_messages(self, messages: list[dict[str, Any]]) -> Any:
        """Convert the local system/user/assistant protocol to DeepSeek's template."""
        if messages[0]["role"] != "system":
            raise ValueError("Expected first system message")
        conversation = []
        images = []
        for message in messages[1:]:
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    image_parts = [part["image"] for part in content if part["type"] == "image"]
                    text_parts = [part["text"] for part in content if part["type"] == "text"]
                    if len(image_parts) != 1 or len(text_parts) != 1:
                        raise ValueError("Expected one image and one text in first user turn")
                    images.extend(image_parts)
                    body = "<image>\n" + messages[0]["content"] + "\n\n" + text_parts[0]
                else:
                    body = str(content)
                conversation.append({"role": "<|User|>", "content": body})
            elif message["role"] == "assistant":
                conversation.append({"role": "<|Assistant|>", "content": str(message["content"])})
            else:
                raise ValueError(f"Unsupported role: {message['role']}")
        if len(images) != 1:
            raise ValueError(f"Expected exactly one image, got {len(images)}")
        conversation.append({"role": "<|Assistant|>", "content": ""})
        batch = self.processor(
            conversations=conversation, images=images, force_batchify=True,
            inference_mode=True, system_prompt="",
        )
        if not batch.sft_format or not str(batch.sft_format[0]).rstrip().endswith("<|Assistant|>:"):
            raise RuntimeError("Processor output does not end at the Assistant query marker")
        if batch.input_ids.shape[1] > int(self.model.config.language_config.max_position_embeddings):
            raise RuntimeError(f"Prompt exceeds language context: {batch.input_ids.shape[1]}")
        # prepare_inputs_embeds currently requires the vision and embedding
        # modules on the same device; decoder layers may still be sharded.
        vision_device = next(self.model.vision.parameters()).device
        embed_device = self.model.language.get_input_embeddings().weight.device
        if vision_device != embed_device:
            raise RuntimeError(
                f"DeepSeek vision {vision_device} and token embedding {embed_device} "
                "must share a device for prepare_inputs_embeds"
            )
        return batch.to(embed_device, dtype=next(self.model.vision.parameters()).dtype)

    @torch.inference_mode()
    def prepare(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.model.prepare_inputs_embeds(**batch).detach()
        return embeddings, batch.attention_mask

    @torch.inference_mode()
    def _forward(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor,
        *, capture_layers: Iterable[int] = (), intervention_layer: int | None = None,
        intervention_direction: torch.Tensor | None = None,
        intervention_ratio: float | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[str, float] | None]:
        wanted = sorted(set(int(x) for x in capture_layers))
        if any(not 1 <= x <= self.decoder_depth for x in wanted):
            raise ValueError(f"Capture layer outside 1..{self.decoder_depth}: {wanted}")
        intervention_values = (intervention_layer, intervention_direction, intervention_ratio)
        if any(value is None for value in intervention_values) and not all(
            value is None for value in intervention_values
        ):
            raise ValueError("Intervention layer, direction, and ratio must be provided together")
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

        try:
            for layer in wanted:
                handles.append(self.layers[layer - 1].register_forward_hook(capture(layer)))
            if intervention_layer is not None:
                handles.append(self.layers[intervention_layer - 1].register_forward_hook(intervene))
            output = self.model.language.model(
                inputs_embeds=embeddings, attention_mask=attention_mask,
                use_cache=False, return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        if set(states) != set(wanted):
            raise RuntimeError(f"Captured {sorted(states)}; expected {wanted}")
        head = self.model.language.lm_head
        final = output.last_hidden_state[:, -1, :]
        logits = head(final.to(head.weight.device, head.weight.dtype))[0].float()
        if intervention_layer is not None and intervention_metrics is None:
            raise RuntimeError("Intervention hook did not run")
        return logits, states, intervention_metrics

    @torch.inference_mode()
    def score_actual_options(
        self, batch: Any, option_texts: Mapping[str, str], *,
        prepared: tuple[torch.Tensor, torch.Tensor] | None = None,
        capture_layers: Iterable[int] = (), terminal_token_id: int | None = None,
        intervention_layer: int | None = None,
        intervention_direction: torch.Tensor | None = None,
        intervention_ratio: float | None = None,
    ) -> dict[str, Any]:
        encoded = [
            EncodedOption(str(key), str(text).strip(),
                          self._answer_ids(str(text).strip()))
            for key, text in option_texts.items()
        ]
        terminal = self.tokenizer.eos_token_id if terminal_token_id is None else terminal_token_id
        if isinstance(terminal, (tuple, list)):
            terminal = terminal[0] if terminal else None
        if terminal is None:
            raise RuntimeError("Tokenizer has no EOS token; configure terminal_token_id")
        terminal = int(terminal)
        base_embeds, base_mask = prepared if prepared is not None else self.prepare(batch)
        root_logits, states, root_metrics = self._forward(
            base_embeds, base_mask, capture_layers=capture_layers,
            intervention_layer=intervention_layer,
            intervention_direction=intervention_direction,
            intervention_ratio=intervention_ratio,
        )
        cache: dict[tuple[int, ...], torch.Tensor] = {(): root_logits}
        prefix_metrics: dict[tuple[int, ...], dict[str, float]] = {}

        def provider(prefix: tuple[int, ...], children: tuple[int, ...]) -> dict[int, float]:
            if prefix not in cache:
                ids = torch.tensor([prefix], dtype=torch.long,
                                   device=self.model.language.get_input_embeddings().weight.device)
                suffix = self.model.language.get_input_embeddings()(ids)
                embeds = torch.cat((base_embeds, suffix.to(base_embeds.device)), dim=1)
                mask = torch.cat((base_mask, torch.ones_like(ids, dtype=base_mask.dtype,
                                                             device=base_mask.device)), dim=1)
                cache[prefix], _unused, metric = self._forward(
                    embeds, mask,
                    intervention_layer=intervention_layer,
                    intervention_direction=intervention_direction,
                    intervention_ratio=intervention_ratio,
                )
                if metric is not None:
                    prefix_metrics[prefix] = metric
            return {token: float(cache[prefix][token].item()) for token in children}

        scored = score_options(encoded, terminal, provider)
        by_key = {x.key: x.text for x in encoded}
        return {
            **scored, "answer_text": by_key[scored["answer_key"]],
            "option_texts": by_key,
            "option_token_ids": {x.key: list(x.token_ids) for x in encoded},
            "terminal_token_id": terminal, "states": states,
            "scoring_method": "locally_normalized_valid_continuation_trie",
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
        self, batch: Any, prepared: tuple[torch.Tensor, torch.Tensor] | None = None,
        *, intervention_layer: int | None = None,
        intervention_direction: torch.Tensor | None = None,
        intervention_ratio: float | None = None,
    ) -> dict[str, Any]:
        embeds, mask = prepared if prepared is not None else self.prepare(batch)
        logits, _states, metrics = self._forward(
            embeds, mask, intervention_layer=intervention_layer,
            intervention_direction=intervention_direction,
            intervention_ratio=intervention_ratio,
        )
        values = {letter: float(logits[token].item())
                  for letter, token in self.option_token_ids.items()}
        return {
            "answer": max(LETTERS, key=values.get),
            "option_logits": values,
            "option_token_ids": self.option_token_ids,
            "intervention_metrics": metrics,
        }
