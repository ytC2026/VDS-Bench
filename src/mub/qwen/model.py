"""Qwen2.5-VL native full multimodal forward with decoder output hooks.

Uses transformers 4.57.1's Qwen2_5_VLForConditionalGeneration, leaving image
encoding and M-RoPE to the model. No Qwen3 manual position-ID code is reused.
"""
from __future__ import annotations

import importlib.metadata
import gc
import torch

from mub.qwen.core import resolve
from mub.qwen.engine import PairedEngine


class Qwen25Engine(PairedEngine):
    def __init__(self, model, processor):
        self.model, self.processor = model.eval(), processor
        if model.config.model_type != "qwen2_5_vl":
            raise ValueError("This research adapter requires Qwen2.5-VL")
        self.layers = model.model.language_model.layers

    @classmethod
    def load(cls, config):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        gc.collect()
        torch.cuda.empty_cache()
        path = str(resolve(config["model_path"]))
        processor = AutoProcessor.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            dtype=getattr(torch, config["dtype"]), device_map=config["device_map"],
            attn_implementation=config["attn_implementation"], low_cpu_mem_usage=True,
        )
        torch.manual_seed(config["seed"])
        return cls(model, processor)

    def metadata(self):
        return {"model_type": self.model.config.model_type, "decoder_depth": len(self.layers),
                "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate")},
                "device_map": {k: str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()},
                "training": "actual-option text trie scoring; history specified by the protocol manifest",
                "evaluation": "sampled generation and MUB extract_option; history specified by the protocol manifest",
                "position_ids": "native Qwen2.5-VL multimodal forward; no manual M-RoPE reconstruction"}

    def prepare(self, history, image):
        text = self.processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=False, return_tensors="pt")
        if inputs["input_ids"].shape[0] != 1 or inputs["image_grid_thw"].shape[0] != 1:
            raise ValueError("Exactly one question and one full image per forward are required")
        return inputs.to(self.model.get_input_embeddings().weight.device)

