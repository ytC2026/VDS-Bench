"""Native Transformers 4.57.1 LLaVA-NeXT with Mistral decoder hooks."""
import gc
import importlib.metadata

import torch

from mub.llava.core import resolve
from mub.llava.engine import PairedEngine
from mub.llava.messages import native_messages


class LlavaNextEngine(PairedEngine):
    def __init__(self, model, processor):
        self.model, self.processor = model.eval(), processor
        if (model.config.model_type != 'llava_next'
                or model.config.text_config.model_type != 'mistral'):
            raise ValueError('Use llava-hf/llava-v1.6-mistral-7b-hf, not another LLaVA backbone')
        self.layers = model.model.language_model.layers
        if len(self.layers) != 32:
            raise ValueError('This baseline expects the 32-layer Mistral-7B decoder')
        processor.patch_size = model.config.vision_config.patch_size
        processor.num_additional_image_tokens = 1
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
        if not processor.chat_template:
            raise ValueError('Missing official multimodal chat_template.json in the local checkpoint')
        processor.tokenizer.padding_side = 'left'

    @classmethod
    def load(cls, config):
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        gc.collect()
        torch.cuda.empty_cache()
        path = str(resolve(config['model_path']))
        processor = AutoProcessor.from_pretrained(
            path, local_files_only=True, trust_remote_code=False)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            dtype=getattr(torch, config['dtype']), device_map=config['device_map'],
            attn_implementation=config['attn_implementation'], low_cpu_mem_usage=True)
        torch.manual_seed(config['seed'])
        return cls(model, processor)

    def metadata(self):
        return {
            'model_type': self.model.config.model_type,
            'text_model_type': self.model.config.text_config.model_type,
            'decoder_depth': len(self.layers),
            'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'accelerate')},
            'device_map': {k: str(v) for k, v in getattr(self.model, 'hf_device_map', {}).items()},
            'training': 'shared actual-option text trie; independent per-model fit and held-out calibration',
            'evaluation': 'shared post-answer history, sampled generation and MUB extract_option',
            'serialization': 'official processor multimodal template; all role contents are lists',
            'chat_template': self.processor.chat_template,
            'processor_class': type(self.processor.image_processor).__name__,
            'tokenizer_class': type(self.processor.tokenizer).__name__,
            'image_processing': 'native LLaVA-NeXT any-resolution crops and image token expansion',
            'intervention': 'decoder output at last prefill token; once per generated answer',
        }

    def prepare(self, history, image):
        text = self.processor.apply_chat_template(
            native_messages(history), tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=False,
                                truncation=False, return_tensors='pt')
        if inputs['input_ids'].shape[0] != 1 or tuple(inputs['image_sizes'].shape) != (1, 2):
            raise ValueError('Exactly one question and one full image per forward are required')
        if not (inputs['input_ids'] == self.model.config.image_token_index).any():
            raise ValueError('The official template did not render an image token')
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.model.dtype)
        return inputs
