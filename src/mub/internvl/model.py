"""Official InternVL3-8B vision/Qwen2.5 decoder with post-answer hooks."""
import gc
import importlib
import importlib.metadata
from types import SimpleNamespace
import weakref

import torch

from .core import resolve
from .engine import PairedEngine
from .messages import render_messages
from .preprocess import pixels


class InternVLEngine(PairedEngine):
    def __init__(self, model, tokenizer, config):
        self.model, self.processor, self.config = model.eval(), SimpleNamespace(tokenizer=tokenizer), config
        if model.config.model_type != 'internvl_chat' or model.config.llm_config.model_type != 'qwen2':
            raise ValueError('Use OpenGVLab/InternVL3-8B with the Qwen2.5 backbone')
        self.layers = model.language_model.model.layers
        # InternVL3's published checkpoint deliberately retains this template name.
        if len(self.layers) != 28 or model.template != 'internvl2_5':
            raise ValueError('Expected the 28-layer InternVL3-8B checkpoint')
        if config['image_size'] != (model.config.force_image_size or model.config.vision_config.image_size):
            raise ValueError('Image size differs from the checkpoint')
        self.get_template = importlib.import_module(type(model).__module__).get_conv_template
        self.separator = self.get_template(model.template).sep.strip()
        self.turn_end_token_id = tokenizer.convert_tokens_to_ids(self.separator)
        self.image_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        for token, number in [(self.separator, self.turn_end_token_id), ('<IMG_CONTEXT>', self.image_token_id)]:
            if tokenizer.convert_ids_to_tokens(number) != token:
                raise ValueError(f'Missing checkpoint token: {token}')
        model.img_context_token_id = self.image_token_id
        tokenizer.padding_side = 'left'
        self._image_ref, self._visual = None, None

    @classmethod
    def load(cls, config):
        import transformers
        from transformers import AutoModel, AutoTokenizer
        if transformers.__version__ != config['transformers_version']:
            raise RuntimeError('Use the separate InternVL environment from src/mub/internvl/requirements.txt; '
                               f"expected Transformers {config['transformers_version']}, got {transformers.__version__}")
        gc.collect()
        torch.cuda.empty_cache()
        path = str(resolve(config['model_path']))
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
        model = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True,
                    torch_dtype=getattr(torch, config['dtype']), device_map=config['device_map'],
                    low_cpu_mem_usage=True, use_flash_attn=False)
        torch.manual_seed(config['seed'])
        return cls(model, tokenizer, config)

    def metadata(self):
        return {'model_type': self.model.config.model_type, 'text_model_type': 'qwen2',
                'decoder_depth': len(self.layers), 'model_id': 'OpenGVLab/InternVL3-8B',
                'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'accelerate', 'timm')},
                'device_map': {k: str(v) for k, v in getattr(self.model, 'hf_device_map', {}).items()},
                'template': self.model.template, 'turn_end_token_id': self.turn_end_token_id,
                'image_tokens_per_tile': self.model.num_image_token,
                'image_processing': 'official dynamic tiles, optional thumbnail and ImageNet normalization',
                'training': 'contextual actual-option text trie; both arms share real unsteered initial answer',
                'evaluation': 'native sampled generation; shared baseline history; official MUB extraction',
                'lm_head': 'native projection on last query state; full prefill KV retained',
                'intervention': 'decoder output; once at the last prefill token per generated answer'}

    @torch.inference_mode()
    def prepare(self, history, image):
        if self._image_ref is None or self._image_ref() is not image:
            device = next(self.model.vision_model.parameters()).device
            values = pixels(image, self.config).to(device=device, dtype=self.model.dtype)
            features = self.model.extract_feature(values)
            self._image_ref, self._visual = weakref.ref(image), (values, features)
        values, features = self._visual
        template = self.get_template(self.model.template)
        template.system_message = self.model.system_message
        text = render_messages(history, template, tiles=len(values), image_tokens=self.model.num_image_token)
        inputs = self.processor.tokenizer(text, return_tensors='pt', padding=False, truncation=False)
        device = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if inputs['input_ids'].shape[0] != 1 or int((inputs['input_ids'] == self.image_token_id).sum()) != features.shape[0] * features.shape[1]:
            raise ValueError('Image tokens and visual features do not match')
        if inputs['input_ids'].shape[1] + self.config['generation']['max_new_tokens'] > self.model.config.llm_config.max_position_embeddings:
            raise ValueError('Question would exceed native context length; truncation is disabled')
        return {**inputs, 'pixel_values': values, 'visual_features': features, '_prompt_text': text}

    def option_tokens(self, inputs, text):
        prefix = inputs['input_ids'][0].tolist()
        complete = self.processor.tokenizer.encode(inputs['_prompt_text'] + text)
        if complete[:len(prefix)] != prefix or len(complete) == len(prefix):
            raise ValueError('Unstable option completion boundary')
        return tuple(complete[len(prefix):])

    def embeddings(self, inputs):
        result = self.model.get_input_embeddings()(inputs['input_ids']).clone()
        mask = inputs['input_ids'] == self.image_token_id
        visual = inputs['visual_features'].reshape(-1, result.shape[-1]).to(device=result.device, dtype=result.dtype)
        if int(mask.sum()) != len(visual):
            raise ValueError('Image feature count changed during continuation scoring')
        result[mask] = visual
        return result

    def forward_logits(self, inputs):
        output = self.model.language_model.model(inputs_embeds=self.embeddings(inputs),
                    attention_mask=inputs['attention_mask'], use_cache=False, return_dict=True)
        return self.model.language_model.get_output_embeddings()(output.last_hidden_state[:, -1:]).float()[0, -1]

    def generate_tokens(self, inputs, generation):
        settings = {**generation, 'eos_token_id': self.turn_end_token_id,
                    'pad_token_id': self.processor.tokenizer.pad_token_id, 'num_logits_to_keep': 1}
        return self.model.generate(pixel_values=inputs['pixel_values'],
                    visual_features=inputs['visual_features'], input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'], **settings)
