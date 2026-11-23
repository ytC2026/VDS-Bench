"""Shared decoder hooks, option-text trie scoring and native generation."""
import numpy as np
import torch
from mub.internvl.core import LETTERS
from steering.qwen.trie import EncodedOption, score_options


def hidden_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported decoder output: {type(output)}")


def replace_hidden(output, hidden):
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(type(output))


class PairedEngine:
    @torch.inference_mode()
    def _forward(self, inputs, capture=(), layer=None, axis=None, ratio=0.):
        wanted = sorted(set(capture))
        if any(not 1 <= l <= len(self.layers) for l in wanted):
            raise ValueError(f"Capture layers outside 1..{len(self.layers)}")
        if ratio and (layer is None or axis is None or not 1 <= layer <= len(self.layers) or not 0 < abs(ratio) < 1):
            raise ValueError("Nonzero steering needs a valid layer, axis and signed norm ratio")
        states, handles, intervention = {}, [], {}

        def capture_hook(number):
            def hook(_module, _args, output):
                states[str(number)] = hidden_tensor(output)[0, -1].detach().float().cpu().numpy().copy()
            return hook

        def inject(_module, _args, output):
            hidden = hidden_tensor(output)
            before = hidden[0, -1]
            direction = torch.as_tensor(axis, device=hidden.device, dtype=torch.float32)
            norm, reference = direction.norm(), before.float().norm()
            if direction.shape != before.shape or not torch.isfinite(direction).all() or not torch.isfinite(reference) or norm <= 0 or reference <= 0:
                raise ValueError("Invalid direction or query residual")
            delta = (float(ratio) * reference * direction / norm).to(before.dtype)
            patched = hidden.clone()
            patched[0, -1] = before + delta
            effective = (patched[0, -1].float() - before.float()).norm()
            intervention.update(requested_signed_ratio=float(ratio), pre_hidden_norm=float(reference),
                                effective_ratio=float(effective / reference))
            return replace_hidden(output, patched)

        try:
            for number in wanted:
                handles.append(self.layers[number - 1].register_forward_hook(capture_hook(number)))
            if ratio:
                handles.append(self.layers[layer - 1].register_forward_hook(inject))
            # Fresh full forward every time; no KV cache or steered initial history.
            logits = self.forward_logits(inputs)
        finally:
            for handle in handles:
                handle.remove()
        if set(states) != {str(l) for l in wanted} or (ratio and not intervention):
            raise RuntimeError("A decoder hook failed to run")
        return logits, states, intervention or None

    @torch.inference_mode()
    def score(self, inputs, *, options, capture=(), layer=None, axis=None, ratio=0.):
        """Exactly steering's actual-text trie scoring, including prefix forwards."""
        encoded = [EncodedOption(key, options[key], self.option_tokens(inputs, options[key])) for key in LETTERS]
        root, states, metric = self._forward(inputs, capture, layer, axis, ratio)
        cache = {(): root}

        def provider(prefix, children):
            if prefix not in cache:
                extended = dict(inputs)
                suffix = torch.tensor([prefix], dtype=inputs['input_ids'].dtype,
                                      device=inputs['input_ids'].device)
                extended['input_ids'] = torch.cat((inputs['input_ids'], suffix), dim=1)
                extended['attention_mask'] = torch.cat((inputs['attention_mask'],
                    torch.ones_like(suffix, dtype=inputs['attention_mask'].dtype)), dim=1)
                cache[prefix] = self._forward(extended, (), layer, axis, ratio)[0]
            return {token: float(cache[prefix][token]) for token in children}

        result = score_options(encoded, self.turn_end_token_id, provider)
        if not np.isfinite(list(result['choice_log_scores'].values())).all():
            raise ValueError('Non-finite option-text scores')
        return {'answer': result['answer_key'], 'answer_text': options[result['answer_key']],
                'choice_log_scores': result['choice_log_scores'], 'intervention': metric}, states

    @torch.inference_mode()
    def generate(self, inputs, generation, seed, layer=None, axis=None, ratio=0.):
        """Native generation; inject once at the final prefill position."""
        from transformers import set_seed
        set_seed(seed)
        handle, metric, fired = None, {}, False
        if ratio:
            if layer is None or not 1 <= layer <= len(self.layers) or axis is None or not 0 < abs(ratio) < 1:
                raise ValueError('Invalid generation intervention')

            def inject(_module, _args, output):
                nonlocal fired
                if fired:
                    return output
                fired = True
                hidden = hidden_tensor(output)
                before = hidden[0, -1]
                direction = torch.as_tensor(axis, device=hidden.device, dtype=torch.float32)
                norm, reference = direction.norm(), before.float().norm()
                if direction.shape != before.shape or not torch.isfinite(direction).all() or norm <= 0 or not torch.isfinite(reference) or reference <= 0:
                    raise ValueError('Invalid direction or query residual')
                patched = hidden.clone()
                patched[0, -1] = before + (ratio * reference * direction / norm).to(before.dtype)
                metric.update(requested_signed_ratio=float(ratio), pre_hidden_norm=float(reference),
                              effective_ratio=float((patched[0, -1].float() - before.float()).norm() / reference))
                return replace_hidden(output, patched)

            handle = self.layers[layer - 1].register_forward_hook(inject)
        try:
            output = self.generate_tokens(inputs, generation)
        finally:
            if handle is not None:
                handle.remove()
        if ratio and not fired:
            raise RuntimeError('Generation hook did not run')
        text = self.processor.tokenizer.decode(output[0], skip_special_tokens=True).split(self.separator)[0].strip()
        return {'text': text, 'intervention': metric or None}
