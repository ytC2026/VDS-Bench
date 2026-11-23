"""Lazy native engine loading for this model's standalone pipeline."""
from .core import BACKEND, MODEL


def check_backend(config, model=MODEL):
    if model != MODEL or config.get('backend', 'qwen2.5') != BACKEND:
        raise ValueError('Config belongs to another model; use its own entry/config')


def load_engine(config, model=MODEL):
    check_backend(config, model)
    from .model import InternVLEngine
    engine = InternVLEngine.load(config)
    if any(not 1 <= layer <= len(engine.layers) for layer in config['decoder_layers']):
        raise ValueError('Candidate layer outside this model decoder')
    return engine
