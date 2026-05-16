import os

import torch


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state
    return {"model_state_dict": state}


def checkpoint_value(primary, fallback, key, default):
    if key in primary:
        return primary[key]
    if fallback is not None and key in fallback:
        return fallback[key]
    return default


def build_model_from_checkpoint(model_cls, ckpt_payload, device, default_config=None, config_key="model_config"):
    config = {}
    if default_config:
        config.update(default_config)
    config.update(ckpt_payload.get(config_key, {}))
    model = model_cls(**config).to(device)
    model.load_state_dict(ckpt_payload["model_state_dict"])
    return model, config
