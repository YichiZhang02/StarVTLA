"""Strict, local-only import of native N0-VTLA safetensors."""
import logging
from pathlib import Path

from safetensors.torch import load_file


def native_key(key):
    if key.startswith("model."):
        key = key[6:]
    key = key.replace("tactile_prior.", "tactile_predictor.")
    for component in ("language_model", "vision_tower", "multi_modal_projector"):
        key = key.replace(f"paligemma.{component}.", f"paligemma.model.{component}.")
    return key


def load_native_weights(model, directory, reinitialize_action_projections=False):
    path = Path(directory).expanduser()
    path = path / "model.safetensors" if path.is_dir() else path
    if not path.is_file():
        raise FileNotFoundError(f"N0-VTLA weights not found: {path}")
    source = load_file(str(path), device="cpu")
    weights = {native_key(k): v for k, v in source.items()}
    target = model.state_dict()
    # PaliGemma ties its input embedding and LM head. safetensors may retain only one.
    embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    head = "paligemma_with_expert.paligemma.lm_head.weight"
    if embedding not in weights and head in weights:
        weights[embedding] = weights[head]
    if head not in weights and embedding in weights and head in target:
        weights[head] = weights[embedding]
    projection_keys = {"action_in_proj.weight", "action_in_proj.bias", "action_out_proj.weight", "action_out_proj.bias"}
    mismatches = [k for k in weights.keys() & target.keys() if weights[k].shape != target[k].shape]
    if mismatches and (not reinitialize_action_projections or set(mismatches) - projection_keys):
        raise ValueError(f"Native checkpoint shape mismatch: {mismatches}")
    if mismatches:
        logging.warning("Reinitializing action projections for changed action width; post-training required.")
        for key in projection_keys:
            weights.pop(key, None)
    missing = set(target) - set(weights)
    unexpected = set(weights) - set(target)
    allowed_missing = projection_keys if mismatches else set()
    if missing - allowed_missing or unexpected:
        raise ValueError(f"Native checkpoint contract mismatch. Missing={sorted(missing - allowed_missing)}, "
                         f"unexpected={sorted(unexpected)}. Check predictor_arch, gates and backbone config.")
    model.load_state_dict(weights, strict=not bool(mismatches))
    logging.info("Loaded native N0-VTLA weights from %s; actions use the configured StarVTLA semantics.", path)
