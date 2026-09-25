"""Strict offline reconstruction of the epoch-1 v6 world model."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from tacmind0.tactile.encoder import _Backbone

from .module import MLP, ARPredictor
from .multi_source_action import MultiSourceActionEmbedder
from .visuo_jepa import DualEncoderJEPA

RESOURCE = (
    Path(__file__).resolve().parents[3]
    / "playground"
    / "pretrained_models"
    / "tacmind0"
    / "tac_lewm_v6_epoch1"
)


def build_world_model(resources: Path = RESOURCE) -> DualEncoderJEPA:
    config = json.loads((resources / "world_model_config.json").read_text())
    config.pop("_target_")
    config.pop("encoder")
    predictor = config.pop("predictor")
    predictor.pop("_target_")
    action = config.pop("action_encoder")
    action.pop("_target_")
    components = {}
    for name in ("projector", "pred_proj"):
        spec = config.pop(name)
        spec.pop("_target_")
        spec.pop("norm_fn")
        components[name] = MLP(**spec, norm_fn=nn.BatchNorm1d)
    backbone = json.loads((resources / "backbone_config.json").read_text())
    vision = json.loads((resources / "vision_backbone_config.json").read_text())
    config.update(
        allow_hf_download=False,
        dinov3_path=None,
        tactile_stream_compile=False,
        vision_compile=False,
    )
    model = DualEncoderJEPA(
        encoder=_Backbone(backbone),
        predictor=ARPredictor(**predictor),
        action_encoder=MultiSourceActionEmbedder(**action),
        vision_encoder=DINOv3ViTModel(DINOv3ViTConfig(**vision)),
        **components,
        **config,
    )
    return model


def load_world_model(weights: Path, resources: Path = RESOURCE) -> DualEncoderJEPA:
    model = build_world_model(resources)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    # Transformers 5 moved DINOv3 layers under model.layer; embeddings/norm did not move.
    expected = model.state_dict()
    for key in list(state):
        if key.startswith("vision_encoder.layer."):
            renamed = key.replace(
                "vision_encoder.layer.", "vision_encoder.model.layer.", 1
            )
            if renamed in expected:
                if renamed in state:
                    raise ValueError(f"Duplicate DINOv3 checkpoint key: {renamed}")
                state[renamed] = state.pop(key)
    model.load_state_dict(state, strict=True)
    return model
