"""Central registry for tactile backbone and training implementations."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Any

from .frameworks.anytouch1 import (
    CHECKPOINT_PREFIXES as ANYTOUCH1_PREFIXES,
    AnyTouch1Backbone,
    checkpoint_kwargs as anytouch1_checkpoint_kwargs,
)
from .frameworks.anytouch2 import (
    CHECKPOINT_PREFIXES as ANYTOUCH2_PREFIXES,
    AnyTouch2Backbone,
    checkpoint_kwargs as anytouch2_checkpoint_kwargs,
)
from .common.backbone import FeatureBackbone
from .common.training import TrainingRecipe
from .frameworks.sparsh_vjepa import (
    CHECKPOINT_PREFIXES as SPARSH_VJEPA_PREFIXES,
    SparshVJEPABackbone,
    checkpoint_kwargs as sparsh_vjepa_checkpoint_kwargs,
)
from .frameworks.wan22_vae import (
    CHECKPOINT_PREFIXES as WAN22_VAE_PREFIXES,
    Wan22VAEBackbone,
    checkpoint_kwargs as wan22_vae_checkpoint_kwargs,
)


@dataclass(frozen=True)
class EncoderSpec:
    model_id: str
    backbone_class: type[FeatureBackbone]
    training_module: str
    checkpoint_prefixes: tuple[str, ...]
    checkpoint_kwargs: Callable[[Mapping[str, Any]], dict[str, Any]]

    @cached_property
    def training_recipe(self) -> TrainingRecipe:
        recipe = import_module(self.training_module).RECIPE
        if recipe.model_id != self.model_id:
            raise RuntimeError(
                f"Training recipe model_id does not match registry key {self.model_id!r}"
            )
        return recipe


ENCODER_REGISTRY = {
    "anytouch1": EncoderSpec(
        model_id="anytouch1",
        backbone_class=AnyTouch1Backbone,
        training_module="vtla.tac_encoder.frameworks.anytouch1.training",
        checkpoint_prefixes=ANYTOUCH1_PREFIXES,
        checkpoint_kwargs=anytouch1_checkpoint_kwargs,
    ),
    "anytouch2": EncoderSpec(
        model_id="anytouch2",
        backbone_class=AnyTouch2Backbone,
        training_module="vtla.tac_encoder.frameworks.anytouch2.training",
        checkpoint_prefixes=ANYTOUCH2_PREFIXES,
        checkpoint_kwargs=anytouch2_checkpoint_kwargs,
    ),
    "sparsh_vjepa": EncoderSpec(
        model_id="sparsh_vjepa",
        backbone_class=SparshVJEPABackbone,
        training_module="vtla.tac_encoder.frameworks.sparsh_vjepa.training",
        checkpoint_prefixes=SPARSH_VJEPA_PREFIXES,
        checkpoint_kwargs=sparsh_vjepa_checkpoint_kwargs,
    ),
    "wan22_vae": EncoderSpec(
        model_id="wan22_vae",
        backbone_class=Wan22VAEBackbone,
        training_module="vtla.tac_encoder.frameworks.wan22_vae.training",
        checkpoint_prefixes=WAN22_VAE_PREFIXES,
        checkpoint_kwargs=wan22_vae_checkpoint_kwargs,
    ),
}


def supported_model_ids() -> tuple[str, ...]:
    return tuple(ENCODER_REGISTRY)


def get_encoder_spec(model_id: str) -> EncoderSpec:
    try:
        spec = ENCODER_REGISTRY[model_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown model_id {model_id!r}; expected one of {supported_model_ids()}"
        ) from error
    if spec.backbone_class.model_id != spec.model_id:
        raise RuntimeError(f"Backbone model_id does not match registry key {model_id!r}")
    return spec


def build_backbone(model_id: str, **kwargs: Any) -> FeatureBackbone:
    return get_encoder_spec(model_id).backbone_class(**kwargs)


def get_training_recipe(model_id: str) -> TrainingRecipe:
    return get_encoder_spec(model_id).training_recipe


def load_backbone_checkpoint(model: FeatureBackbone, path: str | Path) -> dict[str, Any]:
    if not hasattr(model, "load_pretrained"):
        raise TypeError(f"{type(model).__name__} does not support external checkpoints")
    report = model.load_pretrained(path)
    if isinstance(report, tuple):
        missing, unexpected = report
        return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    return report


# Compatibility mappings for callers that inspected the old registries directly.
MODEL_REGISTRY = {
    model_id: spec.backbone_class for model_id, spec in ENCODER_REGISTRY.items()
}


class _TrainingRecipeRegistry(Mapping[str, TrainingRecipe]):
    def __getitem__(self, model_id: str) -> TrainingRecipe:
        return ENCODER_REGISTRY[model_id].training_recipe

    def __iter__(self) -> Iterator[str]:
        return iter(ENCODER_REGISTRY)

    def __len__(self) -> int:
        return len(ENCODER_REGISTRY)


TRAINING_RECIPES = _TrainingRecipeRegistry()
