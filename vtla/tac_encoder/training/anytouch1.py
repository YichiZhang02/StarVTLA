"""AnyTouch stage-1 masked image reconstruction recipe."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from vtla.tac_encoder.eval import save_reconstruction_visualization
from vtla.tac_encoder.models.anytouch1 import AnyTouch1Backbone

from .base import StepOutput, TrainingRecipe, require_fully_pretrained, split_state_dict


class AnyTouch1TrainingRecipe(TrainingRecipe):
    model_id = "anytouch1"
    objective = "masked_pixel_reconstruction"
    default_weight_decay = 0.1
    encoder_prefixes = (
        "model.touch_model.",
        "model.touch_projection.",
        "model.video_patch_embedding.",
        "model.video_position_embedding.",
        "model.sensor_token",
        "normalization_",
    )

    def build_model(self, args) -> nn.Module:
        model = AnyTouch1Backbone(
            num_frames=args.num_frames,
            image_size=args.image_size,
            arch=args.anytouch1_arch,
            mask_ratio=args.mask_ratio,
            pretrained_path=args.pretrained_path,
        )
        require_fully_pretrained(model, model.load_report)
        return model

    def step(self, model: nn.Module, images: Tensor, args) -> StepOutput:
        output = model(images, args.mask_ratio)
        return StepOutput(loss=output.loss, metrics={"masked_pixel_mse": output.loss.detach()})

    def encoder_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        return split_state_dict(model, self.encoder_prefixes)[0]

    def trainer_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        return split_state_dict(model, self.encoder_prefixes)[1]

    def save_visualization(
        self, model, dataset, indices, destination: Path, device, args, autocast_dtype
    ) -> None:
        save_reconstruction_visualization(
            model, dataset, indices, destination, device, args.mask_ratio, autocast_dtype
        )


RECIPE = AnyTouch1TrainingRecipe()
