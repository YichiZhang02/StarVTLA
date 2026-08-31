"""AnyTouch2 joint pixel and temporal-residual reconstruction recipe."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from vtla.tac_encoder.eval import save_reconstruction_visualization

from ...common.training import (
    StepOutput,
    TrainingRecipe,
    require_fully_pretrained,
    split_state_dict,
)
from .model import AnyTouch2Backbone


class AnyTouch2TrainingRecipe(TrainingRecipe):
    model_id = "anytouch2"
    objective = "masked_pixel_and_temporal_residual_reconstruction"
    encoder_prefixes = AnyTouch2Backbone.checkpoint_prefixes

    def build_model(self, args) -> nn.Module:
        kwargs = {
            "num_frames": args.num_frames,
            "image_size": args.image_size,
            "pretrained_path": args.pretrained_path,
        }
        for key, name in (
            ("embed_dim", "encoder_dim"),
            ("projection_dim", "projection_dim"),
            ("depth", "encoder_depth"),
            ("num_heads", "encoder_heads"),
            ("decoder_dim", "decoder_dim"),
            ("decoder_depth", "decoder_depth"),
            ("decoder_heads", "decoder_heads"),
        ):
            value = getattr(args, name)
            if value is not None:
                kwargs[key] = value
        model = AnyTouch2Backbone(**kwargs)
        require_fully_pretrained(model, model.load_report)
        return model

    def step(self, model: nn.Module, images: Tensor, args) -> StepOutput:
        output = model(images, args.mask_ratio)
        return StepOutput(
            loss=output.loss,
            metrics={
                "loss": output.loss.detach(),
                "pixel_mse": output.pixel_loss.detach(),
                "residual_mse": output.residual_loss.detach(),
            },
        )

    def encoder_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        return split_state_dict(model, self.encoder_prefixes)[0]

    def trainer_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        return split_state_dict(model, self.encoder_prefixes)[1]

    @torch.no_grad()
    def save_visualization(
        self, model, dataset, indices, destination: Path, device, args, autocast_dtype
    ) -> None:
        save_reconstruction_visualization(
            model, dataset, indices, destination, device, args.mask_ratio, autocast_dtype
        )
        if not indices:
            return
        images = torch.stack([dataset[index]["images"] for index in indices]).to(device)
        model.eval()
        with torch.random.fork_rng(
            devices=[device.index or 0] if device.type == "cuda" else []
        ):
            torch.manual_seed(0)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                output = model.forward_reconstruction(images, args.mask_ratio)
        prediction = output.residual_prediction.float()
        target = output.residual_target.float()
        error = (prediction - target).abs().mean(dim=3, keepdim=True).expand_as(target)
        scale = torch.cat([prediction.abs().flatten(), target.abs().flatten()]).quantile(0.99).clamp_min(1e-6)
        display = (prediction / (2 * scale) + 0.5, target / (2 * scale) + 0.5, error / scale)
        rows = target.shape[0] * target.shape[1] * target.shape[2]
        figure, axes = plt.subplots(rows, 3, figsize=(9, max(2.0, rows * 2.2)), squeeze=False)
        row = 0
        for sample in range(target.shape[0]):
            for sensor in range(target.shape[1]):
                for time in range(target.shape[2]):
                    for column, values in enumerate(display):
                        axes[row, column].imshow(
                            values[sample, sensor, time].permute(1, 2, 0).cpu().clamp(0, 1)
                        )
                        axes[row, column].axis("off")
                    row += 1
        for column, title in enumerate(("pred residual", "target residual", "error")):
            axes[0, column].set_title(title)
        figure.tight_layout()
        residual_path = destination.with_name(destination.stem + "_residual" + destination.suffix)
        residual_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(residual_path, dpi=120)
        plt.close(figure)


RECIPE = AnyTouch2TrainingRecipe()
