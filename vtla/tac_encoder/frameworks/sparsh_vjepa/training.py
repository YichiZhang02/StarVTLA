"""Sparsh V-JEPA latent-prediction recipe matching the reference objective."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch import Tensor, nn

from ...common.checkpoint import read_state_dict
from ...common.training import StepOutput, TrainingRecipe, WarmupCosineScheduler
from .model import (
    _SinusoidalPositionEmbedding,
    _SparshEncoder,
)


MASK_CONFIGS = (
    dict(aspect_ratio=(0.75, 1.5), num_blocks=8, spatial_scale=(0.15, 0.15)),
    dict(aspect_ratio=(0.75, 1.5), num_blocks=2, spatial_scale=(0.70, 0.70)),
)


def _gather(tokens: Tensor, indices: Tensor) -> Tensor:
    return torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


class _EncoderWrapper(nn.Module):
    def __init__(self, backbone: _SparshEncoder) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, video: Tensor, mask: Tensor | None = None) -> Tensor:
        tokens = self.backbone.add_position_embedding(self.backbone.patch_embed(video))
        if mask is not None:
            tokens = _gather(tokens, mask)
        return self.backbone.forward_tokens(tokens)


class _PredictorBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        temporal_units: int,
        grid_size: int,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_mask_tokens: int = 2,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, input_dim)
        shared_mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.mask_token = nn.ParameterList([shared_mask_token] * num_mask_tokens)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=True,
                    init_values=1.0,
                    norm_layer=lambda dim, **kwargs: nn.LayerNorm(dim, eps=1e-6, **kwargs),
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.pos_embed = _SinusoidalPositionEmbedding(
            embed_dim,
            source_grid=(temporal_units, grid_size, grid_size),
            target_grid=(temporal_units, grid_size, grid_size),
        )

    def forward(
        self,
        context: Tensor,
        context_mask: Tensor,
        target_mask: Tensor,
        mask_index: int,
    ) -> Tensor:
        context = self.input_projection(context)
        position = self.pos_embed(device=context.device, dtype=context.dtype)[0]
        context = context + position[context_mask]
        target_position = position[target_mask]
        mask_token = self.mask_token[mask_index % len(self.mask_token)]
        targets = mask_token.view(1, 1, -1) + target_position
        hidden = torch.cat([context, targets], dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden[:, context.shape[1] :])
        return self.output_projection(hidden)


class _PredictorWrapper(nn.Module):
    def __init__(self, backbone: _PredictorBackbone) -> None:
        super().__init__()
        self.backbone = backbone


class _BlockMaskGenerator:
    def __init__(
        self,
        *,
        temporal_units: int,
        grid_size: int,
        spatial_scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
        num_blocks: int,
    ) -> None:
        self.temporal_units = temporal_units
        self.grid_size = grid_size
        self.spatial_scale = spatial_scale
        self.aspect_ratio = aspect_ratio
        self.num_blocks = num_blocks
        self.step = -1

    def _block_size(self, generator: torch.Generator) -> tuple[int, int]:
        scale = self.spatial_scale[0] + torch.rand(1, generator=generator).item() * (
            self.spatial_scale[1] - self.spatial_scale[0]
        )
        area = int(self.grid_size * self.grid_size * scale)
        ratio = self.aspect_ratio[0] + torch.rand(1, generator=generator).item() * (
            self.aspect_ratio[1] - self.aspect_ratio[0]
        )
        height = min(self.grid_size, int(round(math.sqrt(area * ratio))))
        width = min(self.grid_size, int(round(math.sqrt(area / ratio))))
        return max(1, height), max(1, width)

    def __call__(self, batch_size: int) -> tuple[Tensor, Tensor]:
        self.step += 1
        generator = torch.Generator().manual_seed(self.step)
        height, width = self._block_size(generator)
        contexts, targets = [], []
        min_context = min_target = self.temporal_units * self.grid_size**2
        for _ in range(batch_size):
            while True:
                keep = torch.ones(
                    self.temporal_units, self.grid_size, self.grid_size, dtype=torch.bool
                )
                for _ in range(self.num_blocks):
                    top = int(torch.randint(0, self.grid_size - height + 1, (1,)).item())
                    left = int(torch.randint(0, self.grid_size - width + 1, (1,)).item())
                    keep[:, top : top + height, left : left + width] = False
                context = torch.nonzero(keep.flatten()).flatten()
                target = torch.nonzero(~keep.flatten()).flatten()
                if len(context) and len(target):
                    break
            min_context = min(min_context, len(context))
            min_target = min(min_target, len(target))
            contexts.append(context)
            targets.append(target)
        return (
            torch.stack([value[:min_context] for value in contexts]),
            torch.stack([value[:min_target] for value in targets]),
        )


@dataclass
class VJEPAOutput:
    loss: Tensor
    jepa_loss: Tensor
    reg_loss: Tensor
    predictions: list[Tensor]
    targets: list[Tensor]
    context_masks: list[Tensor]
    target_masks: list[Tensor]


class SparshVJEPATrainingModel(nn.Module):
    def __init__(
        self,
        *,
        num_frames: int,
        image_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        predictor_dim: int = 384,
        predictor_depth: int = 12,
        predictor_heads: int = 6,
        checkpoint_source_grid: tuple[int, int, int] = (2, 20, 15),
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = 16
        self.tubelet_size = 2
        self.feature_dim = embed_dim
        self.temporal_units = num_frames // self.tubelet_size
        self.grid_size = image_size // self.patch_size
        self.num_patches = self.temporal_units * self.grid_size**2
        encoder = _SparshEncoder(
            num_frames=num_frames,
            tubelet_size=self.tubelet_size,
            image_size=image_size,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            position_source_grid=checkpoint_source_grid,
        )
        self.context_encoder = _EncoderWrapper(encoder)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = _PredictorWrapper(
            _PredictorBackbone(
                input_dim=embed_dim,
                temporal_units=self.temporal_units,
                grid_size=self.grid_size,
                embed_dim=predictor_dim,
                depth=predictor_depth,
                num_heads=predictor_heads,
            )
        )
        self.mask_generators = [
            _BlockMaskGenerator(
                temporal_units=self.temporal_units,
                grid_size=self.grid_size,
                **config,
            )
            for config in MASK_CONFIGS
        ]

    def _flatten(self, images: Tensor) -> Tensor:
        b, sensors, frames, channels, height, width = images.shape
        if (frames, channels, height, width) != (
            self.num_frames,
            3,
            self.image_size,
            self.image_size,
        ):
            raise ValueError("input shape does not match Sparsh V-JEPA configuration")
        return images.reshape(b * sensors, frames, channels, height, width).permute(0, 2, 1, 3, 4)

    def forward(self, images: Tensor) -> VJEPAOutput:
        video = self._flatten(images)
        mask_pairs = [generator(len(video)) for generator in self.mask_generators]
        context_masks = [pair[0].to(video.device) for pair in mask_pairs]
        target_masks = [pair[1].to(video.device) for pair in mask_pairs]
        predictions, targets = [], []
        with torch.no_grad():
            target_full = F.layer_norm(
                self.target_encoder(video), (self.feature_dim,)
            )
        for index, (context_mask, target_mask) in enumerate(zip(context_masks, target_masks)):
            context = self.context_encoder(video, context_mask)
            predictions.append(
                self.predictor.backbone(context, context_mask, target_mask, index)
            )
            targets.append(_gather(target_full, target_mask))
        jepa_loss = sum((pred - target).abs().mean() for pred, target in zip(predictions, targets))
        jepa_loss = jepa_loss / len(predictions)
        patch_std = sum(
            torch.sqrt(prediction.var(dim=1) + 1e-4) for prediction in predictions
        ) / len(predictions)
        reg_loss = F.relu(1.0 - patch_std).mean()
        return VJEPAOutput(
            loss=jepa_loss + reg_loss,
            jepa_loss=jepa_loss,
            reg_loss=reg_loss,
            predictions=predictions,
            targets=targets,
            context_masks=context_masks,
            target_masks=target_masks,
        )


def _canonical_training_key(key: str) -> str:
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def _encoder_only_state(
    raw: dict[str, Tensor],
    expected: dict[str, Tensor],
) -> dict[str, Tensor]:
    prefixes = (
        "target_encoder.backbone.",
        "context_encoder.backbone.",
        "target_encoder.",
        "context_encoder.",
        "encoder.",
        "backbone.",
    )
    prefix = next(
        (candidate for candidate in prefixes if any(key.startswith(candidate) for key in raw)),
        "",
    )
    candidates = {
        key.removeprefix(prefix): value
        for key, value in raw.items()
        if not prefix or key.startswith(prefix)
    }
    compatible = {
        key: value
        for key, value in candidates.items()
        if key in expected and expected[key].shape == value.shape
    }
    missing = [key for key in expected if key not in compatible]
    mismatched = [
        key
        for key, value in candidates.items()
        if key in expected and expected[key].shape != value.shape
    ]
    if missing or mismatched:
        raise ValueError(
            "Sparsh V-JEPA encoder checkpoint is incomplete or incompatible: "
            f"missing={missing[:12]}, shape_mismatch={mismatched[:12]}"
        )
    return compatible


def _load_full_pretrained(model: SparshVJEPATrainingModel, path: str) -> dict:
    """Load full JEPA state or seed both encoders from an encoder-only checkpoint."""
    if not path:
        raise ValueError("Sparsh V-JEPA training requires a pretrained encoder checkpoint")
    raw = {_canonical_training_key(key): value for key, value in read_state_dict(path).items()}
    has_context = any(key.startswith("context_encoder.") for key in raw)
    has_target = any(key.startswith("target_encoder.") for key in raw)
    has_predictor = any(key.startswith("predictor.") for key in raw)
    model_state = model.state_dict()
    if has_context and has_target:
        compatible = {
            key: value
            for key, value in raw.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        missing, _ = model.load_state_dict(compatible, strict=False)
        allowed_missing = (
            [key for key in model_state if key.startswith("predictor.")]
            if not has_predictor
            else []
        )
        required_missing = [key for key in missing if key not in allowed_missing]
        if required_missing or (has_predictor and missing):
            raise ValueError(
                "Full V-JEPA checkpoint does not cover the required model state: "
                f"{list(missing)[:12]}"
            )
        return {
            "source": path,
            "initialization": "full" if has_predictor else "encoders_only",
            "predictor_init": "checkpoint" if has_predictor else "scratch",
            "loaded_tensors": len(compatible),
            "missing_keys": list(missing),
        }

    expected_encoder = model.context_encoder.backbone.state_dict()
    encoder = _encoder_only_state(raw, expected_encoder)
    state = {
        **{f"context_encoder.backbone.{key}": value for key, value in encoder.items()},
        **{f"target_encoder.backbone.{key}": value for key, value in encoder.items()},
    }
    missing, _ = model.load_state_dict(state, strict=False)
    unexpected_missing = [key for key in missing if not key.startswith("predictor.")]
    if unexpected_missing:
        raise ValueError(
            "Encoder-only V-JEPA checkpoint did not initialize both encoders: "
            f"{unexpected_missing[:12]}"
        )
    return {
        "source": path,
        "initialization": "encoder_only",
        "predictor_init": "scratch",
        "loaded_tensors": len(state),
        "missing_keys": list(missing),
    }


class SparshVJEPATrainingRecipe(TrainingRecipe):
    model_id = "sparsh_vjepa"
    objective = "vjepa_masked_latent_prediction"
    default_weight_decay = 0.04
    default_warmup_epochs = 40
    default_min_lr = 1e-6

    def build_model(self, args) -> nn.Module:
        model = SparshVJEPATrainingModel(
            num_frames=args.num_frames,
            image_size=args.image_size,
            embed_dim=args.encoder_dim or 384,
            depth=args.encoder_depth or 12,
            num_heads=args.encoder_heads or 6,
            predictor_dim=args.decoder_dim or 384,
            predictor_depth=args.decoder_depth or 12,
            predictor_heads=args.decoder_heads or 6,
        )
        model.load_report = _load_full_pretrained(model, args.pretrained_path)
        return model

    def scheduler(self, optimizer, args, steps_per_epoch) -> WarmupCosineScheduler:
        # Sparsh's official V-JEPA config uses 40 warmup epochs and a 1e-6 floor.
        warmup_epochs = (
            self.default_warmup_epochs if args.warmup_epochs is None else args.warmup_epochs
        )
        final_lr = self.default_min_lr if args.min_lr is None else args.min_lr
        return WarmupCosineScheduler(
            optimizer,
            total_steps=max(1, args.epochs * steps_per_epoch),
            warmup_steps=max(0, warmup_epochs * steps_per_epoch),
            start_lr=min(2e-4, args.lr),
            final_lr=final_lr,
            final_weight_decay=0.4,
        )

    def step(self, model: nn.Module, images: Tensor, args) -> StepOutput:
        del args
        output = model(images)
        return StepOutput(
            loss=output.loss,
            metrics={
                "loss": output.loss.detach(),
                "jepa_l1": output.jepa_loss.detach(),
                "variance_regularization": output.reg_loss.detach(),
            },
        )

    def after_optimizer_step(self, model: nn.Module, step: int, total_steps: int) -> None:
        momentum = 0.998 + (1.0 - 0.998) * min(step, total_steps) / max(total_steps, 1)
        with torch.no_grad():
            for target, context in zip(
                model.target_encoder.parameters(), model.context_encoder.parameters()
            ):
                target.data.mul_(momentum).add_(context.detach().data, alpha=1.0 - momentum)

    def encoder_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        return {
            "encoder." + key.removeprefix("backbone."): value
            for key, value in model.target_encoder.state_dict().items()
        }

    def trainer_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        return {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("target_encoder.")
        }

    def restore_state(self, model, encoder_state, trainer_state) -> None:
        target = {
            "target_encoder.backbone." + key.removeprefix("encoder."): value
            for key, value in encoder_state.items()
        }
        model.load_state_dict({**trainer_state, **target}, strict=True)

    @torch.no_grad()
    def save_visualization(
        self, model, dataset, indices, destination: Path, device, args, autocast_dtype
    ) -> None:
        if not indices:
            return
        images = torch.stack([dataset[index]["images"] for index in indices]).to(device)
        model.eval()
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            output = model(images)
        prediction, target = output.predictions[0].float(), output.targets[0].float()
        error = (prediction - target).abs().mean(dim=-1)
        prediction = prediction.abs().mean(dim=-1)
        target = target.abs().mean(dim=-1)
        target_mask = output.target_masks[0]
        rows = prediction.shape[0] * model.temporal_units
        figure, axes = plt.subplots(rows, 3, figsize=(9, max(2.0, rows * 2.2)), squeeze=False)
        row = 0
        for sample in range(prediction.shape[0]):
            for time in range(model.temporal_units):
                maps = []
                for values in (prediction, target, error):
                    canvas = torch.full(
                        (model.num_patches,), float("nan"), device=values.device
                    )
                    canvas[target_mask[sample]] = values[sample]
                    maps.append(canvas.reshape(model.temporal_units, model.grid_size, model.grid_size)[time])
                for column, values in enumerate(maps):
                    axes[row, column].imshow(values.cpu(), cmap="viridis")
                    axes[row, column].axis("off")
                row += 1
        for column, title in enumerate(("pred latent", "target latent", "error")):
            axes[0, column].set_title(title)
        figure.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=120)
        plt.close(figure)


RECIPE = SparshVJEPATrainingRecipe()
