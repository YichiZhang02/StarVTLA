"""AnyTouch stage-1 adapter with frame-independent ``T=4`` reconstruction."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .mae.build import build_model, load_pretrained

from ...common.backbone import EncodedFeatures, ReconstructionBackbone, ReconstructionOutput


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AnyTouch1Backbone(ReconstructionBackbone):
    model_id = "anytouch1"
    tubelet_size = 1
    checkpoint_prefixes = (
        "model.touch_model.",
        "model.touch_projection.",
        "model.video_patch_embedding.",
        "model.video_position_embedding.",
        "model.sensor_token",
        "normalization_",
    )

    def __init__(
        self,
        *,
        num_frames: int = 4,
        image_size: int = 224,
        arch: str = "vit_l",
        mask_ratio: float = 0.75,
        pretrained_path: str = "",
        visible_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size != 224:
            raise ValueError("AnyTouch1 released architecture requires image_size=224")
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.model = build_model(
            arch=arch,
            mask_ratio=mask_ratio,
            visible_loss_weight=visible_loss_weight,
        )
        missing, unexpected = load_pretrained(
            self.model, pretrained_path, verbose=bool(pretrained_path)
        )
        self.load_report = {
            "source": pretrained_path or "scratch",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        self.patch_size = int(self.model.patch_size)
        self.feature_dim = int(self.model.touch_projection.out_features)
        self.grid_size = image_size // self.patch_size
        self.register_buffer("normalization_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("normalization_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def load_pretrained(self, path: str):
        missing, unexpected = load_pretrained(self.model, path)
        self.load_report = {
            "source": path,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        return missing, unexpected

    def pooled_tokens_per_sensor(self, pool_size: int) -> int:
        return self.num_frames * (1 + pool_size**2)

    def discard_training_modules(self) -> None:
        """Remove reconstruction-only parameters after loading a downstream encoder."""
        for name in (
            "decoder_pos_embed",
            "mask_token",
            "decoder_embed",
            "touch_decoder_blocks",
            "decoder_norm",
            "decoder_pred",
            "decoder_pred_video",
        ):
            setattr(self.model, name, None)

    def _flatten(self, images: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        if images.ndim != 6:
            raise ValueError(f"expected [B,S,T,C,H,W], got {tuple(images.shape)}")
        b, sensors, frames = images.shape[:3]
        if frames != self.num_frames:
            raise ValueError(f"expected T={self.num_frames}, got {frames}")
        flat = images.reshape(b * sensors * frames, *images.shape[3:])
        return flat, (b, sensors, frames)

    def _normalize(self, flat: Tensor) -> Tensor:
        return (flat - self.normalization_mean) / self.normalization_std

    def _denormalize(self, flat: Tensor) -> Tensor:
        return flat * self.normalization_std + self.normalization_mean

    def forward_reconstruction(self, images: Tensor, mask_ratio: float) -> ReconstructionOutput:
        flat, (b, sensors, frames) = self._flatten(images)
        normalized = self._normalize(flat)
        previous = self.model.mask_ratio
        self.model.mask_ratio = float(mask_ratio)
        try:
            sensor_type = torch.full((len(flat),), -1, device=flat.device, dtype=torch.long)
            loss, prediction, mask = self.model(normalized, sensor_type=sensor_type)
        finally:
            self.model.mask_ratio = previous
        reconstruction = self._denormalize(self.model.unpatchify(prediction)).clamp(0, 1)
        reconstruction = reconstruction.reshape(b, sensors, frames, *reconstruction.shape[1:])
        mask = mask.reshape(b, sensors, frames, -1)
        return ReconstructionOutput(loss=loss, reconstruction=reconstruction, mask=mask)

    def encode_features(self, images: Tensor) -> EncodedFeatures:
        flat, (b, sensors, frames) = self._flatten(images)
        normalized = self._normalize(flat)
        previous = self.model.mask_ratio
        self.model.mask_ratio = 0.0
        try:
            sensor_type = torch.full((len(flat),), -1, device=flat.device, dtype=torch.long)
            latent, _, _ = self.model.forward_encoder(normalized, sensor_type=sensor_type)
        finally:
            self.model.mask_ratio = previous
        prefix = self.model.n_prefix
        global_tokens = latent[:, 0].reshape(b, sensors, frames, self.feature_dim)
        patches = latent[:, prefix:].reshape(
            b, sensors, frames, self.grid_size, self.grid_size, self.feature_dim
        )
        return EncodedFeatures(
            global_tokens=global_tokens,
            spatial_grid=patches,
            global_time_ids=torch.arange(frames, device=images.device),
            interleave_global=True,
        )

    def patchify(self, images: Tensor) -> Tensor:
        flat, (b, sensors, frames) = self._flatten(images)
        return self.model.patchify(flat).reshape(b, sensors, frames, -1, self.patch_size**2 * 3)

    def unpatchify(self, patches: Tensor) -> Tensor:
        if patches.ndim != 5:
            raise ValueError("AnyTouch1 patches must be [B,S,T,N,P]")
        b, sensors, frames = patches.shape[:3]
        flat = self.model.unpatchify(patches.reshape(b * sensors * frames, *patches.shape[3:]))
        return flat.reshape(b, sensors, frames, *flat.shape[1:])
