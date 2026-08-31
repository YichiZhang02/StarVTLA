"""Wan2.2 VAE adapter for frame-independent tactile reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from vtla.frameworks.fastwam.core.wan_video_vae import (
    CausalConv3d,
    Decoder3d_38,
    Encoder3d_38,
    count_conv3d,
    patchify as wan_patchify,
    unpatchify as wan_unpatchify,
)

from ...common.backbone import EncodedFeatures, ReconstructionBackbone, ReconstructionOutput
from ...common.checkpoint import load_filtered_state, read_state_dict


WAN22_LATENT_MEAN = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)
WAN22_LATENT_STD = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)


@dataclass
class Wan22VAEReconstructionOutput(ReconstructionOutput):
    reconstruction_loss: Tensor
    kl_loss: Tensor


class _Wan22VAECore(torch.nn.Module):
    """Wan2.2 modules with an encoder-only construction mode for policy use."""

    def __init__(
        self,
        *,
        latent_dim: int,
        base_dim: int,
        decoder_base_dim: int,
        include_decoder: bool,
    ) -> None:
        super().__init__()
        self.encoder = Encoder3d_38(dim=base_dim, z_dim=latent_dim * 2)
        self.conv1 = CausalConv3d(latent_dim * 2, latent_dim * 2, 1)
        if include_decoder:
            self.conv2 = CausalConv3d(latent_dim, latent_dim, 1)
            self.decoder = Decoder3d_38(
                dim=decoder_base_dim,
                z_dim=latent_dim,
                temperal_upsample=[True, True, False],
            )
        else:
            self.conv2 = None
            self.decoder = None
        self.clear_cache()

    def clear_cache(self) -> None:
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
        self._conv_num = 0 if self.decoder is None else count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num


class Wan22VAEBackbone(ReconstructionBackbone):
    """Fine-tunable Wan2.2 VAE with one independent VAE sample per tactile frame."""

    model_id = "wan22_vae"
    patch_size = 16
    tubelet_size = 1
    checkpoint_prefixes = (
        "vae.encoder.",
        "vae.conv1.",
        "latent_mean",
        "latent_inv_std",
    )

    def __init__(
        self,
        *,
        num_frames: int = 4,
        image_size: int = 224,
        latent_dim: int = 48,
        base_dim: int = 160,
        decoder_base_dim: int = 256,
        kl_weight: float = 1e-6,
        include_decoder: bool = True,
        pretrained_path: str = "",
    ) -> None:
        super().__init__()
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if image_size % self.patch_size:
            raise ValueError("Wan2.2 VAE image_size must be divisible by 16")
        if kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.feature_dim = int(latent_dim)
        self.grid_size = self.image_size // self.patch_size
        self.kl_weight = float(kl_weight)
        self.vae = _Wan22VAECore(
            latent_dim=int(latent_dim),
            base_dim=int(base_dim),
            decoder_base_dim=int(decoder_base_dim),
            include_decoder=bool(include_decoder),
        )
        if latent_dim == len(WAN22_LATENT_MEAN):
            mean = torch.tensor(WAN22_LATENT_MEAN)
            inv_std = torch.tensor(WAN22_LATENT_STD).reciprocal()
        else:
            # Non-standard dimensions are useful for small structural tests only.
            mean = torch.zeros(latent_dim)
            inv_std = torch.ones(latent_dim)
        self.register_buffer("latent_mean", mean.view(1, latent_dim, 1, 1, 1))
        self.register_buffer("latent_inv_std", inv_std.view(1, latent_dim, 1, 1, 1))
        self.load_report = None
        if pretrained_path:
            self.load_report = self.load_pretrained(pretrained_path)

    def _validate_images(self, images: Tensor) -> tuple[int, int, int]:
        if images.ndim != 6:
            raise ValueError(f"expected [B,S,T,C,H,W], got {tuple(images.shape)}")
        b, sensors, frames, channels, height, width = images.shape
        expected = (self.num_frames, 3, self.image_size, self.image_size)
        if (frames, channels, height, width) != expected:
            raise ValueError(
                "input shape does not match Wan2.2 VAE configuration: "
                f"expected [B,S,{self.num_frames},3,{self.image_size},{self.image_size}], "
                f"got {tuple(images.shape)}"
            )
        return b, sensors, frames

    def _posterior(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, tuple[int, int, int]]:
        b, sensors, frames = self._validate_images(images)
        target = images.reshape(b * sensors * frames, 3, self.image_size, self.image_size)
        target = target.mul(2.0).sub(1.0).unsqueeze(2)
        self.vae.clear_cache()
        self.vae._enc_conv_idx = [0]
        encoded, self.vae._enc_feat_map, self.vae._enc_conv_idx = self.vae.encoder(
            wan_patchify(target, patch_size=2),
            feat_cache=self.vae._enc_feat_map,
            feat_idx=self.vae._enc_conv_idx,
        )
        mean, logvar = self.vae.conv1(encoded).chunk(2, dim=1)
        self.vae.clear_cache()
        return mean, logvar.clamp(-30.0, 20.0), target, (b, sensors, frames)

    def _decode(self, latent: Tensor) -> Tensor:
        self.vae.clear_cache()
        self.vae._conv_idx = [0]
        decoded, self.vae._feat_map, self.vae._conv_idx = self.vae.decoder(
            self.vae.conv2(latent),
            feat_cache=self.vae._feat_map,
            feat_idx=self.vae._conv_idx,
            first_chunk=True,
        )
        decoded = wan_unpatchify(decoded, patch_size=2)
        self.vae.clear_cache()
        return decoded

    def pooled_tokens_per_sensor(self, pool_size: int) -> int:
        return 1 + self.num_frames * pool_size**2

    def forward_reconstruction(self, images: Tensor, mask_ratio: float) -> ReconstructionOutput:
        del mask_ratio
        if self.vae.conv2 is None or self.vae.decoder is None:
            raise RuntimeError("Wan2.2 VAE decoder was discarded for downstream inference")
        mean, logvar, target, (b, sensors, frames) = self._posterior(images)
        if self.training:
            latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            latent = mean
        decoded = self._decode(latent)
        reconstruction_loss = F.l1_loss(decoded, target)
        kl_loss = -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()
        loss = reconstruction_loss + self.kl_weight * kl_loss
        reconstruction = decoded.squeeze(2).add(1.0).mul(0.5).clamp(0.0, 1.0)
        reconstruction = reconstruction.reshape(
            b, sensors, frames, 3, self.image_size, self.image_size
        )
        mask = images.new_zeros((b, sensors, frames, self.grid_size**2))
        return Wan22VAEReconstructionOutput(
            loss=loss,
            reconstruction=reconstruction,
            mask=mask,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
        )

    def encode_features(self, images: Tensor) -> EncodedFeatures:
        mean, _, _, (b, sensors, frames) = self._posterior(images)
        normalized = (mean - self.latent_mean) * self.latent_inv_std
        spatial = normalized.squeeze(2).reshape(
            b,
            sensors,
            frames,
            self.feature_dim,
            self.grid_size,
            self.grid_size,
        )
        spatial = spatial.permute(0, 1, 2, 4, 5, 3)
        global_tokens = spatial.mean(dim=(2, 3, 4)).unsqueeze(2)
        return EncodedFeatures(
            global_tokens=global_tokens,
            spatial_grid=spatial,
            global_time_ids=torch.full((1,), -1, device=images.device, dtype=torch.long),
        )

    def discard_training_modules(self) -> None:
        self.vae.conv2 = None
        self.vae.decoder = None

    def patchify(self, images: Tensor) -> Tensor:
        b, sensors, frames = self._validate_images(images)
        p = self.patch_size
        grid = self.grid_size
        patches = images.reshape(b, sensors, frames, 3, grid, p, grid, p)
        patches = patches.permute(0, 1, 2, 4, 6, 5, 7, 3)
        return patches.reshape(b, sensors, frames, grid * grid, p * p * 3)

    def unpatchify(self, patches: Tensor) -> Tensor:
        if patches.ndim != 5:
            raise ValueError("Wan2.2 VAE patches must be [B,S,T,N,P]")
        b, sensors, frames, count, values = patches.shape
        if (
            frames != self.num_frames
            or count != self.grid_size**2
            or values != self.patch_size**2 * 3
        ):
            raise ValueError("Wan2.2 VAE patch shape does not match the configured image grid")
        p = self.patch_size
        grid = self.grid_size
        images = patches.reshape(b, sensors, frames, grid, grid, p, p, 3)
        images = images.permute(0, 1, 2, 7, 3, 5, 4, 6)
        return images.reshape(b, sensors, frames, 3, self.image_size, self.image_size)

    def load_pretrained(self, path: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(path)
        if checkpoint_path.suffix in {".pth", ".pt", ".bin"}:
            loaded: Any = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            for wrapper in ("state_dict", "model", "module", "model_state"):
                if isinstance(loaded, dict) and isinstance(loaded.get(wrapper), dict):
                    loaded = loaded[wrapper]
            if not isinstance(loaded, dict):
                raise TypeError(f"Checkpoint {checkpoint_path} does not contain a state dict")
            raw = {
                str(key): value for key, value in loaded.items() if isinstance(value, Tensor)
            }
        else:
            raw = read_state_dict(checkpoint_path)
        state = {}
        for key, value in raw.items():
            key = key.removeprefix("module.")
            if key.startswith("vae."):
                state[key] = value
            else:
                state["vae." + key.removeprefix("model.")] = value
        report = load_filtered_state(self, state, source=str(path))
        trainable = dict(self.named_parameters())
        required_missing = [key for key in report["missing_keys"] if key in trainable]
        required_mismatch = [key for key in report["shape_mismatch"] if key in trainable]
        if required_missing or required_mismatch:
            raise ValueError(
                "Wan2.2 VAE checkpoint is incomplete or incompatible: "
                f"missing={required_missing[:8]}, shape_mismatch={required_mismatch[:8]}"
            )
        return report
