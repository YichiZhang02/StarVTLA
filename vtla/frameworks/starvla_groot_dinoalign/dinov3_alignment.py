"""Frozen DINOv3 teacher, illumination augmentation, and feature alignment loss."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IlluminationAugment(nn.Module):
    """Geometry-preserving photometric perturbations for the Qwen student view."""

    def __init__(
        self,
        probability: float,
        brightness_range: tuple[float, float],
        contrast_range: tuple[float, float],
        gamma_range: tuple[float, float],
        shadow_probability: float,
        shadow_strength_range: tuple[float, float],
    ) -> None:
        super().__init__()
        self.probability = probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.shadow_probability = shadow_probability
        self.shadow_strength_range = shadow_strength_range

    @staticmethod
    def _sample_range(
        bounds: tuple[float, float], shape: tuple[int, ...], device: torch.device
    ) -> Tensor:
        low, high = bounds
        return torch.empty(shape, device=device, dtype=torch.float32).uniform_(low, high)

    def forward(self, images: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return images

        original_shape = images.shape
        flat = images.reshape(-1, *images.shape[-3:]).float()
        n, _, height, width = flat.shape
        apply = (torch.rand(n, 1, 1, 1, device=flat.device) < self.probability).float()

        brightness = self._sample_range(self.brightness_range, (n, 1, 1, 1), flat.device)
        flat = flat * (1.0 + apply * (brightness - 1.0))

        contrast = self._sample_range(self.contrast_range, (n, 1, 1, 1), flat.device)
        mean = flat.mean(dim=(1, 2, 3), keepdim=True)
        adjusted = (flat - mean) * contrast + mean
        flat = flat + apply * (adjusted - flat)

        gamma = self._sample_range(self.gamma_range, (n, 1, 1, 1), flat.device)
        adjusted = flat.clamp(0.0, 1.0).pow(gamma)
        flat = flat + apply * (adjusted - flat)

        if self.shadow_probability > 0.0:
            shadow_apply = apply * (
                torch.rand(n, 1, 1, 1, device=flat.device) < self.shadow_probability
            ).float()
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, height, device=flat.device),
                torch.linspace(-1.0, 1.0, width, device=flat.device),
                indexing="ij",
            )
            angle = torch.rand(n, 1, 1, device=flat.device) * (2.0 * math.pi)
            offset = torch.empty(n, 1, 1, device=flat.device).uniform_(-0.65, 0.65)
            softness = torch.empty(n, 1, 1, device=flat.device).uniform_(0.08, 0.25)
            signed_distance = (
                xx.unsqueeze(0) * torch.cos(angle) + yy.unsqueeze(0) * torch.sin(angle) - offset
            )
            shadow_mask = torch.sigmoid(signed_distance / softness).unsqueeze(1)
            strength = self._sample_range(
                self.shadow_strength_range, (n, 1, 1, 1), flat.device
            )
            flat = flat * (1.0 - shadow_apply * strength * shadow_mask)

        return flat.clamp(0.0, 1.0).reshape(original_shape).to(images.dtype)


class DinoV3Teacher(nn.Module):
    """Local-checkpoint timm DINOv3 teacher that returns pooled patch tokens."""

    def __init__(
        self,
        model_name: str,
        checkpoint: str,
        input_size: int,
        patch_size: int,
        expected_hidden_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        try:
            import timm
            from timm.models import load_checkpoint
        except ImportError as exc:
            raise ImportError(
                "starvla_groot_dinoalign requires timm with DINOv3 support."
            ) from exc

        checkpoint_path = Path(checkpoint).expanduser()
        if checkpoint_path.is_dir():
            candidates = (
                checkpoint_path / "model.safetensors",
                checkpoint_path / "pytorch_model.bin",
            )
            checkpoint_path = next((path for path in candidates if path.is_file()), checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"DINOv3 checkpoint not found: {checkpoint_path}. Set "
                "--policy.dinov3_checkpoint to a local ViT-B/16 checkpoint file or "
                "a directory containing model.safetensors/pytorch_model.bin."
            )

        model = timm.create_model(model_name, pretrained=False, num_classes=0)
        load_checkpoint(model, str(checkpoint_path), device="cpu", strict=True)

        hidden_size = int(getattr(model, "embed_dim", 0))
        if hidden_size != expected_hidden_size:
            raise ValueError(
                f"DINOv3 teacher hidden size is {hidden_size}, expected {expected_hidden_size}. "
                "Check dinov3_model_name/dinov3_hidden_size."
            )
        model_patch_size = tuple(getattr(model.patch_embed, "patch_size", ()))
        if model_patch_size != (patch_size, patch_size):
            raise ValueError(
                f"DINOv3 teacher patch size is {model_patch_size}, expected {(patch_size, patch_size)}."
            )

        self.model = model.to(device=device, dtype=dtype).eval()
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=device).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=device).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def forward(self, images: Tensor, output_grid: tuple[int, int]) -> Tensor:
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images.float(),
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        model_dtype = next(self.model.parameters()).dtype
        x = (images.float() - self.image_mean) / self.image_std
        features = self.model.forward_features(x.to(dtype=model_dtype))
        if isinstance(features, dict):
            for key in ("x_norm_patchtokens", "last_hidden_state", "x"):
                if key in features:
                    features = features[key]
                    break
            else:
                raise RuntimeError(
                    f"Unsupported DINOv3 forward_features keys: {sorted(features)}"
                )
        if not isinstance(features, Tensor) or features.dim() != 3:
            raise RuntimeError(
                "DINOv3 forward_features must return [batch, prefix+patches, hidden]."
            )

        patch_grid = self.input_size // self.patch_size
        patch_count = patch_grid * patch_grid
        if features.shape[1] < patch_count:
            raise RuntimeError(
                f"DINOv3 returned {features.shape[1]} tokens, fewer than {patch_count} patches."
            )
        # DINOv3 ViTs prepend one CLS and four register tokens. Taking the final
        # grid tokens is robust to the exact number of prefix/register tokens.
        patches = features[:, -patch_count:, :]
        patches = patches.transpose(1, 2).reshape(
            features.shape[0], self.hidden_size, patch_grid, patch_grid
        )
        patches = F.adaptive_avg_pool2d(patches.float(), output_grid)
        return patches.flatten(2).transpose(1, 2).contiguous()


class DinoAlignmentHead(nn.Module):
    """Project Qwen image tokens and compute local/global cosine distillation."""

    def __init__(self, qwen_hidden_size: int, dino_hidden_size: int) -> None:
        super().__init__()
        self.student_projection = nn.Sequential(
            nn.LayerNorm(qwen_hidden_size),
            nn.Linear(qwen_hidden_size, dino_hidden_size),
        )

    def forward(
        self,
        qwen_tokens: Tensor,
        dino_tokens: Tensor,
        global_loss_weight: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-sample total, patch, and global alignment losses."""
        if qwen_tokens.shape[:-1] != dino_tokens.shape[:-1]:
            raise ValueError(
                f"Qwen/DINO token layouts differ: {qwen_tokens.shape} vs {dino_tokens.shape}"
            )
        # Keep the small alignment head and cosine loss in fp32 even when Qwen
        # emits bf16 tokens outside an active autocast context.
        student = F.normalize(self.student_projection(qwen_tokens.float()), dim=-1)
        teacher = F.normalize(dino_tokens.detach().float(), dim=-1)

        # [B, V, T] -> [B]
        patch_loss = (1.0 - (student * teacher).sum(dim=-1)).mean(dim=(1, 2))
        student_global = F.normalize(student.mean(dim=2), dim=-1)
        teacher_global = F.normalize(teacher.mean(dim=2), dim=-1)
        global_loss = (1.0 - (student_global * teacher_global).sum(dim=-1)).mean(dim=1)
        total = (1.0 - global_loss_weight) * patch_loss + global_loss_weight * global_loss
        return total, patch_loss, global_loss
