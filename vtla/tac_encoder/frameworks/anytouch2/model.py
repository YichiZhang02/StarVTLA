"""AnyTouch2 video encoder and its released temporal pixel decoder."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from transformers import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPVisionTransformer

from ...common.backbone import EncodedFeatures, ReconstructionBackbone, ReconstructionOutput
from ...common.checkpoint import (
    interpolate_video_position_embedding,
    load_filtered_state,
    read_state_dict,
)
from .patching import patchify_video, unpatchify_video


ANYTOUCH2_MEAN = (0.48145466, 0.4578275, 0.40821073)
ANYTOUCH2_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class AnyTouch2ReconstructionOutput(ReconstructionOutput):
    pixel_loss: Tensor
    residual_loss: Tensor
    residual_prediction: Tensor
    residual_target: Tensor


def _fixed_2d_sincos_positions(embed_dim: int, grid_size: int) -> Tensor:
    if embed_dim % 4:
        raise ValueError("decoder dimension must be divisible by 4 for 2D sin-cos positions")
    positions = torch.arange(grid_size, dtype=torch.float32)
    grid_w = positions.repeat(grid_size)
    grid_h = positions.repeat_interleave(grid_size)
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / (10000**omega)

    def encode(values: Tensor) -> Tensor:
        angles = values.unsqueeze(1) * omega.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=1)

    spatial = torch.cat([encode(grid_w), encode(grid_h)], dim=1)
    return torch.cat([torch.zeros(1, embed_dim), spatial], dim=0).unsqueeze(0)


class AnyTouch2Backbone(ReconstructionBackbone):
    model_id = "anytouch2"
    checkpoint_prefixes = (
        "touch_model.",
        "touch_projection.",
        "sensor_token",
        "normalization_",
    )

    def __init__(
        self,
        *,
        num_frames: int = 4,
        image_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        embed_dim: int = 768,
        projection_dim: int = 512,
        depth: int = 12,
        num_heads: int = 12,
        decoder_dim: int = 512,
        decoder_depth: int = 6,
        decoder_heads: int = 8,
        pretrained_path: str = "",
        checkpoint_source_grid: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        if num_frames % tubelet_size or image_size % patch_size:
            raise ValueError("num_frames/image_size must be divisible by tubelet/patch size")
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.feature_dim = int(projection_dim)
        self.temporal_units = self.num_frames // self.tubelet_size
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.temporal_units * self.grid_size**2

        config = CLIPVisionConfig(
            hidden_size=embed_dim,
            intermediate_size=embed_dim * 4,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=3,
            projection_dim=projection_dim,
            hidden_act="gelu",
            layer_norm_eps=1e-5,
            attn_implementation="eager",
        )
        config._attn_implementation = "eager"
        self.touch_model = CLIPVisionTransformer(config)
        self.touch_model.embeddings.patch_embedding = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            bias=False,
        )
        self.touch_model.embeddings.position_embedding = nn.Embedding(self.num_patches + 1, embed_dim)
        self.touch_projection = nn.Linear(embed_dim, projection_dim, bias=False)
        self.sensor_token = nn.Parameter(torch.zeros(20, 5, embed_dim))
        nn.init.normal_(self.sensor_token, std=0.02)
        decoder_config = CLIPVisionConfig(
            hidden_size=decoder_dim,
            intermediate_size=decoder_dim * 4,
            num_hidden_layers=decoder_depth,
            num_attention_heads=decoder_heads,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=3,
            hidden_act="gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            attn_implementation="eager",
        )
        decoder_config._attn_implementation = "eager"

        # Keep the released AnyTouch2 names at model root so checkpoint keys load directly.
        self.decoder_embed = nn.Linear(projection_dim, decoder_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim), requires_grad=False
        )
        self.touch_decoder_blocks = nn.ModuleList(
            [CLIPEncoderLayer(decoder_config) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=decoder_config.layer_norm_eps)
        self.decoder_pred_video = nn.Linear(
            decoder_dim, tubelet_size * patch_size**2 * 3, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.decoder_embed_diff = nn.Linear(projection_dim, decoder_dim, bias=True)
        self.diff_touch_decoder_blocks = nn.ModuleList(
            [CLIPEncoderLayer(decoder_config) for _ in range(decoder_depth)]
        )
        self.diff_decoder_norm = nn.LayerNorm(decoder_dim, eps=decoder_config.layer_norm_eps)
        self.diff_decoder_pred_video = nn.Linear(
            decoder_dim, tubelet_size * patch_size**2 * 3, bias=True
        )
        self.mask_token_diff = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self._initialize_decoder_positions()
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.mask_token_diff, std=0.02)
        self.register_buffer(
            "normalization_mean", torch.tensor(ANYTOUCH2_MEAN).view(1, 3, 1, 1, 1)
        )
        self.register_buffer(
            "normalization_std", torch.tensor(ANYTOUCH2_STD).view(1, 3, 1, 1, 1)
        )
        self.load_report = None
        if pretrained_path:
            self.load_report = self.load_pretrained(pretrained_path, checkpoint_source_grid)

    def pooled_tokens_per_sensor(self, pool_size: int) -> int:
        return 1 + self.temporal_units * pool_size**2

    def _flatten(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        if images.ndim != 6:
            raise ValueError(f"expected [B,S,T,C,H,W], got {tuple(images.shape)}")
        b, sensors, frames, channels, height, width = images.shape
        if (frames, channels, height, width) != (
            self.num_frames,
            3,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "input shape does not match AnyTouch2 configuration: "
                f"expected [B,S,{self.num_frames},3,{self.image_size},{self.image_size}], "
                f"got {tuple(images.shape)}"
            )
        video = images.reshape(b * sensors, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return video, (b, sensors)

    def _normalize(self, video: Tensor) -> Tensor:
        return (video - self.normalization_mean) / self.normalization_std

    def _denormalize(self, video: Tensor) -> Tensor:
        return video * self.normalization_std + self.normalization_mean

    def _patch_embeddings(self, video: Tensor) -> Tensor:
        patches = self.touch_model.embeddings.patch_embedding(video)
        return patches.flatten(2).transpose(1, 2)

    def _initialize_decoder_positions(self) -> None:
        spatial = _fixed_2d_sincos_positions(self.decoder_pos_embed.shape[-1], self.grid_size)
        temporal = torch.cat(
            [spatial[:, :1], *[spatial[:, 1:] for _ in range(self.temporal_units)]], dim=1
        )
        self.decoder_pos_embed.data.copy_(temporal)

    def _random_masking(self, tokens: Tensor, mask_ratio: float) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the same spatial mask to every tube, matching AnyTouch2 pretraining."""
        if not 0 <= mask_ratio < 1:
            raise ValueError("mask_ratio must be in [0, 1)")
        batch, length, dim = tokens.shape
        spatial_tokens = self.grid_size**2
        if length != self.temporal_units * spatial_tokens:
            raise ValueError(f"expected {self.num_patches} patch tokens, got {length}")

        keep = max(1, int(spatial_tokens * (1 - mask_ratio)))
        noise = torch.rand(batch, spatial_tokens, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore_spatial = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        offsets = torch.arange(self.temporal_units, device=tokens.device) * spatial_tokens
        ids_keep_full = (ids_keep[:, None, :] + offsets[None, :, None]).reshape(batch, -1)
        visible = torch.gather(tokens, 1, ids_keep_full.unsqueeze(-1).expand(-1, -1, dim))

        spatial_mask = torch.ones(batch, spatial_tokens, device=tokens.device)
        spatial_mask[:, :keep] = 0
        spatial_mask = torch.gather(spatial_mask, 1, ids_restore_spatial)
        mask = spatial_mask.repeat(1, self.temporal_units)
        ids_restore = (
            ids_restore_spatial[:, None, :] + offsets[None, :, None]
        ).reshape(batch, -1)
        return visible, mask, ids_restore

    def _encode(self, patch_tokens: Tensor, *, mask_ratio: float | None) -> tuple[Tensor, Tensor, Tensor]:
        positions = self.touch_model.embeddings.position_embedding.weight
        tokens = patch_tokens + positions[None, 1:]
        if mask_ratio is None:
            visible = tokens
            mask = torch.zeros(tokens.shape[:2], device=tokens.device)
            ids_restore = torch.arange(tokens.shape[1], device=tokens.device).expand(tokens.shape[0], -1)
        else:
            visible, mask, ids_restore = self._random_masking(tokens, mask_ratio)
        class_token = self.touch_model.embeddings.class_embedding + positions[0]
        class_token = class_token.view(1, 1, -1).expand(tokens.shape[0], -1, -1)
        sensor_tokens = self.sensor_token[-1].unsqueeze(0).expand(tokens.shape[0], -1, -1)
        hidden = torch.cat([class_token, sensor_tokens, visible], dim=1)
        hidden = self.touch_model.pre_layrnorm(hidden)
        hidden = self.touch_model.encoder(inputs_embeds=hidden, attention_mask=None).last_hidden_state
        return hidden, mask, ids_restore

    def _decode_branch(
        self,
        hidden: Tensor,
        ids_restore: Tensor,
        *,
        decoder_embed: nn.Linear,
        mask_token: Tensor,
        blocks: nn.ModuleList,
        norm: nn.LayerNorm,
        prediction_head: nn.Linear,
    ) -> Tensor:
        hidden = decoder_embed(self.touch_projection(hidden))
        prefix = hidden[:, :6]
        visible = hidden[:, 6:]
        visible_per_tube = visible.shape[1] // self.temporal_units
        missing_per_tube = self.grid_size**2 - visible_per_tube
        if missing_per_tube < 0 or visible_per_tube * self.temporal_units != visible.shape[1]:
            raise ValueError("visible AnyTouch2 tokens cannot be split across temporal tubes")

        visible = visible.reshape(
            hidden.shape[0], self.temporal_units, visible_per_tube, hidden.shape[-1]
        )
        mask_tokens = mask_token.expand(
            hidden.shape[0], self.temporal_units, missing_per_tube, -1
        )
        restored = torch.cat([visible, mask_tokens], dim=2).reshape(
            hidden.shape[0], self.num_patches, hidden.shape[-1]
        )
        restored = torch.gather(
            restored,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, restored.shape[-1]),
        )

        decoded = torch.cat(
            [
                prefix[:, :1] + self.decoder_pos_embed[:, :1],
                prefix[:, 1:6],
                restored + self.decoder_pos_embed[:, 1:],
            ],
            dim=1,
        )
        for block in blocks:
            decoded = block(decoded, attention_mask=None)
        decoded = prediction_head(norm(decoded))
        return decoded[:, 6:]

    def _decode(self, hidden: Tensor, ids_restore: Tensor) -> Tensor:
        return self._decode_branch(
            hidden,
            ids_restore,
            decoder_embed=self.decoder_embed,
            mask_token=self.mask_token,
            blocks=self.touch_decoder_blocks,
            norm=self.decoder_norm,
            prediction_head=self.decoder_pred_video,
        )

    def _decode_residual(self, hidden: Tensor, ids_restore: Tensor) -> Tensor:
        return self._decode_branch(
            hidden,
            ids_restore,
            decoder_embed=self.decoder_embed_diff,
            mask_token=self.mask_token_diff,
            blocks=self.diff_touch_decoder_blocks,
            norm=self.diff_decoder_norm,
            prediction_head=self.diff_decoder_pred_video,
        )

    def _spatial_patchify(self, video: Tensor) -> Tensor:
        b, channels, frames, height, width = video.shape
        p = self.patch_size
        grid_h, grid_w = height // p, width // p
        patches = video.reshape(b, channels, frames, grid_h, p, grid_w, p)
        patches = torch.einsum("bcthpwq->bhwtpqc", patches)
        return patches.reshape(b, grid_h * grid_w, frames, p * p * channels)

    def forward_reconstruction(self, images: Tensor, mask_ratio: float) -> ReconstructionOutput:
        video, (b, sensors) = self._flatten(images)
        normalized = self._normalize(video)
        hidden, mask, ids_restore = self._encode(self._patch_embeddings(normalized), mask_ratio=mask_ratio)
        prediction = self._decode(hidden, ids_restore)
        residual_prediction_patches = self._decode_residual(hidden, ids_restore)
        target = patchify_video(normalized, self.patch_size, self.tubelet_size)
        per_patch = (prediction - target).square().mean(dim=-1)
        pixel_loss = (per_patch * mask).sum() / mask.sum().clamp_min(1)
        reconstruction = unpatchify_video(
            prediction,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
            temporal_units=self.temporal_units,
            grid_height=self.grid_size,
            grid_width=self.grid_size,
        )
        reconstruction = self._denormalize(reconstruction).clamp(0, 1)

        residual_video = unpatchify_video(
            residual_prediction_patches,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
            temporal_units=self.temporal_units,
            grid_height=self.grid_size,
            grid_width=self.grid_size,
        )
        residual_prediction = residual_video[:, :, 1:]
        residual_target = normalized[:, :, 1:] - normalized[:, :, :-1]
        residual_per_patch = (
            self._spatial_patchify(residual_prediction)
            - self._spatial_patchify(residual_target)
        ).square().mean(dim=-1).mean(dim=-1)
        spatial_mask = mask[:, : self.grid_size**2]
        residual_loss = (
            (residual_per_patch * spatial_mask).sum()
            / spatial_mask.sum().clamp_min(1)
        )
        loss = pixel_loss + residual_loss
        reconstruction = reconstruction.permute(0, 2, 1, 3, 4).reshape(
            b, sensors, self.num_frames, 3, self.image_size, self.image_size
        )
        mask = mask.reshape(b, sensors, self.temporal_units, self.grid_size, self.grid_size)
        return AnyTouch2ReconstructionOutput(
            loss=loss,
            reconstruction=reconstruction,
            mask=mask,
            pixel_loss=pixel_loss,
            residual_loss=residual_loss,
            residual_prediction=residual_prediction.permute(0, 2, 1, 3, 4).reshape(
                b, sensors, self.num_frames - 1, 3, self.image_size, self.image_size
            ),
            residual_target=residual_target.permute(0, 2, 1, 3, 4).reshape(
                b, sensors, self.num_frames - 1, 3, self.image_size, self.image_size
            ),
        )

    def encode_features(self, images: Tensor) -> EncodedFeatures:
        video, (b, sensors) = self._flatten(images)
        hidden, _, _ = self._encode(self._patch_embeddings(self._normalize(video)), mask_ratio=None)
        hidden = self.touch_model.post_layernorm(hidden)
        global_tokens = self.touch_projection(hidden[:, :1]).reshape(b, sensors, 1, self.feature_dim)
        spatial = self.touch_projection(hidden[:, 6:]).reshape(
            b,
            sensors,
            self.temporal_units,
            self.grid_size,
            self.grid_size,
            self.feature_dim,
        )
        return EncodedFeatures(
            global_tokens=global_tokens,
            spatial_grid=spatial,
            global_time_ids=torch.full((1,), -1, device=images.device, dtype=torch.long),
        )

    def discard_training_modules(self) -> None:
        """Remove both reconstruction branches from a downstream encoder instance."""
        for name in (
            "decoder_pos_embed",
            "mask_token",
            "mask_token_diff",
            "decoder_embed",
            "touch_decoder_blocks",
            "decoder_norm",
            "decoder_pred_video",
            "decoder_embed_diff",
            "diff_touch_decoder_blocks",
            "diff_decoder_norm",
            "diff_decoder_pred_video",
        ):
            setattr(self, name, None)

    def patchify(self, images: Tensor) -> Tensor:
        video, (b, sensors) = self._flatten(images)
        patches = patchify_video(video, self.patch_size, self.tubelet_size)
        return patches.reshape(b, sensors, *patches.shape[1:])

    def unpatchify(self, patches: Tensor) -> Tensor:
        if patches.ndim != 4:
            raise ValueError("AnyTouch2 patches must be [B,S,N,P]")
        b, sensors = patches.shape[:2]
        video = unpatchify_video(
            patches.reshape(b * sensors, *patches.shape[2:]),
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
            temporal_units=self.temporal_units,
            grid_height=self.grid_size,
            grid_width=self.grid_size,
        )
        return video.permute(0, 2, 1, 3, 4).reshape(
            b, sensors, self.num_frames, 3, self.image_size, self.image_size
        )

    def load_pretrained(
        self,
        path: str | Path,
        source_grid: tuple[int, int, int] | None = None,
    ) -> dict[str, Any]:
        raw = read_state_dict(path)
        state = {}
        for key, value in raw.items():
            for prefix in ("module.", "model.", "touch_mae_model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            state[key] = value
        position_key = "touch_model.embeddings.position_embedding.weight"
        if position_key in state and state[position_key].shape != self.state_dict()[position_key].shape:
            if source_grid is None:
                source_tokens = state[position_key].shape[0] - 1
                spatial = int((source_tokens / self.temporal_units) ** 0.5)
                if self.temporal_units * spatial * spatial != source_tokens:
                    raise ValueError("AnyTouch2 checkpoint position grid is ambiguous; set checkpoint_source_grid")
                source_grid = (self.temporal_units, spatial, spatial)
            state[position_key] = interpolate_video_position_embedding(
                state[position_key],
                source_grid=source_grid,
                target_grid=(self.temporal_units, self.grid_size, self.grid_size),
                has_cls=True,
            )
        decoder_position_key = "decoder_pos_embed"
        if (
            decoder_position_key in state
            and state[decoder_position_key].shape != self.state_dict()[decoder_position_key].shape
        ):
            if source_grid is None:
                source_tokens = state[decoder_position_key].shape[1] - 1
                spatial = int((source_tokens / self.temporal_units) ** 0.5)
                if self.temporal_units * spatial * spatial != source_tokens:
                    raise ValueError(
                        "AnyTouch2 decoder position grid is ambiguous; set checkpoint_source_grid"
                    )
                source_grid = (self.temporal_units, spatial, spatial)
            state[decoder_position_key] = interpolate_video_position_embedding(
                state[decoder_position_key],
                source_grid=source_grid,
                target_grid=(self.temporal_units, self.grid_size, self.grid_size),
                has_cls=True,
            )
        report = load_filtered_state(self, state, source=str(path))
        required_prefixes = (
            "touch_model.",
            "touch_projection.",
            "sensor_token",
            "decoder_embed.",
            "decoder_pos_embed",
            "touch_decoder_blocks.",
            "decoder_norm.",
            "decoder_pred_video.",
            "mask_token",
            "decoder_embed_diff.",
            "diff_touch_decoder_blocks.",
            "diff_decoder_norm.",
            "diff_decoder_pred_video.",
            "mask_token_diff",
        )
        required_missing = [
            key for key in report["missing_keys"] if key.startswith(required_prefixes)
        ]
        required_mismatch = [
            key for key in report["shape_mismatch"] if key.startswith(required_prefixes)
        ]
        if required_missing or required_mismatch:
            raise ValueError(
                "AnyTouch2 encoder/decoder checkpoint is incompatible: "
                f"missing={required_missing[:8]}, shape_mismatch={required_mismatch[:8]}"
            )
        return report
