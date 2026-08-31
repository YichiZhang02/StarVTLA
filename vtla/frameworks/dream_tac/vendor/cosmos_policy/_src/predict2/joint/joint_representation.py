# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointVideoAction:
    """
    Joint variable z = (video_latent, action) for CFGRL-style joint guidance.

    This codebase’s diffusion nets operate on 5D tensors shaped (B, C, T, H, W).
    Robot actions are typically (B, T, A).

    We provide a *packing* that makes joint diffusion possible without changing the
    solver API: represent actions as (B, A, T, 1, 1) and concatenate along channel:

        z_packed = cat([video_latent, action_as_channels], dim=1)

    Training must be done with this packing so the net learns to model both video
    and action jointly. At inference-time, the same packing supports joint CFG.
    """

    video_latent: torch.Tensor  # (B, Cv, T, H, W)
    action: torch.Tensor  # (B, T, A) or (B, 1, T, A) depending on caller


def pack_joint(video_latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Pack (video_latent, action) into a single 5D tensor (B, Cv+A, T, H, W).
    Action is broadcast over spatial dims to match H,W.
    """
    if video_latent.ndim != 5:
        raise ValueError(f"video_latent must be (B,C,T,H,W), got {tuple(video_latent.shape)}")

    B, _, T, H, W = video_latent.shape

    if action.ndim == 2:
        raise ValueError("action must include time dimension (B,T,A)")
    if action.ndim == 3:
        # (B, T, A) -> (B, A, T, 1, 1)
        action_btA = action
    elif action.ndim == 4 and action.shape[1] == 1:
        # (B, 1, T, A) -> (B, T, A)
        action_btA = action[:, 0]
    else:
        raise ValueError(f"action must be (B,T,A) or (B,1,T,A), got {tuple(action.shape)}")

    if action_btA.shape[0] != B or action_btA.shape[1] != T:
        raise ValueError(
            f"action batch/time must match video_latent: action {tuple(action_btA.shape)} vs video {tuple(video_latent.shape)}"
        )

    action_B_A_T_1_1 = action_btA.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    action_B_A_T_H_W = action_B_A_T_1_1.expand(B, action_B_A_T_1_1.shape[1], T, H, W)
    return torch.cat([video_latent, action_B_A_T_H_W], dim=1)


def unpack_joint(z_packed: torch.Tensor, action_dim: int) -> JointVideoAction:
    """
    Inverse of `pack_joint`.
    Assumes the last `action_dim` channels correspond to action broadcast.
    """
    if z_packed.ndim != 5:
        raise ValueError(f"z_packed must be (B,C,T,H,W), got {tuple(z_packed.shape)}")
    if action_dim <= 0:
        raise ValueError("action_dim must be > 0")

    B, C, T, H, W = z_packed.shape
    if action_dim >= C:
        raise ValueError(f"action_dim={action_dim} must be < total channels {C}")

    video_latent = z_packed[:, : C - action_dim]
    action_broadcast = z_packed[:, C - action_dim :]

    # Recover action as (B,T,A) by taking spatial mean (all positions are identical by construction).
    action_B_A_T = action_broadcast.mean(dim=[3, 4])  # (B, A, T)
    action_B_T_A = action_B_A_T.permute(0, 2, 1).contiguous()
    return JointVideoAction(video_latent=video_latent, action=action_B_T_A)

