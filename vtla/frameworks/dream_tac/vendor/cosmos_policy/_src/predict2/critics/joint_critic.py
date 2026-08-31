# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn


class JointCritic(Protocol):
    """
    Scores joint samples (video_latent, action) given context x.

    This is intentionally lightweight and model-agnostic so it can wrap:
    - a learned Q / value model
    - an energy model
    - a consistency discriminator (action↔video)
    """

    def score(self, *, context: dict, video_latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            scores: (B,) higher is better.
        """
        ...


@dataclass(frozen=True)
class OptimalityBinnerConfig:
    num_bins: int = 2  # 2 => binary {0,1}
    # Robust binning via quantiles computed on a running buffer / dataset statistics.
    # If provided, expects shape (num_bins-1,) sorted ascending.
    thresholds: torch.Tensor | None = None


class OptimalityBinner:
    """
    Converts critic scores into monotone optimality labels o∈{0..K-1}.

    CFGRL uses a binary optimality variable; we generalize to multi-bin labels
    to improve stability and allow finer-grained guidance sweeps.
    """

    def __init__(self, config: OptimalityBinnerConfig):
        self.config = config
        assert self.config.num_bins >= 2

    @torch.no_grad()
    def bin(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 1:
            raise ValueError(f"scores must be (B,), got {tuple(scores.shape)}")

        K = self.config.num_bins
        if self.config.thresholds is None:
            # Default: median split for binary, or equal-width bins in score-space for K>2.
            if K == 2:
                thr = scores.median()
                return (scores >= thr).long()
            smin, smax = scores.min(), scores.max()
            # Avoid zero width.
            edges = torch.linspace(smin, smax + 1e-6, steps=K + 1, device=scores.device)
            # bucketize into [0..K-1]
            return torch.bucketize(scores, edges[1:-1], right=False).long()

        thr = self.config.thresholds.to(device=scores.device)
        if thr.numel() != K - 1:
            raise ValueError(f"thresholds must have {K-1} elements, got {thr.numel()}")
        return torch.bucketize(scores, thr, right=False).long()


class SimpleJointMLPCritic(nn.Module):
    """
    Minimal reference critic: pools video latents and concatenates actions.
    This is *not* intended to be your final paper model, but is useful for wiring.
    """

    def __init__(self, video_ch: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(video_ch + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def score(self, *, context: dict, video_latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del context
        if video_latent.ndim != 5:
            raise ValueError(f"video_latent must be (B,C,T,H,W), got {tuple(video_latent.shape)}")
        if action.ndim != 3:
            raise ValueError(f"action must be (B,T,A), got {tuple(action.shape)}")

        # Pool video and action over time & space to get a cheap scalar score.
        v = video_latent.mean(dim=[2, 3, 4])  # (B,C)
        a = action.mean(dim=1)  # (B,A)
        x = torch.cat([v, a], dim=-1)
        return self.net(x).squeeze(-1)

