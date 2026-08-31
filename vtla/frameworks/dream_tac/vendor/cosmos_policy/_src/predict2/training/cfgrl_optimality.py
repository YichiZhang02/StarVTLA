# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CFGRLOptimalityConfig:
    """
    Training-time recipe for CFGRL-style optimality-conditioning.

    CFGRL (Frans et al., 2025) trains a single conditional model with an optimality
    variable o and includes an unconditional dropout case o=∅ during training.

    In this repo, `optimality` is represented as an integer bin:
      - o = -1 => ∅ (unconditional)
      - o in [0..K-1] => conditioned bins (binary K=2 by default)
    """

    p_uncond: float = 0.1  # probability to drop optimality -> ∅
    num_bins: int = 2
    null_bin: int = -1
    positive_bin: int = 1  # for binary case, o=1 is “optimal”
    negative_bin: int = 0


@torch.no_grad()
def apply_cfgrl_optimality_dropout(optimality: torch.Tensor, cfg: CFGRLOptimalityConfig) -> torch.Tensor:
    """
    Apply unconditional dropout: with probability p_uncond, set o = null_bin.
    """
    if optimality.ndim != 1:
        raise ValueError(f"optimality must be (B,), got {tuple(optimality.shape)}")

    if cfg.p_uncond <= 0:
        return optimality
    if cfg.p_uncond >= 1:
        return torch.full_like(optimality, fill_value=cfg.null_bin)

    B = optimality.shape[0]
    drop = torch.rand(B, device=optimality.device) < cfg.p_uncond
    out = optimality.clone()
    out[drop] = cfg.null_bin
    return out


@torch.no_grad()
def label_binary_optimality_from_score(scores: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Utility: scores -> binary optimality label.
    - Used for advantage-like scores (A>=0) or calibrated critic scores.
    Returns: (B,) long tensor in {0,1}
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be (B,), got {tuple(scores.shape)}")
    return (scores >= threshold).long()

