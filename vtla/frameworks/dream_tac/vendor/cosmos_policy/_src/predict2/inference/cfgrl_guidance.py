# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


@torch.no_grad()
def set_optimality_for_sampling(
    data_batch: dict, *, optimality_bin: int, num_bins: int = 2, null_bin: int = -1
) -> dict:
    """
    Prepare a data_batch for CFGRL-style sampling.

    - `optimality_bin` sets the conditional branch label (o=1 in CFGRL’s binary case).
    - The unconditional branch is produced by the conditioner’s uncond path, which drops
      the optimality token to null_bin due to its embedder dropout_rate=1.0.
    """
    if "optimality" not in data_batch:
        raise KeyError("data_batch must contain key 'optimality' (B,) long tensor")

    if not isinstance(data_batch["optimality"], torch.Tensor):
        raise TypeError("data_batch['optimality'] must be a torch.Tensor")

    if data_batch["optimality"].dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise TypeError("data_batch['optimality'] must be an integer tensor")

    if optimality_bin == null_bin:
        raise ValueError("optimality_bin cannot be null_bin; use a real conditioned bin (e.g. 1)")

    if not (0 <= int(optimality_bin) < int(num_bins)):
        raise ValueError(f"optimality_bin must be in [0,{num_bins-1}], got {optimality_bin}")

    data_batch = dict(data_batch)
    data_batch["optimality"] = torch.full_like(data_batch["optimality"], fill_value=int(optimality_bin))
    return data_batch


def guidance_schedule_linear(w0: float, w1: float, step: int, num_steps: int) -> float:
    """
    Simple divergence-safeguard schedule: increase w over time.
    Useful when high guidance early causes solver instability.
    """
    if num_steps <= 1:
        return float(w1)
    alpha = float(step) / float(num_steps - 1)
    return (1.0 - alpha) * float(w0) + alpha * float(w1)

