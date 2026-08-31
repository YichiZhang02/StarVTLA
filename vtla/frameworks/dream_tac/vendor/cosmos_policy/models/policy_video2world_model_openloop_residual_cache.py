# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Open-loop-only: hard residual cache for DiT blocks is implemented in
# cosmos_policy.experiments.robot.openloop_hard_residual_cache (patch applied at runtime).
# This module re-exports helpers for discoverability; training / default inference use
# cosmos_policy.models.policy_video2world_model unchanged.

from cosmos_policy.experiments.robot.openloop_hard_residual_cache import (
    apply_openloop_hard_residual_cache,
    remove_openloop_hard_residual_cache,
    reset_openloop_denoise_counter,
)

__all__ = [
    "apply_openloop_hard_residual_cache",
    "remove_openloop_hard_residual_cache",
    "reset_openloop_denoise_counter",
]
