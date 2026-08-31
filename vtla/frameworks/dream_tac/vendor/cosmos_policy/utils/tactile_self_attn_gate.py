# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scalar tactile gate from consecutive tactile RGB frames (uint8), for self-attn bias."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def mean_abs_diff_uint8_pair(
    left_curr: np.ndarray,
    right_curr: np.ndarray,
    left_prev: np.ndarray,
    right_prev: np.ndarray,
) -> float:
    """Mean |Δ| over all pixels and channels, normalized to ~[0,1] scale."""
    dl = np.abs(left_curr.astype(np.float32) - left_prev.astype(np.float32)).mean() / 255.0
    dr = np.abs(right_curr.astype(np.float32) - right_prev.astype(np.float32)).mean() / 255.0
    return float(max(dl, dr))


def scalar_gate_from_raw(
    raw: float,
    median: float = 0.002,
    mad: float = 0.001,
    sigmoid_k: float = 4.0,
    g_min: float = 0.15,
    g_max: float = 1.0,
) -> float:
    """Map instantaneous raw event to gate in [g_min, g_max] (fixed robust stats, no EMA)."""
    z = sigmoid_k * (raw - median) / (mad + 1e-6)
    z = max(-30.0, min(30.0, z))
    g01 = 1.0 / (1.0 + math.exp(-z))
    return g_min + (g_max - g_min) * g01
