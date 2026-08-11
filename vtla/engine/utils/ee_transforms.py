#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batched (torch) end-effector pose transforms for the EE action/state modes.

Supported rotation representations (``rot_mode`` parameter):

- ``"rot6d"`` (default): per arm ``[pos(3), rot6d(6), gripper(1)]`` = 10 dims.
  rot6d is the first two columns of the rotation matrix (Zhou et al. 2019).
- ``"quat"``: per arm ``[pos(3), qx, qy, qz, qw, gripper(1)]`` = 8 dims.
  Quaternion convention: ``(x, y, z, w)`` (scalar-last / ROS convention).

For a dual-arm robot the full vector is ``n_arms * per_arm_dim`` dims, ordered
``right`` arm first then ``left`` (matching the offline conversion tools).

All relative/absolute pose math is performed in rotation-matrix space internally,
so ``ee_to_relative`` / ``ee_to_absolute`` are correct for both formats.

Two conversions (per arm; positions and rotations in the reference's *local* frame,
gripper kept absolute — mirrors the ``T0^{-1}·Tt`` convention of the offline script):

- relative  (action -> relative-to-reference):  p_rel = R_s^T (p_a - p_s),  R_rel = R_s^T R_a
- absolute  (relative -> back to reference frame): p_a = p_s + R_s p_rel,  R_a = R_s R_rel

where ``s`` is the reference pose (the current observation EE pose) and ``a`` is the action pose.
The two are exact inverses.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Per-arm layout constants per rot_mode.
# ---------------------------------------------------------------------------
# rot6d: [pos(3), rot6d_0..5(6), gripper(1)] = 10 dims
PER_ARM_DIM = 10        # kept for backward compatibility
# quat:  [pos(3), qx, qy, qz, qw(4), gripper(1)] = 8 dims
PER_ARM_DIM_QUAT = 8

PER_ARM_DIM_BY_ROT_MODE: dict[str, int] = {
    "rot6d": 10,
    "quat": 8,
}

# Legacy slice constants (rot6d-only; kept for import backward compatibility).
_POS = slice(0, 3)
_ROT6D = slice(3, 9)
_GRIP = slice(9, 10)


def per_arm_dim(rot_mode: str) -> int:
    """Return the packed dimension per arm for ``rot_mode``."""
    try:
        return PER_ARM_DIM_BY_ROT_MODE[rot_mode]
    except KeyError:
        raise ValueError(
            f"Unknown rot_mode '{rot_mode}'. Expected one of {list(PER_ARM_DIM_BY_ROT_MODE)}."
        )


# ---------------------------------------------------------------------------
# rot6d <-> matrix (existing, unchanged)
# ---------------------------------------------------------------------------

def matrix_to_rot6d(matrix: Tensor) -> Tensor:
    """``(..., 3, 3)`` rotation matrix -> ``(..., 6)`` rot6d (first two columns)."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """``(..., 6)`` rot6d -> ``(..., 3, 3)`` rotation matrix via Gram-Schmidt (Zhou 2019)."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    # Remove the b1 component from a2, then normalise.
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    # Columns of the rotation matrix are b1, b2, b3.
    return torch.stack([b1, b2, b3], dim=-1)


# ---------------------------------------------------------------------------
# quat <-> matrix  (scalar-last convention: [x, y, z, w])
# ---------------------------------------------------------------------------

def quat_to_matrix(quat: Tensor) -> Tensor:
    """``(..., 4)`` quaternion ``[x, y, z, w]`` -> ``(..., 3, 3)`` rotation matrix.

    Uses the standard formula; does NOT normalise the input — pass unit quaternions.
    """
    x, y, z, w = quat.unbind(-1)
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mat = torch.stack(
        [
            1.0 - 2.0 * (y2 + z2),   2.0 * (xy - wz),          2.0 * (xz + wy),
            2.0 * (xy + wz),          1.0 - 2.0 * (x2 + z2),    2.0 * (yz - wx),
            2.0 * (xz - wy),          2.0 * (yz + wx),           1.0 - 2.0 * (x2 + y2),
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    return mat


def matrix_to_quat(matrix: Tensor) -> Tensor:
    """``(..., 3, 3)`` rotation matrix -> ``(..., 4)`` quaternion ``[x, y, z, w]``.

    Uses Shepperd's method (case-split on the dominant diagonal element for numerical
    stability). Output quaternions are unit-normalised.
    """
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    n = m.shape[0]

    q = torch.zeros(n, 4, dtype=m.dtype, device=m.device)  # [x, y, z, w]

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    # Case 1: trace > 0  → extract from w.
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2.0  # 4w
        q[mask1, 3] = 0.25 * s
        q[mask1, 0] = (m[mask1, 2, 1] - m[mask1, 1, 2]) / s
        q[mask1, 1] = (m[mask1, 0, 2] - m[mask1, 2, 0]) / s
        q[mask1, 2] = (m[mask1, 1, 0] - m[mask1, 0, 1]) / s

    # Case 2: m[0,0] is the largest diagonal element.
    mask2 = ~mask1 & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + m[mask2, 0, 0] - m[mask2, 1, 1] - m[mask2, 2, 2]) * 2.0  # 4x
        q[mask2, 3] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s
        q[mask2, 0] = 0.25 * s
        q[mask2, 1] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s
        q[mask2, 2] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s

    # Case 3: m[1,1] is the largest.
    mask3 = ~mask1 & ~mask2 & (m[:, 1, 1] > m[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + m[mask3, 1, 1] - m[mask3, 0, 0] - m[mask3, 2, 2]) * 2.0  # 4y
        q[mask3, 3] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s
        q[mask3, 0] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s
        q[mask3, 1] = 0.25 * s
        q[mask3, 2] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s

    # Case 4: m[2,2] is the largest.
    mask4 = ~mask1 & ~mask2 & ~mask3
    if mask4.any():
        s = torch.sqrt(1.0 + m[mask4, 2, 2] - m[mask4, 0, 0] - m[mask4, 1, 1]) * 2.0  # 4z
        q[mask4, 3] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s
        q[mask4, 0] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s
        q[mask4, 1] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s
        q[mask4, 2] = 0.25 * s

    # Normalise for robustness (input matrix may not be perfectly orthogonal).
    q = torch.nn.functional.normalize(q, dim=-1)
    return q.reshape(*batch_shape, 4)


# ---------------------------------------------------------------------------
# Generic pack / unpack (rot_mode aware)
# ---------------------------------------------------------------------------

def _unpack(x: Tensor, n_arms: int, rot_mode: str = "rot6d") -> tuple[Tensor, Tensor, Tensor]:
    """Packed ``(..., n_arms*pad)`` -> (pos ``(..., n_arms, 3)``, R ``(..., n_arms, 3, 3)``, grip ``(..., n_arms, 1)``).

    Rotation is always returned as a rotation **matrix** regardless of ``rot_mode``; pack/unpack
    are the only places that know about the storage format.
    """
    pad = per_arm_dim(rot_mode)
    expected = n_arms * pad
    if x.shape[-1] != expected:
        raise ValueError(
            f"Expected last dim {expected} for {n_arms} arms with rot_mode='{rot_mode}', "
            f"got {x.shape[-1]}."
        )
    blocks = x.reshape(*x.shape[:-1], n_arms, pad)
    pos = blocks[..., 0:3]
    grip = blocks[..., pad - 1: pad]
    if rot_mode == "rot6d":
        rot = rot6d_to_matrix(blocks[..., 3:9])
    elif rot_mode == "quat":
        rot = quat_to_matrix(blocks[..., 3:7])
    else:
        raise ValueError(f"Unknown rot_mode '{rot_mode}'.")
    return pos, rot, grip


def _pack(pos: Tensor, rot: Tensor, grip: Tensor, rot_mode: str = "rot6d") -> Tensor:
    """Inverse of :func:`_unpack`: (pos, rotation_matrix, grip) -> packed ``(..., n_arms*pad)``."""
    if rot_mode == "rot6d":
        rot_packed = matrix_to_rot6d(rot)
    elif rot_mode == "quat":
        rot_packed = matrix_to_quat(rot)
    else:
        raise ValueError(f"Unknown rot_mode '{rot_mode}'.")
    blocks = torch.cat([pos, rot_packed, grip], dim=-1)  # (..., n_arms, pad)
    return blocks.reshape(*blocks.shape[:-2], -1)  # (..., n_arms * pad)


def _align_reference(reference: Tensor, other: Tensor) -> Tensor:
    """Broadcast the per-sample ``reference`` over any extra (e.g. chunk) dims that ``other`` has."""
    while reference.ndim < other.ndim:
        reference = reference.unsqueeze(-2)
    return reference


# ---------------------------------------------------------------------------
# Relative / absolute pose conversion
# ---------------------------------------------------------------------------

def ee_to_relative(
    reference_ee: Tensor,
    action_ee: Tensor,
    n_arms: int = 2,
    rot_mode: str = "rot6d",
) -> Tensor:
    """Convert absolute(-in-reference-frame) action poses to poses relative to ``reference_ee``.

    Args:
        reference_ee: ``(B, n_arms*pad)`` reference EE pose (the current observation).
        action_ee: ``(B, n_arms*pad)`` or ``(B, T, n_arms*pad)`` action EE pose(s).
        n_arms: Number of arms packed in the vector.
        rot_mode: Rotation storage format — ``"rot6d"`` (default) or ``"quat"``.

    Returns:
        Relative EE pose(s), same shape as ``action_ee``. Gripper dims are passed through (absolute).
    """
    reference_ee = _align_reference(reference_ee, action_ee)
    p_s, R_s, _ = _unpack(reference_ee, n_arms, rot_mode)
    p_a, R_a, grip_a = _unpack(action_ee, n_arms, rot_mode)

    R_s_T = R_s.transpose(-1, -2)
    p_rel = torch.matmul(R_s_T, (p_a - p_s).unsqueeze(-1)).squeeze(-1)
    R_rel = torch.matmul(R_s_T, R_a)
    return _pack(p_rel, R_rel, grip_a, rot_mode)


def ee_to_absolute(
    reference_ee: Tensor,
    relative_ee: Tensor,
    n_arms: int = 2,
    rot_mode: str = "rot6d",
) -> Tensor:
    """Inverse of :func:`ee_to_relative`: convert relative poses back into ``reference_ee``'s frame.

    Args:
        reference_ee: ``(B, n_arms*pad)`` reference EE pose (the current observation).
        relative_ee: ``(B, n_arms*pad)`` or ``(B, T, n_arms*pad)`` relative EE pose(s).
        n_arms: Number of arms packed in the vector.
        rot_mode: Rotation storage format — ``"rot6d"`` (default) or ``"quat"``.

    Returns:
        Absolute(-in-reference-frame) EE pose(s), same shape as ``relative_ee``.
    """
    reference_ee = _align_reference(reference_ee, relative_ee)
    p_s, R_s, _ = _unpack(reference_ee, n_arms, rot_mode)
    p_rel, R_rel, grip = _unpack(relative_ee, n_arms, rot_mode)

    p_a = p_s + torch.matmul(R_s, p_rel.unsqueeze(-1)).squeeze(-1)
    R_a = torch.matmul(R_s, R_rel)
    return _pack(p_a, R_a, grip, rot_mode)
