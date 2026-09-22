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

"""Geometric TCP pose composition and zero-centered relative action encoding.

Geometric relative poses retain identity rot6d. Only encode_relative_tcp subtracts
the identity encoding; decode_relative_tcp adds it back before composition.
All packed poses use [xyz(3), first two rotation columns(6), absolute gripper(1)].
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
PER_ARM_DIM = 10


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
def _unpack(x: Tensor, n_arms: int) -> tuple[Tensor, Tensor, Tensor]:
    """Unpack rot6d poses into position, rotation matrix and absolute gripper."""
    pad = PER_ARM_DIM
    expected = n_arms * pad
    if x.shape[-1] != expected:
        raise ValueError(
            f"Expected last dim {expected} for {n_arms} arms, "
            f"got {x.shape[-1]}."
        )
    blocks = x.reshape(*x.shape[:-1], n_arms, pad)
    pos = blocks[..., 0:3]
    grip = blocks[..., pad - 1: pad]
    rot = rot6d_to_matrix(blocks[..., 3:9])
    return pos, rot, grip


def _pack(pos: Tensor, rot: Tensor, grip: Tensor) -> Tensor:
    """Pack position, rotation matrix and absolute gripper as rot6d poses."""
    rot_packed = matrix_to_rot6d(rot)
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
) -> Tensor:
    """Geometric SE(3) relative poses in the reference TCP frame; grippers stay absolute."""
    reference_ee = _align_reference(reference_ee, action_ee)
    p_s, R_s, _ = _unpack(reference_ee, n_arms)
    p_a, R_a, grip_a = _unpack(action_ee, n_arms)

    R_s_T = R_s.transpose(-1, -2)
    p_rel = torch.matmul(R_s_T, (p_a - p_s).unsqueeze(-1)).squeeze(-1)
    R_rel = torch.matmul(R_s_T, R_a)
    return _pack(p_rel, R_rel, grip_a)


def ee_to_absolute(
    reference_ee: Tensor,
    relative_ee: Tensor,
    n_arms: int = 2,
) -> Tensor:
    """Compose geometric relative poses with reference TCP poses."""
    reference_ee = _align_reference(reference_ee, relative_ee)
    p_s, R_s, _ = _unpack(reference_ee, n_arms)
    p_rel, R_rel, grip = _unpack(relative_ee, n_arms)

    p_a = p_s + torch.matmul(R_s, p_rel.unsqueeze(-1)).squeeze(-1)
    R_a = torch.matmul(R_s, R_rel)
    return _pack(p_a, R_a, grip)


def encode_relative_tcp(reference: Tensor, action: Tensor, n_arms: int = 2) -> Tensor:
    """TCP-local SE(3) action, with identity rot6d subtracted (gripper absolute)."""
    reference = _align_reference(reference, action)
    p_s, R_s, _ = _unpack(reference, n_arms)
    p_a, R_a, grip = _unpack(action, n_arms)
    inverse = R_s.transpose(-1, -2)
    position = torch.matmul(inverse, (p_a - p_s).unsqueeze(-1)).squeeze(-1)
    # Rs.T @ (Ra - Rs) == Rs.T @ Ra - I for rotations. This form gives
    # exactly zero for identical stored orientations, avoiding roundoff in Rs.T @ Rs.
    rotation_delta = matrix_to_rot6d(torch.matmul(inverse, R_a - R_s))
    return torch.cat((position, rotation_delta, grip), -1).flatten(-2)


def decode_relative_tcp(reference: Tensor, action: Tensor, n_arms: int = 2) -> Tensor:
    """Restore identity BEFORE interpreting rotation residuals as a rotation.

    Reject nonfinite commands. Finite degenerate predicted rotations hold the
    anchor orientation; translations and absolute grippers remain independent.
    """
    if action.shape[-1] != 10 * n_arms:
        raise ValueError(f"Expected {10 * n_arms} action dimensions, got {action.shape[-1]}")
    if not torch.isfinite(action).all() or not torch.isfinite(reference).all():
        raise ValueError("Nonfinite TCP action or anchor")
    blocks = action.reshape(*action.shape[:-1], n_arms, 10)
    identity = blocks.new_tensor([1, 0, 0, 0, 1, 0])
    rotation = blocks[..., 3:9] + identity
    a1, a2 = rotation[..., :3], rotation[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    orthogonal = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    valid = (a1.norm(dim=-1) > 1e-6) & (orthogonal.norm(dim=-1) > 1e-6)
    rotation = torch.where(valid[..., None], rotation, identity)
    relative = torch.cat((blocks[..., :3], rotation, blocks[..., 9:]), -1).flatten(-2)
    return ee_to_absolute(reference, relative, n_arms=n_arms)
