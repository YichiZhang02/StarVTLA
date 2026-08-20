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

"""Realman forward-kinematics helpers shared between the offline conversion tool and the
real-time EpisodeEEPreprocessorStep.

These functions mirror the logic in tools/convert_joints_to_eepose.py so that the
same FK math can be reused at inference time without importing from the tools/ directory.

Dual-arm layout (16-dim joint vector → 20-dim EE vector):
  - right arm: joints[0:7] + gripper[7]   → [xyz(3), rot6d(6), gripper(1)] = 10 dims
  - left arm:  joints[8:15] + gripper[15] → [xyz(3), rot6d(6), gripper(1)] = 10 dims
  - (indices are derived from feature names via joint_indices())
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

DOF = 7
PER_ARM_DIM = 10
EE_DIM = 20

# Realman SDK lives under deployment/sdk (vendored, no pip install required).
_SDK_PATH = Path(__file__).resolve().parents[3] / "deployment" / "sdk"


def _ensure_sdk_on_path() -> None:
    if str(_SDK_PATH) not in sys.path:
        sys.path.insert(0, str(_SDK_PATH))


def make_realman_algo():
    """Return an Algo instance for the RM-75-E arm (offline FK only, no arm connection)."""
    _ensure_sdk_on_path()
    from Robotic_Arm.rm_ctypes_wrap import rm_force_type_e, rm_robot_arm_model_e
    from Robotic_Arm.rm_robot_interface import Algo

    return Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E, rm_force_type_e.RM_MODEL_RM_B_E)


def joint_indices(names: list[str]) -> dict:
    """Derive one or two complete arm layouts from observation.state feature names.

    Args:
        names: Ordered list of feature names for each dimension of observation.state.

    Returns:
        Dict with keys ``left_joints``, ``right_joints``, ``left_grip``, ``right_grip``.
    """
    idx: dict = {"left_joints": [], "right_joints": [], "left_grip": None, "right_grip": None}
    for i, n in enumerate(names):
        low = n.lower()
        side = "left" if low.startswith("left") else "right" if low.startswith("right") else None
        if side is None:
            continue
        if "gripper" in low:
            idx[f"{side}_grip"] = i
        elif "joint" in low:
            idx[f"{side}_joints"].append(i)
    sides: list[str] = []
    for side in ("right", "left"):
        joints = idx[f"{side}_joints"]
        grip = idx[f"{side}_grip"]
        if not joints and grip is None:
            continue
        if len(joints) != DOF:
            raise ValueError(
                f"Expected {DOF} {side}_joints, found {len(joints)} in names={names}"
            )
        if grip is None:
            raise ValueError(f"Missing {side} gripper index in names={names}")
        sides.append(side)
    if not sides:
        raise ValueError(f"No complete left/right arm found in names={names}")
    idx["sides"] = tuple(sides)
    return idx


def split_arms(vec: np.ndarray, jidx: dict):
    """Split a joint vector into per-arm tuples in canonical arm order."""
    vec = np.asarray(vec, dtype=np.float64)
    return tuple(
        (vec[jidx[f"{side}_joints"]], float(vec[jidx[f"{side}_grip"]]))
        for side in jidx["sides"]
    )


def fk(algo, joints_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-arm FK: 7 joint radians → (pos xyz (3,), rotation matrix (3,3))."""
    joints_deg = np.degrees(joints_rad).tolist()
    pose = algo.rm_algo_forward_kinematics(joints_deg, flag=0)  # [x,y,z, qw,qx,qy,qz]
    pos = np.array(pose[:3], dtype=np.float64)
    qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
    mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return pos, mat


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → 6-dim rot6d (first two columns)."""
    return np.concatenate([mat[:, 0], mat[:, 1]]).astype(np.float64)


def mat_to_rot(mat: np.ndarray, rot_mode: str) -> np.ndarray:
    """3×3 rotation matrix → rotation representation in ``rot_mode`` format.

    Args:
        mat: (3, 3) rotation matrix.
        rot_mode: ``"rot6d"`` (6-dim first-two-columns) or ``"quat"`` (4-dim [x, y, z, w]).

    Returns:
        1D numpy array of length 6 (rot6d) or 4 (quat).
    """
    if rot_mode == "rot6d":
        return mat_to_rot6d(mat)
    elif rot_mode == "quat":
        q = R.from_matrix(mat).as_quat()  # (x, y, z, w)
        return q.astype(np.float64)
    else:
        raise ValueError(f"Unknown rot_mode '{rot_mode}'. Expected 'rot6d' or 'quat'.")


def relative_arm_ee(pos, mat, grip, p0, R0, rot_mode: str = "rot6d") -> np.ndarray:
    """Single-arm: absolute EE → pose relative to episode-start frame T0.

    pos_rel = R0^T (pt - p0),  R_rel = R0^T · Rt,  gripper kept absolute.
    Returns a vector [pos(3), rot(rot_dim), gripper(1)] where rot_dim is 6 for rot6d or 4 for quat.
    """
    R0t = R0.T
    p_rel = R0t @ (pos - p0)
    R_rel = R0t @ mat
    return np.concatenate([p_rel, mat_to_rot(R_rel, rot_mode), [grip]]).astype(np.float64)


def fk_both(algo, joint_vector: np.ndarray, jidx: dict):
    """FK for every arm present in ``jidx`` (one or two arms)."""
    return tuple((fk(algo, joints), grip) for joints, grip in split_arms(joint_vector, jidx))


def to_episode_ee(algo, vec16: np.ndarray, jidx: dict, baseline, rot_mode: str = "rot6d") -> np.ndarray:
    """Convert 16-dim joint vector to EE pose relative to episode-start frame.

    Args:
        algo: Realman Algo instance (from make_realman_algo()).
        vec16: 16-dim joint state (right-arm joints+gripper, then left-arm).
        jidx: Index dict from joint_indices().
        baseline: ((R_p0, R_R0), (L_p0, L_R0)) from the episode's first frame.
        rot_mode: Rotation storage format — ``"rot6d"`` (default, 10 dims/arm) or ``"quat"`` (8 dims/arm).

    Returns:
        float32 array of shape (n_arms * per_arm_dim,).
    """
    arms = fk_both(algo, vec16, jidx)
    return np.concatenate([
        relative_arm_ee(pos, mat, grip, pos0, mat0, rot_mode)
        for ((pos, mat), grip), (pos0, mat0) in zip(arms, baseline, strict=True)
    ]).astype(np.float32)


def absolute_arm_ee(pos, mat, grip, rot_mode: str = "rot6d") -> np.ndarray:
    """Single-arm: absolute EE in the robot base frame (no T0). [pos(3), rot(rot_dim), gripper(1)]."""
    return np.concatenate([pos, mat_to_rot(mat, rot_mode), [grip]]).astype(np.float64)


def to_absolute_ee(algo, vec16: np.ndarray, jidx: dict, rot_mode: str = "rot6d") -> np.ndarray:
    """Convert 16-dim joint vector to base-frame EE pose (Tt, no episode baseline).

    Same packing/layout as :func:`to_episode_ee` (RIGHT arm first then LEFT, per arm
    ``[pos(3), rot(rot_dim), gripper(1)]``) but expressed in the robot base frame directly.
    Used by state_mode='absolute_rot6d' or 'absolute_quat'.

    Returns:
        float32 array of shape (n_arms * per_arm_dim,).
    """
    return np.concatenate([
        absolute_arm_ee(pos, mat, grip, rot_mode)
        for (pos, mat), grip in fk_both(algo, vec16, jidx)
    ]).astype(np.float32)


def compute_baseline(algo, vec16: np.ndarray, jidx: dict) -> tuple:
    """Compute the episode-start FK baseline from the first-frame joint state.

    Returns:
        ((R_p0, R_R0), (L_p0, L_R0))
    """
    return tuple(pose for pose, _grip in fk_both(algo, vec16, jidx))
