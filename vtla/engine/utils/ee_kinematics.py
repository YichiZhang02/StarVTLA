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

"""Shared offline/online FK and TCP geometry. FK returns flange poses;
all packed model EE poses are TCP rot6d, after applying the robot tool calibration.
Quaternion conversion occurs only at the SDK boundary."""

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


def make_realman_algo(force_type: str):
    """Return an RM-75-E Algo configured for a concrete robot's EE variant."""
    _ensure_sdk_on_path()
    from Robotic_Arm.rm_ctypes_wrap import rm_force_type_e, rm_robot_arm_model_e
    from Robotic_Arm.rm_robot_interface import Algo

    force_types = {
        "base": rm_force_type_e.RM_MODEL_RM_B_E,
        "isf": rm_force_type_e.RM_MODEL_RM_ISF_E,
    }
    if force_type not in force_types:
        raise ValueError(f"Unsupported RealMan kinematics force_type={force_type!r}")
    return Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E, force_types[force_type])


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


def flange_to_tcp(
    flange_pos: np.ndarray,
    flange_rot: np.ndarray,
    flange_tcp_xyz_m: tuple[float, float, float] | np.ndarray,
    flange_tcp_rpy_deg: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose ``T_base_flange @ T_flange_tcp``."""
    flange_pos = np.asarray(flange_pos, dtype=np.float64)
    flange_rot = np.asarray(flange_rot, dtype=np.float64)
    offset_pos = np.asarray(flange_tcp_xyz_m, dtype=np.float64)
    offset_rot = R.from_euler("xyz", flange_tcp_rpy_deg, degrees=True).as_matrix()
    return flange_pos + flange_rot @ offset_pos, flange_rot @ offset_rot


def tcp_to_flange(
    tcp_pos: np.ndarray,
    tcp_rot: np.ndarray,
    flange_tcp_xyz_m: tuple[float, float, float] | np.ndarray,
    flange_tcp_rpy_deg: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the flange-to-TCP extrinsic and return ``T_base_flange``."""
    tcp_pos = np.asarray(tcp_pos, dtype=np.float64)
    tcp_rot = np.asarray(tcp_rot, dtype=np.float64)
    offset_pos = np.asarray(flange_tcp_xyz_m, dtype=np.float64)
    offset_rot = R.from_euler("xyz", flange_tcp_rpy_deg, degrees=True).as_matrix()
    flange_rot = tcp_rot @ offset_rot.T
    return tcp_pos - flange_rot @ offset_pos, flange_rot


def _to_tcp(
    pos: np.ndarray,
    mat: np.ndarray,
    side: str,
    flange_tcp_calibration: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if flange_tcp_calibration is None or side not in flange_tcp_calibration:
        raise ValueError(f"Missing flange-to-TCP calibration for side={side!r}.")
    xyz, rpy = flange_tcp_calibration[side]
    return flange_to_tcp(pos, mat, xyz, rpy)


def relative_arm_ee(pos, mat, grip, p0, R0) -> np.ndarray:
    """Express a TCP pose in its episode-start frame; retain geometric rot6d."""
    R0t = R0.T
    p_rel = R0t @ (pos - p0)
    R_rel = R0t @ mat
    return np.concatenate([p_rel, mat_to_rot6d(R_rel), [grip]]).astype(np.float64)


def fk_both(
    algo,
    joint_vector: np.ndarray,
    jidx: dict,
    flange_tcp_calibration: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] | None = None,
):
    """FK for every arm present in ``jidx`` (one or two arms)."""
    arms = []
    for side, (joints, grip) in zip(jidx["sides"], split_arms(joint_vector, jidx), strict=True):
        pos, mat = fk(algo, joints)
        pos, mat = _to_tcp(pos, mat, side, flange_tcp_calibration)
        arms.append(((pos, mat), grip))
    return tuple(arms)


def to_episode_ee(
    algo,
    vec16: np.ndarray,
    jidx: dict,
    baseline,
    flange_tcp_calibration=None,
) -> np.ndarray:
    """Convert joints to episode-relative TCP poses using the calibrated TCP baseline."""
    arms = fk_both(algo, vec16, jidx, flange_tcp_calibration)
    return np.concatenate([
        relative_arm_ee(pos, mat, grip, pos0, mat0)
        for ((pos, mat), grip), (pos0, mat0) in zip(arms, baseline, strict=True)
    ]).astype(np.float32)


def absolute_arm_ee(pos, mat, grip) -> np.ndarray:
    """Single-arm: absolute EE in the robot base frame (no T0). [pos(3), rot(rot_dim), gripper(1)]."""
    return np.concatenate([pos, mat_to_rot6d(mat), [grip]]).astype(np.float64)


def to_absolute_ee(
    algo,
    vec16: np.ndarray,
    jidx: dict,
    flange_tcp_calibration=None,
) -> np.ndarray:
    """Convert joints to base-frame TCP poses using the robot tool calibration."""
    return np.concatenate([
        absolute_arm_ee(pos, mat, grip)
        for (pos, mat), grip in fk_both(algo, vec16, jidx, flange_tcp_calibration)
    ]).astype(np.float32)


def compute_baseline(
    algo,
    vec16: np.ndarray,
    jidx: dict,
    flange_tcp_calibration=None,
) -> tuple:
    """Compute the episode-start FK baseline from the first-frame joint state.

    Returns:
        ((R_p0, R_R0), (L_p0, L_R0))
    """
    return tuple(
        pose
        for pose, _grip in fk_both(
            algo, vec16, jidx, flange_tcp_calibration
        )
    )
