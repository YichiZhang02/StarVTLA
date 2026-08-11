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

"""Shared EE-mode processor wiring used by all policy processor factories (pi05 / act / diffusion).

EE modes (state_mode='episode_rot6d'/'absolute_rot6d'/'episode_quat'/'absolute_quat',
action_mode='rot6d'/'quat') reuse the existing relative-action machinery: route_ee_batch
(in train.py) puts the episode/absolute EE state and action under the canonical
observation.state / action keys, then these helpers (a) remap the normalization stats
to those canonical keys and (b) build the pose-aware relative/absolute steps.

Backward-compat aliases (episode_ee → episode_rot6d, relative_ee → rot6d, etc.) are
handled by SensorRoutingMixin._normalise_ee_modes() before these utilities are called.
"""

from vtla.engine.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from vtla.engine.utils.constants import ACTION, OBS_STATE

from .sensor_routing import (
    ACTION_ABSOLUTE_EE,
    ACTION_ABSOLUTE_QUAT,
    ACTION_EPISODE_EE,
    ACTION_EPISODE_QUAT,
    ACTION_RELATIVE_EE,
    ACTION_RELATIVE_QUAT,
    OBS_STATE_ABSOLUTE_EE,
    OBS_STATE_ABSOLUTE_QUAT,
    OBS_STATE_EPISODE_EE,
    OBS_STATE_EPISODE_QUAT,
)

# Maps (state_mode, is_absolute) -> (obs_ee_key, action_ee_key, relative_stats_key)
_STATE_MODE_KEYS = {
    "episode_rot6d":  (OBS_STATE_EPISODE_EE,   ACTION_EPISODE_EE,   ACTION_RELATIVE_EE),
    "absolute_rot6d": (OBS_STATE_ABSOLUTE_EE,  ACTION_ABSOLUTE_EE,  ACTION_RELATIVE_EE),
    "episode_quat":   (OBS_STATE_EPISODE_QUAT, ACTION_EPISODE_QUAT, ACTION_RELATIVE_QUAT),
    "absolute_quat":  (OBS_STATE_ABSOLUTE_QUAT,ACTION_ABSOLUTE_QUAT,ACTION_RELATIVE_QUAT),
}


def remap_ee_dataset_stats(dataset_stats, config):
    """Return ``dataset_stats`` with EE stats placed under the canonical keys (shallow copy).

    Supports all four EE state modes (episode_rot6d, absolute_rot6d, episode_quat,
    absolute_quat) and both action modes (rot6d, quat).  No-op for joint modes.
    Returns the original dict unchanged if no EE mode is active.
    """
    if dataset_stats is None:
        return dataset_stats

    # Normalise legacy aliases (episode_ee → episode_rot6d, etc.).
    state_mode  = getattr(config, "state_mode",  "joint")
    action_mode = getattr(config, "action_mode", "joint")
    from .sensor_routing import _STATE_MODE_ALIASES, _ACTION_MODE_ALIASES
    state_mode  = _STATE_MODE_ALIASES.get(state_mode,  state_mode)
    action_mode = _ACTION_MODE_ALIASES.get(action_mode, action_mode)

    state_ee  = state_mode  in _STATE_MODE_KEYS
    action_ee = action_mode in ("rot6d", "quat")
    if not (state_ee or action_ee):
        return dataset_stats

    dataset_stats = dict(dataset_stats)

    if state_ee:
        obs_key, _act_key, _rel_key = _STATE_MODE_KEYS[state_mode]
        if obs_key in dataset_stats:
            dataset_stats[OBS_STATE] = dataset_stats[obs_key]

    if action_ee:
        # Determine the relative-action stats key based on the rotation format.
        is_quat    = action_mode == "quat"
        rel_key    = ACTION_RELATIVE_QUAT if is_quat else ACTION_RELATIVE_EE
        if rel_key not in dataset_stats:
            raise KeyError(
                f"action_mode='{action_mode}' needs '{rel_key}' stats. Re-run "
                "tools/convert_joints_to_eepose.py to (re)generate them."
            )
        dataset_stats[ACTION] = dataset_stats[rel_key]

    return dataset_stats


def make_ee_relative_steps(config):
    """Build the paired (relative, absolute) action steps for a policy processor.

    In EE mode (action_mode='rot6d' or 'quat') they run in SE(3) ``pose`` mode with the
    correct rotation format; otherwise they fall back to the joint element-wise behaviour
    gated by the pi05-only ``use_relative_actions`` flag (no-op for act/diffusion).
    """
    action_mode = getattr(config, "action_mode", "joint")
    from .sensor_routing import _ACTION_MODE_ALIASES
    action_mode = _ACTION_MODE_ALIASES.get(action_mode, action_mode)

    action_ee = action_mode in ("rot6d", "quat")
    enabled   = getattr(config, "use_relative_actions", False) or action_ee
    rot_mode  = "quat" if action_mode == "quat" else "rot6d"

    relative_step = RelativeActionsProcessorStep(
        enabled=enabled,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
        mode="pose" if action_ee else "joint",
        n_arms=getattr(config, "ee_num_arms", 2),
        rot_mode=rot_mode,
    )
    absolute_step = AbsoluteActionsProcessorStep(enabled=enabled, relative_step=relative_step)
    return relative_step, absolute_step
