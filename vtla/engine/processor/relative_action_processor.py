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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from vtla.engine.configs import PipelineFeatureType, PolicyFeature
from vtla.engine.types import EnvTransition, TransitionKey
from vtla.engine.utils.constants import ACTION, OBS_STATE
from vtla.engine.utils.ee_transforms import ee_to_absolute, ee_to_relative

OBS_STATE_EPISODE_EE    = OBS_STATE + "_episode_ee"
OBS_STATE_EPISODE_JOINT = OBS_STATE + "_episode_joint"
ACTION_EPISODE_EE       = ACTION + "_episode_ee"
OBS_STATE_ABSOLUTE_EE   = OBS_STATE + "_absolute_ee"
ACTION_ABSOLUTE_EE      = ACTION + "_absolute_ee"
# quat variants (new)
OBS_STATE_EPISODE_QUAT  = OBS_STATE + "_episode_quat"
ACTION_EPISODE_QUAT     = ACTION + "_episode_quat"
OBS_STATE_ABSOLUTE_QUAT = OBS_STATE + "_absolute_quat"
ACTION_ABSOLUTE_QUAT    = ACTION + "_absolute_quat"

# Backward-compat aliases (same as in sensor_routing.py).
_STATE_ALIASES = {
    "joint": "absolute_joint", "episode_ee": "episode_rot6d", "absolute_ee": "absolute_rot6d",
}
_ACTION_ALIASES = {
    "joint": "absolute_joint", "rot6d": "relative_rot6d", "quat": "relative_quat",
    "relative_ee": "relative_rot6d",
}
ACTION_ANCHOR = OBS_STATE + "_action_anchor"


def route_ee_batch(batch: dict, state_mode: str, action_mode: str) -> dict:
    """Select EE columns as the canonical ``observation.state`` / ``action`` (in place).

    Supports the current naming (``episode_rot6d`` / ``absolute_rot6d`` / ``episode_quat`` /
    ``absolute_quat`` for state; ``rot6d`` / ``quat`` for action) as well as the legacy names
    (``episode_ee`` / ``absolute_ee`` / ``relative_ee``). Mutates and returns ``batch``.
    """
    # Normalise legacy aliases.
    state_mode  = _STATE_ALIASES.get(state_mode,  state_mode)
    action_mode = _ACTION_ALIASES.get(action_mode, action_mode)

    def _route(dst: str, src: str) -> None:
        if src in batch:
            batch[dst] = batch.pop(src)
            if src + "_is_pad" in batch:
                batch[dst + "_is_pad"] = batch.pop(src + "_is_pad")

    # Preserve an action anchor independently of the state presented to the model. EE anchors are
    # always absolute so postprocessing directly recovers a base-frame robot target.
    action_reference, action_representation = action_mode.split("_", 1)
    if action_reference == "relative":
        if action_representation == "joint" and OBS_STATE in batch:
            batch[ACTION_ANCHOR] = batch[OBS_STATE]
        elif action_representation == "rot6d" and OBS_STATE_ABSOLUTE_EE in batch:
            batch[ACTION_ANCHOR] = batch[OBS_STATE_ABSOLUTE_EE]
        elif action_representation == "quat" and OBS_STATE_ABSOLUTE_QUAT in batch:
            batch[ACTION_ANCHOR] = batch[OBS_STATE_ABSOLUTE_QUAT]
        if ACTION_ANCHOR not in batch:
            raise KeyError(
                f"action_mode='{action_mode}' requires a current '{action_representation}' anchor; "
                "re-run the dataset preprocessing script to generate the required state columns."
            )

    # State routing.
    if state_mode == "none":
        batch.pop(OBS_STATE, None)
    elif state_mode == "episode_joint":
        _route(OBS_STATE, OBS_STATE_EPISODE_JOINT)
    if state_mode == "episode_rot6d":
        _route(OBS_STATE, OBS_STATE_EPISODE_EE)
    elif state_mode == "absolute_rot6d":
        _route(OBS_STATE, OBS_STATE_ABSOLUTE_EE)
    elif state_mode == "episode_quat":
        _route(OBS_STATE, OBS_STATE_EPISODE_QUAT)
    elif state_mode == "absolute_quat":
        _route(OBS_STATE, OBS_STATE_ABSOLUTE_QUAT)

    # Action source selection is independent of state reference and action reference. Relative
    # encoding is performed by RelativeActionsProcessorStep after this routing step.
    if action_representation == "rot6d":
        _route(ACTION, ACTION_ABSOLUTE_EE)
    elif action_representation == "quat":
        _route(ACTION, ACTION_ABSOLUTE_QUAT)

    return batch

from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .pipeline import ProcessorStep, ProcessorStepRegistry

# Re-export for backward compatibility
__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "to_relative_actions",
    "to_absolute_actions",
]


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert absolute actions to relative: relative = action - state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert relative actions back to absolute: absolute = relative + state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


@ProcessorStepRegistry.register("delta_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to relative actions (action -= state) for masked dimensions.

    Mirrors OpenPI's DeltaActions transform. Applied during preprocessing so the model
    trains on relative offsets instead of absolute positions.
    Caches the last seen state so a paired AbsoluteActionsProcessorStep can reverse
    the conversion during postprocessing.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint names to keep absolute (not converted to relative).
        action_names: Action dimension names from dataset metadata, used to build
            the mask from exclude_joints. If None, all dims are converted.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    # mode="joint": element-wise action-=state (exclude_joints masked).
    # mode="pose":  SE(3) per-arm relative EE (St^-1 . action), gripper kept absolute. Used by
    #               action_mode="rot6d"/"quat"; exclude_joints is ignored (gripper handled internally).
    mode: str = "joint"
    n_arms: int = 2
    # Rotation format for pose mode: "rot6d" (default) or "quat".
    rot_mode: str = "rot6d"
    state_key: str = ACTION_ANCHOR
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))

        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(self.state_key) if observation else None
        # Old serialized processors did not carry the hidden anchor key.
        if state is None and observation:
            state = observation.get(OBS_STATE)

        # In pose mode the reference is a single EE pose per sample. Policies with a multi-step
        # observation window (e.g. diffusion: state is (B, n_obs, D)) collapse to the most recent
        # frame (t=0) as the relativization anchor.
        if state is not None and state.ndim == 3:
            state = state[:, -1]

        # Always cache the (resolved) state for the paired AbsoluteActionsProcessorStep
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        if observation and self.state_key in observation:
            new_transition[TransitionKey.OBSERVATION] = dict(observation)
            new_transition[TransitionKey.OBSERVATION].pop(self.state_key, None)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        if self.mode == "pose":
            if state.device != action.device or state.dtype != action.dtype:
                state = state.to(device=action.device, dtype=action.dtype)
            new_transition[TransitionKey.ACTION] = ee_to_relative(
                state, action, n_arms=self.n_arms, rot_mode=self.rot_mode
            )
        else:
            mask = self._build_mask(action.shape[-1])
            new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask)
        return new_transition

    def get_cached_state(self) -> torch.Tensor | None:
        """Return the cached ``observation.state`` used as the reference point for relative/absolute action conversions."""
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
            "mode": self.mode,
            "n_arms": self.n_arms,
            "rot_mode": self.rot_mode,
            "state_key": self.state_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Converts relative actions back to absolute actions (action += state) for all dimensions.

    Mirrors OpenPI's AbsoluteActions transform. Applied during postprocessing so
    predicted relative offsets are converted back to absolute positions for execution.
    Reads the cached state from its paired RelativeActionsProcessorStep.

    Attributes:
        enabled: Whether to apply the absolute conversion.
        relative_step: Reference to the paired RelativeActionsProcessorStep that caches state.
    """

    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires a paired RelativeActionsProcessorStep "
                "but relative_step is None. Ensure relative_step is set when constructing the postprocessor."
            )

        cached_state = self.relative_step.get_cached_state()
        if cached_state is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires state from RelativeActionsProcessorStep "
                "but no state has been cached. Ensure the preprocessor runs before the postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        if self.relative_step.mode == "pose":
            # Align cached_state to the action's device/dtype. The state is cached before
            # DeviceProcessorStep moves the transition, so it can be on CPU while the
            # unnormalized action is on CUDA.
            if cached_state.device != action.device or cached_state.dtype != action.dtype:
                cached_state = cached_state.to(device=action.device, dtype=action.dtype)
            new_transition[TransitionKey.ACTION] = ee_to_absolute(
                cached_state, action,
                n_arms=self.relative_step.n_arms,
                rot_mode=self.relative_step.rot_mode,
            )
        else:
            mask = self.relative_step._build_mask(action.shape[-1])
            new_transition[TransitionKey.ACTION] = to_absolute_actions(action, cached_state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
