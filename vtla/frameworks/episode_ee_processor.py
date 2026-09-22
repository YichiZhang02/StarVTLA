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

"""Inference-only joint observation to TCP rot6d state and hidden action anchor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from deployment.robots import RobotConfig
from vtla.engine.configs import PipelineFeatureType, PolicyFeature
from vtla.engine.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from vtla.engine.processor.relative_action_processor import ACTION_ANCHOR
from vtla.engine.utils.constants import OBS_STATE
from vtla.engine.utils.ee_kinematics import (
    compute_baseline,
    joint_indices,
    make_realman_algo,
    to_absolute_ee,
    to_episode_ee,
)


@dataclass
@ProcessorStepRegistry.register(name="episode_ee_state_processor")
class EpisodeEEPreprocessorStep(ObservationProcessorStep):
    """Convert joint observations to absolute or episode-relative TCP rot6d states."""

    state_feature_names: list[str] = field(default_factory=list)
    relative_to_baseline: bool = True
    # Number of arms packed in the EE vector.
    n_arms: int = 2
    # Selects the physical RealMan B/ISF kinematics used when the dataset was processed.
    robot_type: str | None = None

    def __post_init__(self) -> None:
        force_type = RobotConfig.get_kinematics_force_type(self.robot_type)
        self._algo = make_realman_algo(force_type)
        self._jidx: dict = joint_indices(self.state_feature_names)
        self._flange_tcp_calibration = (
            {
                side: RobotConfig.get_flange_tcp_calibration(self.robot_type, side)
                for side in self._jidx["sides"]
            }
        )
        self._baseline: tuple | None = None   # ((R_p0, R_R0), (L_p0, L_R0))


    def reset(self) -> None:
        """Clear the episode-start baseline; called at the start of each episode."""
        self._baseline = None


    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Replace ``observation.state`` (joints) with the EE pose in the configured format.

        episode modes (``relative_to_baseline=True``):  T0^{-1}·Tt relative to first frame.
        absolute modes (``relative_to_baseline=False``): base-frame FK (Tt, no T0).
        """
        raw = observation.get(OBS_STATE)
        if raw is None:
            return observation

        if isinstance(raw, torch.Tensor):
            vec16 = raw.detach().cpu().numpy().astype(np.float64).flatten()
        else:
            vec16 = np.asarray(raw, dtype=np.float64).flatten()

        if not self.relative_to_baseline:
            ee_vec = to_absolute_ee(
                self._algo,
                vec16,
                self._jidx,


                flange_tcp_calibration=self._flange_tcp_calibration,
            )
            observation[OBS_STATE] = torch.from_numpy(ee_vec)
            return observation

        if self._baseline is None:
            self._baseline = compute_baseline(
                self._algo,
                vec16,
                self._jidx,

                flange_tcp_calibration=self._flange_tcp_calibration,
            )


        ee_vec = to_episode_ee(
            self._algo,
            vec16,
            self._jidx,
            self._baseline,


            flange_tcp_calibration=self._flange_tcp_calibration,
        )
        observation[OBS_STATE] = torch.from_numpy(ee_vec)
        return observation


    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Update the declared shape of ``observation.state`` from joints to EE dim."""
        from vtla.engine.configs import FeatureType

        ee_dim = self.n_arms * 10
        for bucket in features.values():
            if OBS_STATE in bucket:
                ft = bucket[OBS_STATE]
                if ft.type is FeatureType.STATE:
                    bucket[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(ee_dim,))
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "state_feature_names": self.state_feature_names,
            "relative_to_baseline": self.relative_to_baseline,
            "n_arms": self.n_arms,
            "robot_type": self.robot_type,
        }


@dataclass
@ProcessorStepRegistry.register(name="action_anchor_processor")
class ActionAnchorPreprocessorStep(ObservationProcessorStep):
    """Cache the current observation in the action representation without exposing it to the model."""

    state_feature_names: list[str] = field(default_factory=list)
    representation: str = "joint"
    n_arms: int = 2
    robot_type: str | None = None

    def __post_init__(self) -> None:
        needs_fk = self.representation in ("rot6d",)
        self._jidx = joint_indices(self.state_feature_names) if needs_fk else None
        force_type = RobotConfig.get_kinematics_force_type(self.robot_type) if needs_fk else None
        self._algo = make_realman_algo(force_type) if force_type is not None else None
        self._flange_tcp_calibration = (
            {
                side: RobotConfig.get_flange_tcp_calibration(self.robot_type, side)
                for side in self._jidx["sides"]
            }
            if needs_fk
            else None
        )

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        raw = observation.get(OBS_STATE)
        if raw is None:
            return observation
        if self.representation == "joint":
            observation[ACTION_ANCHOR] = raw.clone() if isinstance(raw, torch.Tensor) else np.array(raw, copy=True)
            return observation
        vec = (
            raw.detach().cpu().numpy().astype(np.float64).flatten()
            if isinstance(raw, torch.Tensor)
            else np.asarray(raw, dtype=np.float64).flatten()
        )
        anchor = to_absolute_ee(
            self._algo,
            vec,
            self._jidx,


            flange_tcp_calibration=self._flange_tcp_calibration,
        )
        observation[ACTION_ANCHOR] = torch.from_numpy(anchor)
        return observation

    def get_config(self) -> dict[str, Any]:
        return {
            "state_feature_names": self.state_feature_names,
            "representation": self.representation,
            "n_arms": self.n_arms,
            "robot_type": self.robot_type,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="joint_state_mode_processor")
class JointStateModePreprocessorStep(ObservationProcessorStep):
    """Apply state_mode=episode_joint or remove state for state_mode=none at inference."""

    state_feature_names: list[str] = field(default_factory=list)
    mode: str = "episode_joint"

    def __post_init__(self) -> None:
        self._baseline: torch.Tensor | None = None
        self._relative_mask = torch.tensor(
            ["gripper" not in str(name).lower() for name in self.state_feature_names], dtype=torch.bool
        )

    def reset(self) -> None:
        self._baseline = None

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        raw = observation.get(OBS_STATE)
        if self.mode == "none":
            observation.pop(OBS_STATE, None)
            return observation
        if raw is None:
            return observation
        value = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw)
        if self._baseline is None:
            self._baseline = value.detach().clone()
        out = value.clone()
        mask = self._relative_mask.to(device=out.device)
        if mask.numel() == out.shape[-1]:
            out[..., mask] -= self._baseline.to(device=out.device, dtype=out.dtype)[..., mask]
        observation[OBS_STATE] = out
        return observation

    def get_config(self) -> dict[str, Any]:
        return {"state_feature_names": self.state_feature_names, "mode": self.mode}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
