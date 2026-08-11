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

"""Inference-only postprocessor step: episode-relative EE action -> world-frame EE action.

When a policy is trained with ``action_mode='relative_ee'`` the postprocessor chain ends with
``AbsoluteActionsProcessorStep`` (pose mode), which recovers the action expressed relative to the
episode's FIRST frame (``S_{t+k} = T0^{-1}·T_{t+k}``). To command the robot we still need the
absolute world pose ``A_{t+k} = A0 · S_{t+k}``, where ``A0`` is the world-frame EE pose at episode
start.

This step performs that final ``ee_to_absolute(A0, action)`` lift. It reads ``A0`` live from the
paired :class:`EpisodeEEPreprocessorStep` (which computes and caches it from the first joint
observation of each episode), so the two stay in lock-step across ``reset()`` at episode boundaries.

It is injected by ``make_pre_post_processors`` (inference path only, see
``vtla/frameworks/factory.py``) and is never serialized into a checkpoint — the live reference to
the preprocessor step is not persistable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vtla.engine.configs import PipelineFeatureType, PolicyFeature
from vtla.engine.types import PolicyAction
from vtla.engine.utils.ee_transforms import ee_to_absolute

from .pipeline import ActionProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="episode_ee_to_world_processor")
class EpisodeEEToWorldStep(ActionProcessorStep):
    """Lift an episode-relative EE action into the world frame via ``A_{t+k} = A0 · S_{t+k}``.

    Args:
        n_arms: Number of arms packed in the EE vector (2 for the dual-arm robot).
        ee_step: The paired :class:`EpisodeEEPreprocessorStep` that caches the episode-start
            world EE pose ``A0`` (read at call time via ``get_baseline_ee``). Not serialized;
            re-supplied each time the inference processors are built.
    """

    n_arms: int = 2
    # Rotation format matching the trained policy: "rot6d" (default) or "quat".
    rot_mode: str = "rot6d"
    ee_step: Any | None = field(default=None, repr=False)

    def action(self, action: PolicyAction) -> PolicyAction:
        if self.ee_step is None:
            raise RuntimeError(
                "EpisodeEEToWorldStep requires a paired EpisodeEEPreprocessorStep but ee_step is "
                "None. Ensure it is wired when the inference postprocessor is constructed."
            )
        a0 = self.ee_step.get_baseline_ee()
        if a0 is None:
            raise RuntimeError(
                "EpisodeEEToWorldStep has no episode-start baseline A0 yet. The preprocessor "
                "(EpisodeEEPreprocessorStep) must run on an observation before the postprocessor."
            )
        if a0.device != action.device or a0.dtype != action.dtype:
            a0 = a0.to(device=action.device, dtype=action.dtype)
        return ee_to_absolute(a0, action, n_arms=self.n_arms, rot_mode=self.rot_mode)

    def get_config(self) -> dict[str, Any]:
        return {"n_arms": self.n_arms, "rot_mode": self.rot_mode}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
