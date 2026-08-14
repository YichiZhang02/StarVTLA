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
"""Inference-time processor that assembles tactile temporal windows.

At *training* time the LeRobot dataset delivers tactile keys as windowed
``[B, F, C, H, W]`` tensors directly (the factory's per-key delta_timestamps
handles this).  At *inference* time the robot provides one frame per camera per
step, so this processor maintains a per-key deque of the required depth and
stacks frames into the ``[1, F, C, H, W]`` shape the model expects.

Key behaviour:
- **No-op at training** (tensor already 5-D) — the step detects whether the
  input is already windowed and passes it through unchanged.
- **No-op when F == 1** — single-frame mode; the step is a pure pass-through.
- **Episode-safe** — call ``reset()`` when the robot resets between episodes to
  flush stale frames; the deque re-initialises by repeating the first frame.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from vtla.engine.configs import PipelineFeatureType, PolicyFeature
from vtla.engine.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from vtla.engine.types import TransitionKey


@dataclass
@ProcessorStepRegistry.register(name="tactile_temporal_window_step")
class TactileTemporalWindowStep(ProcessorStep):
    """Assemble a sliding tactile frame window at inference time.

    Parameters
    ----------
    tactile_keys:
        The tactile image keys to buffer (e.g.
        ``["observation.images.left_cam_finger0", ...]``).
    num_frames:
        Total frames in the window, including the current frame (``F``).
    frame_offset:
        Spacing in steps between consecutive frames in the window (``off``).
        The window samples indices ``[-(F-1)*off, ..., -off, 0]`` relative to
        the current step. A deque of depth ``(F-1)*off + 1`` is kept; only the
        F entries at the offset positions are stacked into the output tensor.
    """

    tactile_keys: list[str] = field(default_factory=list)
    num_frames: int = 1
    frame_offset: int = 1

    # runtime state — not part of the serialised config
    _history: dict[str, deque] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._history = {}

    # ------------------------------------------------------------------
    # ProcessorStep interface
    # ------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        return {
            "tactile_keys": list(self.tactile_keys),
            "num_frames": int(self.num_frames),
            "frame_offset": int(self.frame_offset),
        }

    def reset(self) -> None:
        """Flush all per-key frame histories (call on every robot episode reset)."""
        self._history.clear()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Feature shapes are unchanged (the step only stacks along a new leading dim)."""
        return features

    def __call__(self, transition):
        """Stack per-key tactile frame histories into windowed tensors.

        Accepts both the raw-dict and EnvTransition forms used by the pipeline.
        Modifies the observation dict in-place (shallow copy guard applied).
        """
        self._current_transition = transition
        # Support flat dict-of-tensors and the dict-backed EnvTransition representation.
        is_env_transition = isinstance(transition, dict) and TransitionKey.OBSERVATION in transition
        if is_env_transition:
            obs = dict(transition.get(TransitionKey.OBSERVATION) or {})
        elif isinstance(transition, dict):
            obs = transition
        else:
            obs = transition.observation

        if self.num_frames <= 1 or not self.tactile_keys:
            return transition

        # Determine required deque depth: (F-1)*off + 1 frames in history covers all
        # offset-sampled positions [-(F-1)*off, ..., 0].
        depth = (self.num_frames - 1) * self.frame_offset + 1

        updated: dict[str, Tensor] = {}
        for key in self.tactile_keys:
            if key not in obs:
                continue
            t = obs[key]
            # If the tensor is already 5-D the dataset has already assembled the window
            # (training path or test with synthetic data) — pass through unchanged.
            if t.dim() == 5:
                continue

            # t: [B, C, H, W] (inference, single frame per step)
            if key not in self._history:
                buf: deque = deque(maxlen=depth)
                # Pad with the first observation repeated so the window is immediately full.
                for _ in range(depth):
                    buf.append(t)
                self._history[key] = buf
            else:
                self._history[key].append(t)

            buf = self._history[key]
            # Sample the F frames at the offset positions (oldest to newest).
            # buf[-1] = current; buf[-(1+off)] = one step back; etc.
            frames: list[Tensor] = []
            for i in range(self.num_frames):
                # age = (F-1-i)*offset steps ago  →  index from end = 1 + age
                age = (self.num_frames - 1 - i) * self.frame_offset
                idx = -(1 + age)
                frames.append(buf[idx])

            updated[key] = torch.stack(frames, dim=1)  # [B, F, C, H, W]

        if not updated:
            return transition

        if is_env_transition:
            transition = {**transition, TransitionKey.OBSERVATION: {**obs, **updated}}
        elif isinstance(transition, dict):
            transition = {**transition, **updated}
        else:
            transition = transition._replace(observation={**obs, **updated})

        return transition
