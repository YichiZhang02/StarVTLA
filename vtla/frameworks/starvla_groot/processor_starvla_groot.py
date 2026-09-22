#!/usr/bin/env python

# Copyright 2025 starVLA community & The HuggingFace Inc. team. All rights reserved.
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
"""Pre/post-processor pipelines for the StarvlaGroot (QwenGR00T) policy.

Unlike PI05 there is no tokenizer step: the Qwen processor tokenizes images and
language inside the policy via ``apply_chat_template``. The raw ``task`` string
is carried through complementary data and read by the policy at forward time.
State and action are normalized here (dataset stats) and the action is
un-normalized in the post-processor.
"""

from typing import Any

import torch

from vtla.engine.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from vtla.engine.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from vtla.frameworks.ee_processor_utils import make_ee_relative_steps, remap_ee_dataset_stats
from vtla.frameworks.tactile_temporal_processor import TactileTemporalWindowStep

from .configuration_starvla_groot import StarvlaGrootConfig


def make_starvla_groot_pre_post_processors(
    config: StarvlaGrootConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    # EE modes (TCP state / action_mode=relative_rot6d): remap stats to canonical keys and
    # convert action to/from the TCP-local zero-centered action. No-op for joint modes.
    dataset_stats = remap_ee_dataset_stats(dataset_stats, config)
    relative_step, absolute_step = make_ee_relative_steps(config)

    processor_features = {**config.normalizer_input_features(), **config.output_features}

    # Tactile temporal window step (inference-time buffering). No-op for F == 1.
    tactile_window_step = TactileTemporalWindowStep(
        tactile_keys=list(getattr(config, "tactile_windowed_keys", lambda: [])()),
        num_frames=int(getattr(config, "tactile_num_frames", 1)),
        frame_offset=int(getattr(config, "tactile_frame_offset", 1)),
    )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        tactile_window_step,
        relative_step,
        NormalizerProcessorStep(
            features=processor_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        absolute_step,
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
