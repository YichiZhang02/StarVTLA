#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch

from vtla.engine.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from vtla.engine.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from vtla.frameworks.ee_processor_utils import make_ee_relative_steps, remap_ee_dataset_stats
from vtla.frameworks.tactile_temporal_processor import TactileTemporalWindowStep

from .configuration_act import ACTConfig


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    # EE modes (TCP state / action_mode=relative_rot6d): remap stats to the canonical keys
    # and convert action to/from the TCP-local zero-centered action (raw -> relative -> normalize -> model ->
    # unnormalize -> absolute). No-op for joint modes.
    dataset_stats = remap_ee_dataset_stats(dataset_stats, config)
    relative_step, absolute_step = make_ee_relative_steps(config)

    # Tactile temporal window: buffer F frames at inference and stack into [B, F, C, H, W].
    # No-op at training (dataset already delivers windowed tensors) and when F == 1.
    tactile_window_step = TactileTemporalWindowStep(
        tactile_keys=list(getattr(config, "tactile_windowed_keys", lambda: [])()),
        num_frames=int(getattr(config, "tactile_num_frames", 1)),
        frame_offset=int(getattr(config, "tactile_frame_offset", 1)),
    )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        tactile_window_step,
        relative_step,
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.normalizer_input_features(), **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
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
