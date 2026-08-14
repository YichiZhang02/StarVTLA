"""Pre/post processors for the independently registered DINO-aligned policy."""

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

from .configuration_starvla_groot_dinoalign import StarvlaGrootDinoAlignConfig


def make_starvla_groot_dinoalign_pre_post_processors(
    config: StarvlaGrootDinoAlignConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    dataset_stats = remap_ee_dataset_stats(dataset_stats, config)
    relative_step, absolute_step = make_ee_relative_steps(config)
    processor_features = {**config.normalizer_input_features(), **config.output_features}

    tactile_window_step = TactileTemporalWindowStep(
        tactile_keys=list(config.tactile_windowed_keys()),
        num_frames=int(config.tactile_num_frames),
        frame_offset=int(config.tactile_frame_offset),
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
        PolicyProcessorPipeline(
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

