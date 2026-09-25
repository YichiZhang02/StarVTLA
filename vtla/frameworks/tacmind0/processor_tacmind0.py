"""Use the infra's action/state processors; TacMind0 tokenizes in the policy."""

from typing import Any

from vtla.engine.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from vtla.engine.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from vtla.frameworks.ee_processor_utils import make_ee_relative_steps, remap_ee_dataset_stats
from vtla.frameworks.tactile_temporal_processor import TactileTemporalWindowStep


def make_tacmind0_pre_post_processors(config, dataset_stats=None):
    stats = remap_ee_dataset_stats(dataset_stats, config)
    relative, absolute = make_ee_relative_steps(config)
    features = {**config.normalizer_input_features(), **config.output_features}
    pre = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            TactileTemporalWindowStep(
                tactile_keys=list(config.tactile_keys), num_frames=8, frame_offset=5,
            ),
            relative,
            NormalizerProcessorStep(features=features, norm_map=config.normalization_mapping, stats=stats),
            DeviceProcessorStep(device=config.device),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    post = PolicyProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(features=config.output_features, norm_map=config.normalization_mapping, stats=stats),
            absolute,
            DeviceProcessorStep(device="cpu"),
        ],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return pre, post
