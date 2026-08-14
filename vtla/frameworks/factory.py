#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, TypedDict

from typing_extensions import Unpack

import torch

if TYPE_CHECKING:
    from vtla.datasets.dataset_metadata import LeRobotDatasetMetadata

from vtla.engine.configs import FeatureType, PreTrainedConfig
from vtla.engine.processor import (
    AbsoluteActionsProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from vtla.engine.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from vtla.engine.utils.feature_utils import dataset_to_policy_features

from .act.configuration_act import ACTConfig
from .diffusion.configuration_diffusion import DiffusionConfig
from .fastwam.configuration_fastwam import FastWAMConfig
from .pi05.configuration_pi05 import PI05Config
from .pretrained import PreTrainedPolicy
from .starvla_groot.configuration_starvla_groot import StarvlaGrootConfig
from .utils import validate_visual_features_consistency


def _reconnect_relative_absolute_steps(
    preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline
) -> None:
    """Wire AbsoluteActionsProcessorStep.relative_step to the RelativeActionsProcessorStep after deserialization.

    After a policy is loaded from disk, the preprocessor and postprocessor are reconstructed
    independently from their configs. AbsoluteActionsProcessorStep needs a live reference to
    the RelativeActionsProcessorStep so it can read the cached state at inference time.
    That reference is not serializable, so we re-establish it here after loading.
    """
    relative_step = next((s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)), None)
    if relative_step is None:
        return
    for step in postprocessor.steps:
        if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
            step.relative_step = relative_step


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """
    Retrieves a policy class by its registered name.

    This function uses dynamic imports to avoid loading all policy classes into memory
    at once, improving startup time and reducing dependencies.

    Args:
        name: The name of the policy. Supported names are "tdmpc", "diffusion", "act",
            "multi_task_dit", "vqbet", "pi0", "pi05", "gaussian_actor", "smolvla", "wall_x".
    Returns:
        The policy class corresponding to the given name.

    Raises:
        NotImplementedError: If the policy name is not recognized.
    """
    if name == "act":
        from .act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "diffusion":
        from .diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "fastwam":
        from .fastwam.modeling_fastwam import FastWAMPolicy

        return FastWAMPolicy
    elif name == "pi05":
        from .pi05.modeling_pi05 import PI05Policy

        return PI05Policy
    elif name == "starvla_groot":
        from .starvla_groot.modeling_starvla_groot import StarvlaGrootPolicy

        return StarvlaGrootPolicy
    else:
        try:
            return _get_policy_cls_from_policy_name(name=name)
        except Exception as e:
            raise ValueError(f"Policy type '{name}' is not available.") from e


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """
    Instantiates a policy configuration object based on the policy type.

    This factory function simplifies the creation of policy configuration objects by
    mapping a string identifier to the corresponding config class.

    Args:
        policy_type: The type of the policy. Supported types include "tdmpc",
                     "multi_task_dit", "diffusion", "act", "vqbet", "pi0", "pi05", "gaussian_actor",
                     "smolvla", "wall_x".
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        An instance of a `PreTrainedConfig` subclass.

    Raises:
        ValueError: If the `policy_type` is not recognized.
    """
    if policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "fastwam":
        return FastWAMConfig(**kwargs)
    elif policy_type == "pi05":
        return PI05Config(**kwargs)
    elif policy_type == "starvla_groot":
        return StarvlaGrootConfig(**kwargs)
    else:
        try:
            config_cls = PreTrainedConfig.get_choice_class(policy_type)
            return config_cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Policy type '{policy_type}' is not available.") from e


class ProcessorConfigKwargs(TypedDict, total=False):
    """
    A TypedDict defining the keyword arguments for processor configuration.

    This provides type hints for the optional arguments passed to `make_pre_post_processors`,
    improving code clarity and enabling static analysis.

    Attributes:
        preprocessor_config_filename: The filename for the preprocessor configuration.
        postprocessor_config_filename: The filename for the postprocessor configuration.
        preprocessor_overrides: A dictionary of overrides for the preprocessor configuration.
        postprocessor_overrides: A dictionary of overrides for the postprocessor configuration.
        dataset_stats: Dataset statistics for normalization.
    """

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create or load pre- and post-processor pipelines for a given policy.

    This function acts as a factory. It can either load existing processor pipelines
    from a pretrained path or create new ones from scratch based on the policy
    configuration. Each policy type has a dedicated factory function for its
    processors (e.g., `make_tdmpc_pre_post_processors`).

    Args:
        policy_cfg: The configuration of the policy for which to create processors.
        pretrained_path: An optional path to load pretrained processor pipelines from.
            If provided, pipelines are loaded from this path.
        **kwargs: Keyword arguments for processor configuration, as defined in
            `ProcessorConfigKwargs`.

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    Raises:
        NotImplementedError: If a processor factory is not implemented for the given
            policy configuration type.
    """
    if pretrained_path:
        preprocessor_overrides = {
            key: dict(value) for key, value in (kwargs.get("preprocessor_overrides") or {}).items()
        }
        if isinstance(policy_cfg, FastWAMConfig) and policy_cfg.load_text_encoder:
            fastwam_overrides = preprocessor_overrides.setdefault("fastwam_prepare_batch", {})
            fastwam_overrides["use_text_cache"] = False

        # Custom ProcessorStep subclasses register themselves via
        # @ProcessorStepRegistry.register as an import side-effect. When loading from a
        # pretrained path we return early below (never reaching the make_*_pre_post_processors
        # imports further down), so the policy-specific processor module must be imported here
        # to populate the registry before from_pretrained resolves step classes by name.
        if isinstance(policy_cfg, ACTConfig):
            from .act import processor_act  # noqa: F401
        elif isinstance(policy_cfg, DiffusionConfig):
            from .diffusion import processor_diffusion  # noqa: F401
        elif isinstance(policy_cfg, FastWAMConfig):
            from .fastwam import processor_fastwam  # noqa: F401
        elif isinstance(policy_cfg, PI05Config):
            from .pi05 import processor_pi05  # noqa: F401
        elif isinstance(policy_cfg, StarvlaGrootConfig):
            from .starvla_groot import processor_starvla_groot  # noqa: F401

        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
            ),
            overrides=preprocessor_overrides,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
            ),
            overrides=kwargs.get("postprocessor_overrides", {}),
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        _reconnect_relative_absolute_steps(preprocessor, postprocessor)

        # For EE state modes at inference the dataset does not supply the pre-computed EE column;
        # prepend a step that converts joint angles to the EE pose on-the-fly so the model receives
        # the right input. episode_ee -> pose relative to the episode first frame; absolute_ee ->
        # raw base-frame FK (no baseline).
        state_mode = getattr(policy_cfg, "state_mode", None)
        episode_modes = ("episode_rot6d", "episode_quat", "episode_ee")
        absolute_modes = ("absolute_rot6d", "absolute_quat", "absolute_ee")
        if state_mode in (*episode_modes, *absolute_modes):
            from .episode_ee_processor import EpisodeEEPreprocessorStep

            state_names = getattr(policy_cfg, "state_feature_names", None) or []
            relative = state_mode in episode_modes
            rot_mode = "quat" if state_mode in ("episode_quat", "absolute_quat") else "rot6d"
            n_arms = getattr(policy_cfg, "ee_num_arms", 2)
            ee_step = EpisodeEEPreprocessorStep(
                state_feature_names=state_names,
                relative_to_baseline=relative,
                rot_mode=rot_mode,
                n_arms=n_arms,
            )
            preprocessor.steps = [ee_step, *preprocessor.steps]

            # episode_ee + action_mode='relative_ee': the postprocessor (AbsoluteActionsProcessorStep)
            # only recovers the action relative to the episode first frame (S_{t+k}). Append a step
            # that lifts it to the world frame (A_{t+k} = A0 · S_{t+k}) using A0 cached by ee_step.
            # absolute_ee needs NO such lift: the relative step's cached anchor is the absolute pose
            # Tt, so ee_to_absolute(Tt, a_rel) = T_{t+k} is already the world flange pose.
            if relative and getattr(policy_cfg, "action_mode", None) in ("rot6d", "quat", "relative_ee"):
                from vtla.engine.processor import EpisodeEEToWorldStep

                world_step = EpisodeEEToWorldStep(
                    n_arms=n_arms,
                    rot_mode=rot_mode,
                    ee_step=ee_step,
                )
                postprocessor.steps = [*postprocessor.steps, world_step]

        return preprocessor, postprocessor

    # Create a new processor based on policy type
    if isinstance(policy_cfg, ACTConfig):
        from .act.processor_act import make_act_pre_post_processors

        processors = make_act_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, DiffusionConfig):
        from .diffusion.processor_diffusion import make_diffusion_pre_post_processors

        processors = make_diffusion_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, FastWAMConfig):
        from .fastwam.processor_fastwam import make_fastwam_pre_post_processors

        processors = make_fastwam_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, PI05Config):
        from .pi05.processor_pi05 import make_pi05_pre_post_processors

        processors = make_pi05_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, StarvlaGrootConfig):
        from .starvla_groot.processor_starvla_groot import make_starvla_groot_pre_post_processors

        processors = make_starvla_groot_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    else:
        try:
            processors = _make_processors_from_policy_config(
                config=policy_cfg,
                dataset_stats=kwargs.get("dataset_stats"),
            )
        except Exception as e:
            raise ValueError(f"Processor for policy type '{policy_cfg.type}' is not implemented.") from e

    return processors


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    rename_map: dict[str, str] | None = None,
    for_training: bool = False,
) -> PreTrainedPolicy:
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created. If `cfg.pretrained_path` is
             set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        env_cfg: Environment configuration used to infer feature shapes and types.
                 One of `ds_meta` or `env_cfg` must be provided.
        rename_map: Optional mapping of dataset or environment feature keys to match
                 expected policy feature names (e.g., `"left"` → `"camera1"`).
        for_training: Whether the policy is being constructed by the training entrypoint.
            FastWAM uses cached text contexts during training and online T5 encoding otherwise.

    Returns:
        An instantiated and device-placed policy model.

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided.
        NotImplementedError: If attempting to use an unsupported policy-backend
                             combination (e.g., VQBeT with 'mps').
    """
    if ds_meta is None:
        raise ValueError("VTLA SFT policy training requires dataset metadata.")

    policy_cls = get_policy_class(cfg.type)
    if isinstance(cfg, FastWAMConfig):
        cfg.load_text_encoder = not for_training

    kwargs = {}
    features = dataset_to_policy_features(ds_meta.features)

    # Build input/output features from the dataset meta. Visual (camera) features ALWAYS come from
    # the dataset so the configured camera keys match what the data loader actually provides, and so
    # the EE action columns (action_episode_ee / action_absolute_ee) survive for apply_action_mode.
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not cfg.input_features:
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    if cfg.pretrained_path:
        # A pretrained checkpoint may have been trained with a different proprioceptive (STATE) dim
        # than the raw dataset (e.g. OpenPI's 32-dim padded state, or a 20-dim EE state). Pull only
        # the NON-VISUAL feature shapes from the ckpt config so the model is built to match the
        # checkpoint weights. Visual features are deliberately NOT taken from the ckpt: its camera
        # names may differ (e.g. OpenPI 'base_0_rgb' vs this dataset's 'left_cam_wrist'), and
        # overwriting them would prune every dataset camera (the bug that broke pi05 fine-tuning).
        # For EE state modes apply_state_mode then re-routes observation.state to the dataset's
        # 20-dim EE column anyway, so this only matters for joint/none modes.
        from vtla.engine.configs.policies import PreTrainedConfig as _BaseCfg  # noqa: PLC0415
        _ckpt_cfg = _BaseCfg.from_pretrained(cfg.pretrained_path)
        for key, ft in (_ckpt_cfg.input_features or {}).items():
            if ft.type is not FeatureType.VISUAL:
                cfg.input_features[key] = ft

    # Store action feature names for relative_exclude_joints support
    if ds_meta is not None and hasattr(cfg, "action_feature_names"):
        action_names = ds_meta.features.get(ACTION, {}).get("names")
        if action_names is not None:
            cfg.action_feature_names = list(action_names)

    # Store state feature names so EpisodeEEPreprocessorStep can locate joint/gripper indices.
    # Always read from ds_meta (raw joint names), regardless of pretrained path.
    if ds_meta is not None and hasattr(cfg, "state_feature_names"):
        state_names = ds_meta.features.get(OBS_STATE, {}).get("names")
        if state_names is not None:
            cfg.state_feature_names = list(state_names)

    kwargs["config"] = cfg

    # Pass dataset_stats to the policy if available (needed for some policies like SARM)
    if ds_meta is not None and hasattr(ds_meta, "stats"):
        kwargs["dataset_stats"] = ds_meta.stats

    if ds_meta is not None:
        kwargs["dataset_meta"] = ds_meta

    if not cfg.pretrained_path and cfg.use_peft:
        raise ValueError(
            "Instantiating a policy with `use_peft=True` without a checkpoint is not supported since that requires "
            "the PEFT config parameters to be set. For training with PEFT, see `lerobot_train.py` on how to do that."
        )

    if cfg.pretrained_path and not cfg.use_peft:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    elif cfg.pretrained_path and cfg.use_peft:
        # Load a pretrained PEFT model on top of the policy. The pretrained path points to the folder/repo
        # of the adapter and the adapter's config contains the path to the base policy. So we need the
        # adapter config first, then load the correct policy and then apply PEFT.
        from peft import PeftConfig, PeftModel

        logging.info("Loading policy's PEFT adapter.")

        peft_pretrained_path = str(cfg.pretrained_path)
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        kwargs["pretrained_name_or_path"] = peft_config.base_model_name_or_path
        if not kwargs["pretrained_name_or_path"]:
            # This means that there's a bug or we trained a policy from scratch using PEFT.
            # It is more likely that this is a bug so we'll raise an error.
            raise ValueError(
                "No pretrained model name found in adapter config. Can't instantiate the pre-trained policy on which "
                "the adapter was trained."
            )

        policy = policy_cls.from_pretrained(**kwargs)
        policy = PeftModel.from_pretrained(
            policy, peft_pretrained_path, config=peft_config, is_trainable=True
        )

    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    if not rename_map:
        validate_visual_features_consistency(cfg, features)
        # TODO: (jadechoghari) - add a check_state(cfg, features) and check_action(cfg, features)

    return policy


def _get_policy_cls_from_policy_name(name: str) -> type[PreTrainedConfig]:
    """Get policy class from its registered name using dynamic imports.

    This is used as a helper function to import policies from 3rd party lerobot plugins.

    Args:
        name: The name of the policy.
    Returns:
        The policy class corresponding to the given name.
    """
    if name not in PreTrainedConfig.get_known_choices():
        raise ValueError(
            f"Unknown policy name '{name}'. Available policies: {PreTrainedConfig.get_known_choices()}"
        )

    config_cls = PreTrainedConfig.get_choice_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix("Config")  # e.g., DiffusionConfig -> Diffusion
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention."
            f"Make sure it ends with 'Config'!"
        )
    cls_name = model_name + "Policy"  # e.g., DiffusionConfig -> DiffusionPolicy
    module_path = config_cls.__module__.replace(
        "configuration_", "modeling_"
    )  # e.g., configuration_diffusion -> modeling_diffusion

    module = importlib.import_module(module_path)
    policy_cls = getattr(module, cls_name)
    return policy_cls


def _make_processors_from_policy_config(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """Create pre- and post-processors from a policy configuration using dynamic imports.

    This is used as a helper function to import processor factories from 3rd party lerobot plugins.

    Args:
        config: The policy configuration object.
        dataset_stats: Dataset statistics for normalization.
    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.
    """

    policy_type = config.type
    function_name = f"make_{policy_type}_pre_post_processors"
    module_path = config.__class__.__module__.replace(
        "configuration_", "processor_"
    )  # e.g., configuration_diffusion -> processor_diffusion
    logging.debug(
        f"Instantiating pre/post processors using function '{function_name}' from module '{module_path}'"
    )
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function(config, dataset_stats=dataset_stats)
