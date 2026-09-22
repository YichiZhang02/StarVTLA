"""Shared normalization and action wiring for TCP rot6d and joint policies."""
from vtla.engine.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from vtla.engine.utils.action_modes import parse_action_mode
from vtla.engine.utils.constants import ACTION, OBS_STATE
from .sensor_routing import (
    ACTION_ABSOLUTE_EE, ACTION_RELATIVE_EE, OBS_STATE_ABSOLUTE_EE,
    OBS_STATE_EPISODE_EE, OBS_STATE_EPISODE_JOINT,
)


def remap_ee_dataset_stats(dataset_stats, config):
    if dataset_stats is None:
        return None
    stats = dict(dataset_stats)
    state_key = {
        "episode_joint": OBS_STATE_EPISODE_JOINT,
        "episode_rot6d": OBS_STATE_EPISODE_EE,
        "absolute_rot6d": OBS_STATE_ABSOLUTE_EE,
    }.get(config.state_mode)
    if state_key is not None and state_key in stats:
        stats[OBS_STATE] = stats[state_key]
    reference, representation = parse_action_mode(config.action_mode)
    key = ACTION
    if representation == "rot6d":
        key = ACTION_RELATIVE_EE if reference == "relative" else ACTION_ABSOLUTE_EE
    elif reference == "relative":
        key = ACTION + "_relative_joint"
    if key != ACTION:
        if key not in stats:
            raise KeyError(f"action_mode={config.action_mode!r} needs {key!r} statistics; "
                           "run the current dataset conversion or TCP migration tool.")
        stats[ACTION] = stats[key]
    return stats


def make_ee_relative_steps(config):
    reference, representation = parse_action_mode(config.action_mode)
    enabled = reference == "relative"
    relative_step = RelativeActionsProcessorStep(
        enabled=enabled,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
        mode="pose" if representation == "rot6d" else "joint",
        n_arms=getattr(config, "ee_num_arms", 2),
    )
    return relative_step, AbsoluteActionsProcessorStep(enabled=enabled, relative_step=relative_step)
