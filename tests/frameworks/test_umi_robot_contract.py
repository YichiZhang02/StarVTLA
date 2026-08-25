from types import SimpleNamespace

import pytest
import torch

from vtla.frameworks.act.configuration_act import ACTConfig
from vtla.frameworks.factory import make_policy
from vtla.frameworks.sensor_routing import (
    ACTION_ABSOLUTE_EE,
    ACTION_ABSOLUTE_QUAT,
    OBS_STATE_ABSOLUTE_EE,
    OBS_STATE_ABSOLUTE_QUAT,
    OBS_STATE_EPISODE_EE,
    OBS_STATE_EPISODE_QUAT,
)


class _StubPolicy(torch.nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config


def _feature(shape, names):
    return {"dtype": "float32", "shape": [shape], "names": names}


def _umi_metadata():
    rot6d_names = [
        f"{side}_{name}"
        for side in ("right", "left")
        for name in (
            "ee_x",
            "ee_y",
            "ee_z",
            "ee_rot6d_0",
            "ee_rot6d_1",
            "ee_rot6d_2",
            "ee_rot6d_3",
            "ee_rot6d_4",
            "ee_rot6d_5",
            "gripper",
        )
    ]
    quat_names = [
        f"{side}_{name}"
        for side in ("right", "left")
        for name in (
            "ee_x",
            "ee_y",
            "ee_z",
            "ee_qx",
            "ee_qy",
            "ee_qz",
            "ee_qw",
            "gripper",
        )
    ]
    features = {
        "observation.state": _feature(144, [f"field_{index}" for index in range(144)]),
        "action": _feature(111, [f"field_{index}" for index in range(111)]),
        OBS_STATE_EPISODE_EE: _feature(20, rot6d_names),
        OBS_STATE_ABSOLUTE_EE: _feature(20, rot6d_names),
        ACTION_ABSOLUTE_EE: _feature(20, rot6d_names),
        OBS_STATE_EPISODE_QUAT: _feature(16, quat_names),
        OBS_STATE_ABSOLUTE_QUAT: _feature(16, quat_names),
        ACTION_ABSOLUTE_QUAT: _feature(16, quat_names),
    }
    return SimpleNamespace(robot_type="umi", features=features, stats={})


def _config(state_mode="episode_rot6d", action_mode="relative_rot6d"):
    return ACTConfig(
        device="cpu",
        state_mode=state_mode,
        action_mode=action_mode,
        pretrained_backbone_weights=None,
    )


def test_umi_training_keeps_generic_robot_type_and_detects_arm_count(monkeypatch):
    monkeypatch.setattr("vtla.frameworks.factory.get_policy_class", lambda _: _StubPolicy)
    config = _config()

    policy = make_policy(
        config,
        ds_meta=_umi_metadata(),
        rename_map={"unused": "unused"},
        for_training=True,
    )

    assert policy.config.robot_type == "umi"
    assert policy.config.ee_num_arms == 2


@pytest.mark.parametrize(
    ("state_mode", "action_mode", "message"),
    [
        ("absolute_joint", "relative_rot6d", "does not provide joint state"),
        ("none", "absolute_joint", "only supports EE actions"),
    ],
)
def test_umi_training_rejects_joint_contracts(
    monkeypatch, state_mode, action_mode, message
):
    monkeypatch.setattr("vtla.frameworks.factory.get_policy_class", lambda _: _StubPolicy)

    with pytest.raises(ValueError, match=message):
        make_policy(
            _config(state_mode, action_mode),
            ds_meta=_umi_metadata(),
            rename_map={"unused": "unused"},
            for_training=True,
        )
