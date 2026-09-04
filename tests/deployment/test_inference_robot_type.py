from types import SimpleNamespace

import pytest

from deployment.inference import _resolve_robot_type
from deployment.robots import RmBaseUmiDualConfig, RmIsfUmiLeftConfig, RmIsfUmiRightConfig


def _cfg(checkpoint_type: str, robot, ee_num_arms: int):
    return SimpleNamespace(
        policy=SimpleNamespace(robot_type=checkpoint_type, ee_num_arms=ee_num_arms),
        robot=robot,
    )


def test_umi_checkpoint_requires_explicit_robot_type(monkeypatch):
    monkeypatch.setattr("sys.argv", ["deployment.inference"])
    cfg = _cfg("umi", RmBaseUmiDualConfig(), ee_num_arms=2)

    with pytest.raises(ValueError, match="requires an explicit concrete robot"):
        _resolve_robot_type(cfg)


def test_umi_checkpoint_uses_cli_robot_type_at_runtime(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["deployment.inference", "--robot.type=rm_isf_umi_left"]
    )
    cfg = _cfg("umi", RmBaseUmiDualConfig(), ee_num_arms=1)

    _resolve_robot_type(cfg)

    assert isinstance(cfg.robot, RmIsfUmiLeftConfig)
    assert cfg.policy.robot_type == "rm_isf_umi_left"
    assert cfg.policy.original_checkpoint_robot_type == "umi"
    assert cfg.policy.ee_frame == "tcp"
    assert cfg.robot.ee_frame == "tcp"


def test_concrete_checkpoint_ignores_cli_robot_type(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["deployment.inference", "--robot.type=rm_base_umi_dual"]
    )
    cfg = _cfg("rm_isf_umi_right", RmBaseUmiDualConfig(), ee_num_arms=1)

    _resolve_robot_type(cfg)

    assert isinstance(cfg.robot, RmIsfUmiRightConfig)
    assert cfg.policy.robot_type == "rm_isf_umi_right"
    assert cfg.policy.original_checkpoint_robot_type == "rm_isf_umi_right"
    assert cfg.policy.ee_frame == "flange"
    assert cfg.robot.ee_frame == "flange"


def test_explicit_checkpoint_ee_frame_overrides_auto_rule(monkeypatch):
    monkeypatch.setattr("sys.argv", ["deployment.inference"])
    cfg = _cfg("rm_isf_umi_left", RmIsfUmiLeftConfig(), ee_num_arms=1)
    cfg.policy.ee_frame = "tcp"

    _resolve_robot_type(cfg)

    assert cfg.policy.ee_frame == "tcp"
    assert cfg.robot.ee_frame == "tcp"


def test_umi_checkpoint_rejects_incompatible_arm_count(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["deployment.inference", "--robot.type=rm_isf_umi_left"]
    )
    cfg = _cfg("umi", RmBaseUmiDualConfig(), ee_num_arms=2)

    with pytest.raises(ValueError, match="expects 2 arm"):
        _resolve_robot_type(cfg)
