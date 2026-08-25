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

"""
采数据入口 (遥操作 teleop 驱动)。

最小命令:
    python -m deployment.collect \
        --robot.type=rm_base_umi_dual \
        --dataset.repo_id=pick_pen \
        --dataset.single_task="抓笔" \
        --dataset.num_episodes=20

六维力拖动 (不连接主臂, 空格键切换夹爪):
    python -m deployment.collect \
        --mode=drag \
        --robot.type=rm_isf_umi_left \
        --dataset.repo_id=drag_demo \
        --dataset.single_task="抓笔" \
        --drag_gripper_close_value=0.0

默认行为:
    - 数据存到 playground/data/<repo_id> (不传 --dataset.root 时)
    - 触觉随 robot 配置, 默认开; 不要触觉加 --robot.use_tactile=false
    - 不推 HuggingFace hub (需要才加 --dataset.push_to_hub=true)
    - 其余硬件参数 (IP/串口等) 走各自 config 默认, 需要时照样可 --robot.xxx / --teleop.xxx 覆盖
"""

import math
import sys
from dataclasses import dataclass
from enum import Enum

from deployment._record_engine import RecordConfig, StickyHint, run_record  # noqa: E402
from deployment.teleoperators import TeleoperatorConfig
from vtla.engine.configs import parser


class CollectMode(str, Enum):
    TELEOP = "teleop"
    DRAG = "drag"


@dataclass
class CollectConfig(RecordConfig):
    """Collection control mode and an independent episode-reset policy."""

    mode: CollectMode = CollectMode.TELEOP
    # Independent of mode: both teleop and drag support reset=True/False.
    reset_before_episode: bool = False
    # LingKong normalized gripper convention: 1=open, 0=fully closed.
    drag_gripper_open_value: float = 1.0
    drag_gripper_close_value: float = 0.0
    # RealMan rm_start_multi_drag_teach parameters. Mode 3 enables XYZ and rotation.
    drag_force_precise: bool = True
    drag_force_mode: int = 3
    drag_singular_wall: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in (CollectMode.TELEOP, CollectMode.DRAG):
            raise ValueError(f"collect mode 只支持 teleop/drag, got: {self.mode!r}")
        for name in ("drag_gripper_open_value", "drag_gripper_close_value"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.drag_gripper_close_value >= self.drag_gripper_open_value:
            raise ValueError("drag_gripper_close_value must be smaller than drag_gripper_open_value")
        if self.drag_force_mode not in (1, 2, 3):
            raise ValueError("drag_force_mode must be 1, 2 or 3")


def _resolve_teleop_config(cfg: CollectConfig) -> None:
    """Resolve and validate the teleoperator declared by the concrete robot config."""
    if cfg.mode == CollectMode.DRAG:
        cfg.teleop = None
        return

    teleop_type = cfg.robot.teleop_type
    if not teleop_type:
        raise ValueError(
            f"robot.type={cfg.robot.type!r} does not declare a teleop_type"
        )
    try:
        teleop_cls = TeleoperatorConfig.get_choice_class(teleop_type)
    except KeyError as exc:
        raise ValueError(
            f"robot.type={cfg.robot.type!r} declares unregistered teleop_type={teleop_type!r}"
        ) from exc

    if cfg.teleop is None:
        cfg.teleop = teleop_cls()
    elif not isinstance(cfg.teleop, teleop_cls):
        raise ValueError(
            f"robot.type={cfg.robot.type!r} requires teleop.type={teleop_type!r}, "
            f"got {cfg.teleop.type!r}"
        )


def _validate_reset_home(cfg: CollectConfig) -> None:
    """Reject partial RealMan home targets before connecting hardware."""
    if not cfg.reset_before_episode:
        return

    home_duration_s = float(getattr(cfg.robot, "home_duration_s", 4.0))
    tolerance_deg = float(getattr(cfg.robot, "home_joint_tolerance_deg", 1.0))
    settle_timeout_s = float(getattr(cfg.robot, "home_settle_timeout_s", 2.0))
    if not math.isfinite(home_duration_s) or home_duration_s <= 0:
        raise ValueError("--robot.home_duration_s 必须大于 0")
    if not math.isfinite(tolerance_deg) or tolerance_deg <= 0:
        raise ValueError("--robot.home_joint_tolerance_deg 必须大于 0")
    if not math.isfinite(settle_timeout_s) or settle_timeout_s < 0:
        raise ValueError("--robot.home_settle_timeout_s 不能小于 0")

    home_joints = getattr(cfg.robot, "home_joints", None)
    if home_joints is None:
        # The robot captures its connection-time pose as a fixed home target.
        return
    if not isinstance(home_joints, dict):
        raise ValueError("--robot.home_joints 必须是关节名到角度值的字典")

    sides = getattr(cfg.robot, "arms", None)
    if sides is None:
        sides = cfg.robot.kinematics_sides
    expected = {
        f"{side}_main_joint{joint_index}"
        for side in sides
        for joint_index in range(1, 8)
    }
    provided = set(home_joints)
    missing = sorted(expected - provided)
    unknown = sorted(provided - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"缺少 {missing}")
        if unknown:
            details.append(f"未知 {unknown}")
        raise ValueError(
            "--robot.home_joints 必须完整指定启用臂的 7 个关节: "
            + "; ".join(details)
        )

    non_finite = sorted(
        key for key, value in home_joints.items() if not math.isfinite(float(value))
    )
    if non_finite:
        raise ValueError(f"--robot.home_joints 包含非有限值: {non_finite}")


@parser.wrap()
def collect(cfg: CollectConfig):
    if cfg.policy is not None:
        raise ValueError("collect 是采数据入口, 不接受 --policy.*; 模型推理请用 `python -m deployment.inference`")
    _resolve_teleop_config(cfg)
    _validate_reset_home(cfg)
    if cfg.mode == CollectMode.DRAG:
        # Drag owns the follower directly and records measured joint-space actions.
        if hasattr(cfg.robot, "action_space"):
            cfg.robot.action_space = "joint"
    if cfg.dataset.single_task is None:
        raise ValueError("collect 需要任务描述: 请指定 --dataset.single_task=\"...\"")

    # 默认存到 playground/data/<repo_id 末段> (时间戳命名由调用方/bash 负责)
    if cfg.dataset.root is None:
        cfg.dataset.root = f"playground/data/{cfg.dataset.repo_id.split('/')[-1]}"

    if cfg.mode == CollectMode.DRAG:
        hint = " \033[30;42m 拖动采集中 ↑开始 | 空格开/关夹爪 | →保存 | ←重录 | ESC退出 \033[0m"
    else:
        hint = " \033[30;42m 遥操采集中 ↑开始 | →保存 | ←重录 | ESC退出 \033[0m"
    with StickyHint(hint):
        return run_record(cfg)


def main():
    import faulthandler
    faulthandler.enable(file=sys.stderr, all_threads=True)
    collect()


if __name__ == "__main__":
    main()
