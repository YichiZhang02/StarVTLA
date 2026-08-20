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
        --robot.type=realman_ugripper_dual \
        --teleop.type=bi_realman_ugripper_leader \
        --dataset.repo_id=pick_pen \
        --dataset.single_task="抓笔" \
        --dataset.num_episodes=20

六维力拖动 (不连接主臂, 空格键切换夹爪):
    python -m deployment.collect \
        --mode=drag \
        --robot.type=realman_ugripper_left \
        --dataset.repo_id=drag_demo \
        --dataset.single_task="抓笔" \
        --drag_gripper_close_value=0.0

默认行为:
    - 数据存到 playground/data/<repo_id> (不传 --dataset.root 时)
    - 触觉随 robot 配置, 默认开; 不要触觉加 --robot.use_tactile=false
    - 不推 HuggingFace hub (需要才加 --dataset.push_to_hub=true)
    - 其余硬件参数 (IP/串口等) 走各自 config 默认, 需要时照样可 --robot.xxx / --teleop.xxx 覆盖
"""

import sys
from dataclasses import dataclass
from enum import Enum

from deployment._record_engine import RecordConfig, StickyHint, run_record  # noqa: E402
from vtla.engine.configs import parser


class CollectMode(str, Enum):
    TELEOP = "teleop"
    DRAG = "drag"


@dataclass
class CollectConfig(RecordConfig):
    """Collection-only control ownership and force-drag settings."""

    mode: CollectMode = CollectMode.TELEOP
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


@parser.wrap()
def collect(cfg: CollectConfig):
    if cfg.policy is not None:
        raise ValueError("collect 是采数据入口, 不接受 --policy.*; 模型推理请用 `python -m deployment.inference`")
    if cfg.reset_before_episode:
        raise ValueError("collect 不允许 episode 间自动复位; 请移除 --reset_before_episode=true")
    if cfg.mode == CollectMode.TELEOP and cfg.teleop is None:
        raise ValueError("collect 需要遥操作器: 请指定 --teleop.type=... (如 bi_realman_ugripper_leader)")
    if cfg.mode == CollectMode.DRAG:
        # Drag owns the follower directly. Ignore any inherited/default teleop config
        # and keep the recorded action in measured joint space.
        cfg.teleop = None
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
