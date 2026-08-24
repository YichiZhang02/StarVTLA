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
睿尔曼 RM75b 双臂主臂遥操作器配置

两条主臂通过 USB 串口读取关节位置, 输出加 left_ / right_ 前缀的动作,
与 rm_base_umi_dual 机器人的 action 字段对齐。

夹爪: 主臂原始读数 (约 [min, max], min=夹紧 / max=张开) 归一化到 [0,1] (1=张开),
与机器人 *_main_gripper 同坐标系。
"""

from dataclasses import dataclass, field

from deployment.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rm_leader_dual")
@dataclass
class RmLeaderDualConfig(TeleoperatorConfig):
    """双臂主臂遥操作器配置。"""

    # 启用的手臂
    arms: list[str] = field(default_factory=lambda: ["left", "right"])

    # ============ 主臂串口 ============
    left_port: str = "/dev/ttyRealmanBaseLeaderL"
    right_port: str = "/dev/ttyRealmanBaseLeaderR"
    baudrate: int = 460800
    hex_data: str = "55 AA 02 00 00 67"

    # ============ 数据格式 ============
    use_degrees: bool = False

    # ============ 异步读取 ============
    # 默认开启: 每条主臂后台线程持续读, get_action 走缓存 (~0ms)。
    # 关闭则每帧同步串读两条主臂 (每条 read_position 内部硬 sleep 20ms),
    # 双臂串行 ≥40ms, 会把录制主循环卡在 ~10fps 导致掉帧/快进。
    async_read: bool = True

    # ============ 主臂夹爪 -> 归一化 [0,1] 映射 ============
    # 主臂夹爪原始读数范围: min=夹紧, max=张开 (实测值, 见 ugripper_遥操测试.py)
    leader_gripper_min: float = 0.066
    leader_gripper_max: float = 0.971
    # 以中点 0.5 为中心的增益放大, 让夹爪行程更"满"
    gripper_gain: float = 1.0

    id: str | None = "rm_leader_dual"
