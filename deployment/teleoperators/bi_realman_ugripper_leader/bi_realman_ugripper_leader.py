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
睿尔曼 RM75b 双臂主臂遥操作器

两条主臂 (USB 串口) 读取关节位置, 输出 left_ / right_ 前缀的动作, 与
rm_base_umi_dual 机器人对齐。夹爪原始读数归一化到 [0,1] (1=张开)。
"""

import logging
import os
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np

from deployment.teleoperators.teleoperator import Teleoperator
from vtla.engine.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_bi_realman_ugripper_leader import BiRealmanUGripperLeaderConfig

logger = logging.getLogger(__name__)


class _AsyncLeaderReader(threading.Thread):
    """后台持续读取单条主臂位置, get_action 从缓存取用 (避免 ~22ms 串口阻塞)。"""

    def __init__(self, leader_arm):
        super().__init__(daemon=True, name="LeaderReader")
        self.leader_arm = leader_arm
        self.running = True
        self._lock = threading.Lock()
        self._pos = leader_arm.read_position()

    def run(self):
        while self.running:
            try:
                pos = self.leader_arm.read_position()
                with self._lock:
                    self._pos = pos
            except Exception as e:
                logger.debug(f"主臂异步读取错误: {e}")
            time.sleep(0.001)

    def get_position(self):
        with self._lock:
            return self._pos.copy()

    def stop(self):
        self.running = False


class BiRealmanUGripperLeader(Teleoperator):
    """睿尔曼 RM75b 双臂主臂遥操作器。"""

    config_class = BiRealmanUGripperLeaderConfig
    name = "bi_realman_ugripper_leader"

    DOF = 7
    JOINT_NAMES = [f"main_joint{i}" for i in range(1, 8)]
    GRIPPER_NAME = "main_gripper"

    def __init__(self, config: BiRealmanUGripperLeaderConfig):
        super().__init__(config)
        self.config = config

        for side in config.arms:
            if side not in ("left", "right"):
                raise ValueError(f"无效的手臂名 '{side}', 只支持 'left' / 'right'")

        # side -> {"arm": LeaderArm, "reader": _AsyncLeaderReader|None}
        self._leaders: dict[str, dict] = {side: {"arm": None, "reader": None} for side in config.arms}
        self._connected = False

    # ==================== 特征定义 ====================

    @property
    def _motors_ft(self) -> dict[str, type]:
        ft: dict[str, type] = {}
        for side in self.config.arms:
            for joint in self.JOINT_NAMES:
                ft[f"{side}_{joint}"] = float
            ft[f"{side}_{self.GRIPPER_NAME}"] = float
        return ft

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    # ==================== 连接状态 ====================

    @property
    def is_connected(self) -> bool:
        return self._connected and all(
            d["arm"] is not None and d["arm"].is_connected for d in self._leaders.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def _port(self, side: str) -> str:
        return self.config.left_port if side == "left" else self.config.right_port

    # ==================== 连接/断开 ====================

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} 已连接")

        from deployment.hardware.leader_arms import RealmanLeader as LeaderArm

        for side in self.config.arms:
            port = self._port(side)
            if not os.path.exists(port):
                raise ConnectionError(f"[{side}] 主臂串口 {port} 不存在, 请检查连接与 udev 规则")

            logger.info(f"[{side}] 正在连接主臂 {port}...")
            arm = LeaderArm(port=port, baudrate=self.config.baudrate, hex_data=self.config.hex_data)
            arm.connect()
            self._leaders[side]["arm"] = arm

            if self.config.async_read:
                reader = _AsyncLeaderReader(arm)
                reader.start()
                time.sleep(0.1)
                self._leaders[side]["reader"] = reader

            logger.info(f"[{side}] 主臂连接成功")

        self._connected = True
        self.configure()

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        for side, d in self._leaders.items():
            if d["reader"] is not None:
                d["reader"].stop()
                d["reader"].join(timeout=1.0)
                d["reader"] = None
            if d["arm"] is not None:
                d["arm"].disconnect()
                d["arm"] = None

        self._connected = False
        logger.info(f"{self} 已断开")

    def calibrate(self) -> None:
        logger.info("主臂出厂已校准, 跳过")

    def configure(self) -> None:
        pass

    # ==================== 夹爪归一化 ====================

    def _normalize_gripper(self, raw: float) -> float:
        """主臂夹爪原始读数 -> [0,1] (1=张开)。"""
        lo, hi = self.config.leader_gripper_min, self.config.leader_gripper_max
        norm = (raw - lo) / max(hi - lo, 1e-6)
        norm = 0.5 + (norm - 0.5) * self.config.gripper_gain  # 以中点放大
        return float(max(0.0, min(1.0, norm)))

    # ==================== 读取动作 ====================

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        action: dict[str, Any] = {}
        for side in self.config.arms:
            d = self._leaders[side]
            positions = d["reader"].get_position() if d["reader"] is not None else d["arm"].read_position()

            for i, joint in enumerate(self.JOINT_NAMES):
                val = positions[i]
                action[f"{side}_{joint}"] = np.rad2deg(val) if self.config.use_degrees else val

            action[f"{side}_{self.GRIPPER_NAME}"] = self._normalize_gripper(positions[7])

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass
