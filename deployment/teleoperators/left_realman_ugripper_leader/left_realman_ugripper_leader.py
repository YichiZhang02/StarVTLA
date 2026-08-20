#!/usr/bin/env python

"""RealMan ugripper 单左主臂遥操作器。"""

import logging
import os
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np

from deployment.hardware.leader_arms import RealmanLeader
from deployment.teleoperators.teleoperator import Teleoperator
from vtla.engine.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_left_realman_ugripper_leader import LeftRealmanUGripperLeaderConfig

logger = logging.getLogger(__name__)


class _LeftLeaderReader(threading.Thread):
    """持续读取单左主臂，向采集循环提供最新位置。"""

    def __init__(self, leader: RealmanLeader):
        super().__init__(daemon=True, name="LeftLeaderReader")
        self._leader = leader
        self._running = True
        self._lock = threading.Lock()
        self._positions = leader.read_position()

    def run(self) -> None:
        while self._running:
            try:
                positions = self._leader.read_position()
                with self._lock:
                    self._positions = positions
            except Exception as exc:  # noqa: BLE001
                logger.debug("左主臂异步读取失败: %s", exc)
            time.sleep(0.001)

    def get_position(self) -> np.ndarray:
        with self._lock:
            return self._positions.copy()

    def stop(self) -> None:
        self._running = False


class LeftRealmanUGripperLeader(Teleoperator):
    """输出与 realman_ugripper_left 对齐的 8 维 left_* 动作。"""

    config_class = LeftRealmanUGripperLeaderConfig
    name = "left_realman_ugripper_leader"

    JOINT_NAMES = [f"main_joint{i}" for i in range(1, 8)]
    GRIPPER_NAME = "main_gripper"

    def __init__(self, config: LeftRealmanUGripperLeaderConfig):
        super().__init__(config)
        self.config = config
        self._leader: RealmanLeader | None = None
        self._reader: _LeftLeaderReader | None = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = {f"left_{joint}": float for joint in self.JOINT_NAMES}
        features[f"left_{self.GRIPPER_NAME}"] = float
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._leader is not None and self._leader.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} 已连接")
        if not os.path.exists(self.config.port):
            raise ConnectionError(
                f"左主臂串口 {self.config.port} 不存在，请检查 USB 连接和 udev 规则"
            )

        leader = RealmanLeader(
            port=self.config.port,
            baudrate=self.config.baudrate,
            hex_data=self.config.hex_data,
        )
        try:
            leader.connect()
            self._leader = leader
            if self.config.async_read:
                self._reader = _LeftLeaderReader(leader)
                self._reader.start()
                time.sleep(0.1)
            self.configure()
        except Exception:
            if self._reader is not None:
                self._reader.stop()
                self._reader.join(timeout=1.0)
            try:
                leader.disconnect()
            except Exception:
                pass
            self._leader = None
            self._reader = None
            raise

        logger.info("左主臂已连接: %s", self.config.port)

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        if self._reader is not None:
            self._reader.stop()
            self._reader.join(timeout=1.0)
            self._reader = None
        if self._leader is not None:
            try:
                self._leader.disconnect()
            finally:
                self._leader = None
        logger.info("左主臂已断开")

    def calibrate(self) -> None:
        logger.info("左主臂出厂已校准，跳过")

    def configure(self) -> None:
        pass

    def _normalize_gripper(self, raw: float) -> float:
        lo = self.config.leader_gripper_min
        hi = self.config.leader_gripper_max
        normalized = (float(raw) - lo) / (hi - lo)
        normalized = 0.5 + (normalized - 0.5) * self.config.gripper_gain
        return float(np.clip(normalized, 0.0, 1.0))

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        positions = (
            self._reader.get_position()
            if self._reader is not None
            else self._leader.read_position()
        )
        if len(positions) < 8:
            raise RuntimeError(f"左主臂返回 {len(positions)} 维位置，期望至少 8 维")

        action: dict[str, Any] = {}
        for index, joint in enumerate(self.JOINT_NAMES):
            value = float(positions[index])
            action[f"left_{joint}"] = float(np.rad2deg(value)) if self.config.use_degrees else value
        action[f"left_{self.GRIPPER_NAME}"] = self._normalize_gripper(float(positions[7]))
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass
