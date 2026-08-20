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
睿尔曼从臂 (被控机械臂本体), 通过厂商 Robotic_Arm SDK over TCP/IP 控制。

- 关节状态: rm_get_joint_degree() 约 50ms 延迟, 内部用后台线程持续读取并缓存,
  read_joints() 非阻塞取缓存。
- 下发关节: rm_movej_canfd()。
- 单位: use_degrees=False 时对外用弧度 (SDK 原生是角度, 类内部负责换算)。

构造参数只有 ip/port 等原始值, 不依赖 RobotConfig。
"""

import logging
import threading
import time

import numpy as np

from .._sdk_paths import ensure_realman_sdk
from .base import FollowerArmBase

logger = logging.getLogger(__name__)


class _FollowerStateReader(threading.Thread):
    """后台持续读取从臂关节状态并缓存, 解决 rm_get_joint_degree() ~50ms 延迟。"""

    def __init__(self, arm, use_degrees: bool = False, dof: int = 7):
        super().__init__(daemon=True, name="FollowerStateReader")
        self._arm = arm
        self._use_degrees = use_degrees
        self.running = True
        self._lock = threading.Lock()
        self._joints = [0.0] * dof
        self._last_update = 0.0

    def run(self):
        while self.running:
            try:
                ret, joints_deg = self._arm.rm_get_joint_degree()
                if ret == 0:
                    joints = list(joints_deg) if self._use_degrees else [np.radians(j) for j in joints_deg]
                    with self._lock:
                        self._joints = joints
                        self._last_update = time.time()
            except Exception as e:
                logger.debug(f"异步读取从臂状态错误: {e}")
            time.sleep(0.005)

    def get_state(self) -> list[float]:
        with self._lock:
            return self._joints.copy()

    def get_state_age(self) -> float:
        with self._lock:
            return time.time() - self._last_update if self._last_update > 0 else float("inf")

    def stop(self):
        self.running = False


class RealmanTcpFollower(FollowerArmBase):
    """睿尔曼从臂 (Robotic_Arm SDK over TCP)。"""

    # 类级锁: 多条臂并行 connect 时, 保护 Robotic_Arm SDK 句柄创建 (厂商 SDK 创建非线程安全)。
    _sdk_lock = threading.Lock()

    def __init__(
        self,
        ip: str,
        port: int = 8080,
        dof: int = 7,
        use_degrees: bool = False,
        name: str = "follower",
    ):
        self.name = name
        self._ip = ip
        self._port = port
        self._dof = dof
        self._use_degrees = use_degrees

        self._arm = None       # RoboticArm 实例
        self._handle = None    # rm_create_robot_arm 句柄
        self._connected = False
        self._reader: _FollowerStateReader | None = None
        self._force_drag_active = False

    @staticmethod
    def _import_sdk():
        """延迟导入睿尔曼 SDK。"""
        ensure_realman_sdk()
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        except ImportError as e:
            raise ImportError(
                f"无法导入睿尔曼 SDK (Robotic_Arm)。请确认其在 deployment/sdk/ 下。原始错误: {e}"
            ) from e
        return RoboticArm, rm_thread_mode_e

    @property
    def is_connected(self) -> bool:
        return self._connected and self._arm is not None

    def connect(self) -> None:
        RoboticArm, rm_thread_mode_e = self._import_sdk()
        with self._sdk_lock:
            self._arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            self._handle = self._arm.rm_create_robot_arm(self._ip, self._port)
        if self._handle.id < 0:
            raise ConnectionError(
                f"[{self.name}] 从臂连接失败! IP: {self._ip}, 错误代码: {self._handle.id}"
            )
        self._connected = True
        logger.info(f"[{self.name}] 从臂连接成功 Handle ID: {self._handle.id}")

        self._reader = _FollowerStateReader(self._arm, use_degrees=self._use_degrees, dof=self._dof)
        self._reader.start()
        time.sleep(0.1)

    def read_joints(self) -> np.ndarray:
        """缓存的当前关节位置 (配置单位)。后台线程持续刷新, 非阻塞。"""
        if self._reader is None:
            return np.zeros(self._dof, dtype=float)
        return np.asarray(self._reader.get_state(), dtype=float)

    def read_joints_now(self) -> np.ndarray | None:
        """同步即时读取当前关节位置 (配置单位)。用于下发前的安全限幅。失败返回 None。"""
        if self._arm is None:
            return None
        ret, joints_deg = self._arm.rm_get_joint_degree()
        if ret != 0:
            return None
        if self._use_degrees:
            return np.asarray(joints_deg, dtype=float)
        return np.asarray([np.radians(j) for j in joints_deg], dtype=float)

    def get_state_age(self) -> float:
        """缓存状态的年龄 (秒), 用于过时告警。"""
        return self._reader.get_state_age() if self._reader is not None else float("inf")

    def send_joints(self, positions) -> int:
        """下发关节目标 (配置单位)。内部换算成角度调用 rm_movej_canfd。返回 SDK 错误码 (0=成功)。"""
        target_degrees = []
        for val in positions:
            target_degrees.append(float(np.degrees(val) if not self._use_degrees else val))
        ret = self._arm.rm_movej_canfd(target_degrees, False, 0)
        if ret != 0:
            logger.warning(f"[{self.name}] 发送关节动作失败, 错误码: {ret}")
        return ret

    def send_pose(
        self, pose7, follow: bool = False, trajectory_mode: int = 0, radio: int = 0
    ) -> int:
        """下发笛卡尔目标位姿 (基座系绝对), 控制器侧做 IK。返回 SDK 错误码 (0=成功)。

        pose7: [x, y, z, qw, qx, qy, qz] —— 位置 (米) + 四元数 (RM API 顺序 wxyz)。
        位姿系须与 rm_algo_forward_kinematics 一致 (默认无工具/工作系 = flange/base);
        若控制器设了非单位工具系/工作系, movep 的位姿会被偏移, 调用方需自行对齐。
        """
        if self._arm is None:
            return -1
        ret = self._arm.rm_movep_canfd(list(pose7), follow, trajectory_mode, radio)
        if ret != 0:
            logger.warning(f"[{self.name}] 发送位姿动作失败, 错误码: {ret}")
        return ret

    def start_six_axis_force_drag(
        self,
        *,
        precise: bool = True,
        mode: int = 3,
        singular_wall: bool = True,
    ) -> None:
        """Enable RealMan six-axis force drag on this follower arm."""
        if self._arm is None:
            raise RuntimeError(f"[{self.name}] cannot start force drag while disconnected")
        if mode not in (1, 2, 3):
            raise ValueError(f"force drag mode must be 1, 2 or 3, got {mode}")
        if self._force_drag_active:
            return

        requested_precision = 1 if precise else 0
        ret = int(self._arm.rm_set_force_drag_mode(requested_precision))
        if ret != 0:
            raise RuntimeError(
                f"[{self.name}] rm_set_force_drag_mode({requested_precision}) failed: ret={ret}"
            )
        ret, actual_precision = self._arm.rm_get_force_drag_mode()
        if int(ret) != 0 or int(actual_precision) != requested_precision:
            raise RuntimeError(
                f"[{self.name}] force drag precision verification failed: "
                f"ret={ret}, requested={requested_precision}, actual={actual_precision}"
            )

        ret = int(
            self._arm.rm_start_multi_drag_teach(
                int(mode),
                1 if singular_wall else 0,
            )
        )
        if ret != 0:
            try:
                self._arm.rm_stop_drag_teach()
            except Exception:
                pass
            raise RuntimeError(
                f"[{self.name}] rm_start_multi_drag_teach({mode}) failed: ret={ret}"
            )
        self._force_drag_active = True
        logger.warning(
            "[%s] six-axis force drag enabled: precise=%s mode=%d singular_wall=%s",
            self.name,
            precise,
            mode,
            singular_wall,
        )

    def stop_drag_teach(self) -> None:
        """Stop force drag and latch the measured joints as the new hold target."""
        if not self._force_drag_active:
            return
        if self._arm is None:
            raise RuntimeError(f"[{self.name}] force drag is active but arm is disconnected")

        hold_joints = self.read_joints_now()
        if hold_joints is None:
            cached = self.read_joints()
            if cached.size == self._dof and np.all(np.isfinite(cached)) and self.get_state_age() <= 0.2:
                hold_joints = cached
                logger.warning(
                    "[%s] using %.1fms cached joints for drag-exit hold",
                    self.name,
                    self.get_state_age() * 1000.0,
                )

        ret = int(self._arm.rm_stop_drag_teach())
        if ret != 0:
            raise RuntimeError(f"[{self.name}] rm_stop_drag_teach failed: ret={ret}")
        self._force_drag_active = False

        if hold_joints is None:
            raise RuntimeError(
                f"[{self.name}] force drag stopped but the current hold target could not be captured"
            )
        hold_ret = self.send_joints(hold_joints.tolist())
        if hold_ret != 0:
            raise RuntimeError(
                f"[{self.name}] force drag stopped but current-pose hold failed: ret={hold_ret}"
            )
        logger.warning(
            "[%s] six-axis force drag disabled; current joint pose latched as hold target",
            self.name,
        )

    def get_tool_work_frames(self) -> tuple[dict | None, dict | None]:
        """读当前工具坐标系 / 工作坐标系 (用于上电自检 movep 位姿系是否= flange/base)。

        失败返回 (None, None) 对应项。不抛异常 (自检用)。
        """
        tool = work = None
        if self._arm is None:
            return tool, work
        try:
            ret_t, tool = self._arm.rm_get_current_tool_frame()
            if ret_t != 0:
                tool = None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{self.name}] 读取工具坐标系失败: {e}")
        try:
            ret_w, work = self._arm.rm_get_current_work_frame()
            if ret_w != 0:
                work = None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{self.name}] 读取工作坐标系失败: {e}")
        return tool, work

    def move_to(self, target_positions, duration_s: float = 4.0, fps: float = 30.0) -> None:
        """平滑运动到目标关节位置 (cosine ease-in/out 插值, 通过 rm_movej_canfd 分步执行)。

        target_positions: 与 send_joints / read_joints 单位相同 (取决于 use_degrees 配置)。
        """
        if self._arm is None:
            return

        current = self.read_joints_now()
        if current is None:
            current = self.read_joints()

        target = np.asarray(target_positions, dtype=float)
        start = np.asarray(current, dtype=float)

        n_steps = max(1, int(duration_s * fps))
        dt = duration_s / n_steps

        for i in range(1, n_steps + 1):
            t = i / n_steps
            t_smooth = 0.5 * (1.0 - np.cos(np.pi * t))
            interp = start + (target - start) * t_smooth
            self.send_joints(interp.tolist())
            time.sleep(dt)

        logger.info(f"[{self.name}] move_to 完成")

    def disconnect(self) -> None:
        if self._force_drag_active:
            try:
                self.stop_drag_teach()
            except Exception as e:
                logger.warning(f"[{self.name}] 停止力拖动出错: {e}")
        if self._reader is not None:
            try:
                self._reader.stop()
                self._reader.join(timeout=1.0)
            except Exception as e:
                logger.warning(f"[{self.name}] 停止状态线程出错: {e}")
            self._reader = None
        if self._arm is not None:
            try:
                self._arm.rm_delete_robot_arm()
            except Exception as e:
                logger.warning(f"[{self.name}] 断开从臂出错: {e}")
            self._arm = None
        self._handle = None
        self._connected = False
