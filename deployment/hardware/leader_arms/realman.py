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
睿尔曼 RM75b 主臂 (Leader Arm) 串口通信模块。

主臂通过 USB 串口连接，使用自定义协议读取关节位置。
协议格式：
- 请求: 55 AA 02 00 00 67
- 响应: AA 55 开头，47字节数据帧

采用 SO101 同步读取方式：每次调用 read_position() 时发送请求并等待响应。

本模块自包含 (仅依赖 pyserial / numpy / 标准库), 是主臂串口驱动的唯一来源,
由 robots/ 与 teleoperators/ 共同 import (替代历史上散落的多份拷贝)。
"""

import binascii
import logging
import time

import numpy as np
import serial

from .base import LeaderArmBase

logger = logging.getLogger(__name__)


class RealmanLeader(LeaderArmBase):
    """
    主臂串口通信类 (SO101 同步读取方式)

    每次调用 read_position() 时直接发送请求并等待响应，
    与 SO101 的 bus.sync_read() 方式一致。
    """

    # 协议常量
    FRAME_HEADER = b'\xaa\x55'
    FRAME_LENGTH = 47
    NUM_JOINTS = 7

    def __init__(
        self,
        port: str,
        baudrate: int = 460800,
        hex_data: str = "55 AA 02 00 00 67",
        timeout: float = 0.1,
    ):
        """
        初始化主臂通信

        Args:
            port: 串口路径，如 /dev/ttyRealmanISFLeaderR
            baudrate: 波特率，默认 460800
            hex_data: 请求指令 (hex 字符串)
            timeout: 读取超时 (秒)
        """
        self.port = port
        self.baudrate = baudrate
        self.hex_data = hex_data
        self.timeout = timeout

        self._serial_conn: serial.Serial | None = None
        self._bytes_to_send: bytes = b''

        # 上次读取的位置 (用于读取失败时返回)
        self._last_positions = np.zeros(8, dtype=np.float32)

        # 连接状态
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._is_connected and self._serial_conn is not None and self._serial_conn.isOpen()

    def connect(self) -> None:
        """建立串口连接"""
        if self.is_connected:
            logger.warning(f"主臂 {self.port} 已连接")
            return

        logger.info(f"正在连接主臂 {self.port} @ {self.baudrate}...")

        try:
            self._serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )

            # 清空缓冲区
            self._serial_conn.reset_input_buffer()
            self._serial_conn.reset_output_buffer()

            # 准备请求指令
            self._bytes_to_send = binascii.unhexlify(self.hex_data.replace(" ", ""))

            # 测试通信：发送一次请求并读取响应
            self._serial_conn.write(self._bytes_to_send)
            self._serial_conn.flush()
            time.sleep(0.05)

            # 检查是否有响应
            if self._serial_conn.inWaiting() > 0:
                data = self._serial_conn.read(self._serial_conn.inWaiting())
                if len(data) >= self.FRAME_LENGTH:
                    logger.info(f"主臂 {self.port} 通信测试成功，收到 {len(data)} 字节")
                else:
                    logger.warning(f"主臂 {self.port} 响应数据不完整: {len(data)} 字节")
            else:
                logger.warning(f"主臂 {self.port} 无响应，但继续尝试...")

            self._is_connected = True
            logger.info(f"主臂 {self.port} 连接成功")

        except Exception as e:
            self._is_connected = False
            raise ConnectionError(f"主臂连接失败: {e}")

    def disconnect(self) -> None:
        """断开连接"""
        if self._serial_conn is not None and self._serial_conn.isOpen():
            self._serial_conn.close()

        self._is_connected = False
        logger.info(f"主臂 {self.port} 已断开")

    def read_position(self) -> np.ndarray:
        """
        同步读取当前关节位置 (SO101 方式)

        每次调用都发送请求并等待响应，确保数据实时性。

        Returns:
            np.ndarray: 8个关节位置 (7关节 + 1夹爪)，单位为弧度
        """
        if not self.is_connected:
            logger.warning("主臂未连接，返回上次位置")
            return self._last_positions.copy()

        try:
            # 1. 清空输入缓冲区
            self._serial_conn.reset_input_buffer()

            # 2. 发送请求
            self._serial_conn.write(self._bytes_to_send)
            self._serial_conn.flush()

            # 3. 等待响应到达
            time.sleep(0.02)  # 等待 20ms

            # 4. 读取响应数据
            data = b""
            for _ in range(5):  # 最多尝试 5 次
                waiting = self._serial_conn.inWaiting()
                if waiting > 0:
                    data += self._serial_conn.read(waiting)
                if len(data) >= self.FRAME_LENGTH:
                    break
                time.sleep(0.005)  # 等待 5ms

            # 5. 解析数据
            if len(data) >= self.FRAME_LENGTH:
                header_idx = data.find(self.FRAME_HEADER)
                if header_idx != -1 and len(data) - header_idx >= self.FRAME_LENGTH:
                    frame = data[header_idx : header_idx + self.FRAME_LENGTH]
                    positions = self._parse_frame(frame)
                    if positions is not None:
                        self._last_positions = positions
                        return positions

            # 读取失败，返回上次位置
            logger.debug(f"主臂读取失败，返回上次位置")
            return self._last_positions.copy()

        except Exception as e:
            logger.error(f"主臂读取错误: {e}")
            return self._last_positions.copy()

    def _parse_frame(self, frame: bytes) -> np.ndarray | None:
        """
        解析 AA 55 开头的 47 字节协议帧

        数据结构:
        - [0-1]: 帧头 AA 55
        - [2-6]: 协议头 02 00 00 28 01
        - [7+]: 每5字节一组: [ID(1byte)] [Value(4bytes, int32_le)]

        数值归一化:
        - 关节: /10000 得到角度值
        - 夹爪: /1000 得到角度值

        Returns:
            np.ndarray: 8个关节位置 (弧度)，解析失败返回 None
        """
        try:
            positions = np.zeros(8, dtype=np.float32)

            # 从第7字节开始解析
            current_idx = 7

            # 解析 7 个关节
            for i in range(self.NUM_JOINTS):
                # 提取 4 字节数值 (小端序, 有符号整数)
                val_bytes = frame[current_idx : current_idx + 4]
                val_int = int.from_bytes(val_bytes, byteorder='little', signed=True)

                # 归一化为角度 (/10000)，然后转为弧度
                val_deg = val_int / 10000.0
                positions[i] = np.deg2rad(val_deg)

                # 跳过 ID 字节 (下一组数据的 ID)
                current_idx += 5

            # 解析夹爪 (第8个数据)
            val_bytes = frame[current_idx : current_idx + 4]
            val_int = int.from_bytes(val_bytes, byteorder='little', signed=True)
            # 夹爪归一化比例不同，直接作为弧度值使用
            positions[self.NUM_JOINTS] = val_int / 1000.0

            return positions

        except Exception as e:
            logger.debug(f"帧解析错误: {e}")
            return None

    def __del__(self):
        """析构时断开连接"""
        if self._is_connected:
            self.disconnect()
