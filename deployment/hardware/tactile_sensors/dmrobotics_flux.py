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
dmrobotics Flux 触觉传感器 (gRPC + UDP)。

板子上 dmrobotics Flux gRPC 服务 getDepth + getDeformation2D -> 定点编码为 (H, W, 3)
uint16。通道索引: 0=depth, 1=deform_x, 2=deform_y。

收流跑在独立进程 (spawn): dmrobotics 多传感器/多 gRPC 流在单进程多线程里会互相饿死 GIL,
故每路一个进程, 通过 mp.Queue(maxsize=1) 只回传最新帧。本模块自包含 (不与鱼眼共享收流代码)。

编码遵循 tactile_u16_fixed_v1:
    depth  -> round(depth * 1000)
    deform -> round(deform * 1000 + 30000)
"""

import logging
import multiprocessing as mp
import time
from queue import Empty

import numpy as np
from numpy.typing import NDArray

from .._sdk_paths import ensure_dmrobotics_sdk
from .base import TactileSensorBase
from .encoding import encode_tactile_u16

logger = logging.getLogger(__name__)


def _put_latest(q, item) -> None:
    """maxsize=1 队列: 丢旧留新, 始终只保留最新一帧。"""
    try:
        q.get_nowait()
    except Empty:
        pass
    try:
        q.put_nowait(item)
    except Exception:
        pass


def _tactile_worker(
    dev_id, remote_addr, pc_host, pc_port,
    max_fps, debug,
    q, stop,
):
    """触觉工作进程: dmrobotics Flux -> (H,W,3) uint16。

    通道索引: 0=depth, 1=deform_x, 2=deform_y。
    """
    ensure_dmrobotics_sdk()

    from dmrobotics import Mode, Sensor, SensorOptions

    while not stop.is_set():
        sensor = None
        try:
            opt = SensorOptions(
                dev_id=dev_id, backend="Flux", mode=Mode.STANDARD, show_fps=False,
                enable_raw=False, enable_deformation=True, enable_depth=True,
                enable_shear=False, enable_force=False,
                remote_addr=remote_addr, pc_host=pc_host, pc_port=pc_port,
                max_fps=int(max_fps) if max_fps and max_fps > 0 else 120,
            )
            sensor = Sensor(opt)
            last = -1
            min_interval = 1.0 / max_fps if max_fps and max_fps > 0 else 0.0
            next_emit = 0.0
            n_fr = 0
            proc_ms = 0.0
            t_stat = time.time()
            while not stop.is_set():
                if sensor.getDevStatus() != 0:
                    time.sleep(0.01)
                    continue
                if not sensor.wait_for_new(last, timeout_ms=500):
                    continue
                # 限速到 max_fps: 传感器原生 ~110fps, 无需全要; sleep 让出 CPU 给主循环。
                now = time.perf_counter()
                if min_interval and now < next_emit:
                    time.sleep(next_emit - now)
                next_emit = time.perf_counter() + min_interval
                _s = time.perf_counter()
                fid, depth = sensor.getDepth()
                _, deform = sensor.getDeformation2D()
                last = fid if fid is not None else last
                if depth is None or deform is None:
                    continue

                packed = encode_tactile_u16(depth, deform)

                proc_ms += (time.perf_counter() - _s) * 1000
                n_fr += 1
                _put_latest(q, packed)

                if debug and time.time() - t_stat >= 2.0:
                    dt = time.time() - t_stat
                    print(f"[diag tactile {remote_addr}] 产出fps={n_fr/dt:.1f} "
                          f"取数+处理={proc_ms/max(n_fr,1):.1f}ms", flush=True)
                    n_fr = 0
                    proc_ms = 0.0
                    t_stat = time.time()
        except Exception as e:
            if not stop.is_set():
                print(f"[tactile {remote_addr}] 出错, 2s 后重连: {e}", flush=True)
                time.sleep(2)
        finally:
            try:
                if sensor is not None:
                    sensor.disconnect()
            except Exception:
                pass


class DmroboticsFlux(TactileSensorBase):
    """dmrobotics Flux 触觉传感器 (gRPC + UDP), 自带独立收流进程。"""

    def __init__(
        self,
        dev_id: int,
        remote_addr: str,
        pc_host: str,
        pc_port: int,
        width: int,
        height: int,
        max_fps: float = 0.0,
        debug: bool = False,
        first_frame_timeout: float = 15.0,
        name: str = "cam_finger",
    ):
        self.name = name
        self._dev_id = dev_id
        self._remote_addr = remote_addr
        self._pc_host = pc_host
        self._pc_port = pc_port
        self._max_fps = max_fps
        self._debug = debug
        self._first_frame_timeout = first_frame_timeout

        self._empty_frame = np.zeros((height, width, 3), dtype=np.uint16)

        self._ctx = mp.get_context("spawn")
        self._q = None
        self._stop = None
        self._proc = None
        self._last = None

    @property
    def is_connected(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def connect(self) -> None:
        self._q = self._ctx.Queue(maxsize=1)
        self._stop = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_tactile_worker,
            args=(self._dev_id, self._remote_addr, self._pc_host, self._pc_port,
                  self._max_fps, self._debug,
                  self._q, self._stop),
            daemon=True,
            name=f"stream-{self.name}",
        )
        self._proc.start()

        start = time.time()
        while time.time() - start < self._first_frame_timeout:
            try:
                self._last = self._q.get(timeout=0.2)
                logger.info(
                    f"[{self.name}] 首帧已接收 shape={getattr(self._last, 'shape', None)} "
                    f"dtype={getattr(self._last, 'dtype', None)}"
                )
                return
            except Empty:
                if not self._proc.is_alive():
                    raise ConnectionError(f"[{self.name}] 工作进程在收到首帧前退出")
        logger.warning(f"[{self.name}] 等待首帧超时 ({self._first_frame_timeout}s), 进程已启动, 继续")

    def async_read(self) -> NDArray:
        """取最新帧 (非阻塞); 无新帧返回上次缓存, 从未收到则返回空帧。"""
        try:
            while True:
                self._last = self._q.get_nowait()
        except Empty:
            pass
        return self._last if self._last is not None else self._empty_frame

    def disconnect(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None
        logger.info(f"[{self.name}] 已断开")
