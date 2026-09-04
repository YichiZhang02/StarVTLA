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

"""推理时腕部鱼眼去畸变, 消除训练-推理 gap。

训练数据集的腕部图是 `鱼眼去畸变 → 可选居中裁剪`(见 tools/undistort_dataset_videos.py);
推理时鱼眼相机给的是原生鱼眼帧。本模块对原生帧做**完全相同**的变换, 使 policy 看到的
几何与训练一致。

变换 (与 tools/undistort_dataset_videos.py 逐像素一致):
    1. cv2.fisheye 去畸变, 新内参 = K (原位去畸变, 不额外缩放)
    2. 可选居中裁剪 (新数据不裁剪，最终到 224 由推理视觉预处理完成)

标定为 Kalibr/OpenCV equidistant 鱼眼模型, 内置在 calib/x5_{left,right}_intrinsics.json
(从 tools/calib 复制, 使 deployment 自包含, 不依赖 tools/)。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_CALIB_DIR = Path(__file__).resolve().parent / "calib"

# 手臂 side -> 内置标定文件
_DEFAULT_CALIB = {
    "left": _CALIB_DIR / "x5_left_intrinsics.json",
    "right": _CALIB_DIR / "x5_right_intrinsics.json",
}


def default_calib_path(side: str) -> Path:
    """某条手臂腕部鱼眼的内置标定文件路径。"""
    if side not in _DEFAULT_CALIB:
        raise ValueError(f"无内置标定的 side={side!r} (仅 left/right)")
    return _DEFAULT_CALIB[side]


class WristUndistorter:
    """加载 equidistant 鱼眼标定，预计算 remap，并执行去畸变与可选裁剪。

    用法:
        und = WristUndistorter(calib_path, crop=None)
        rgb_rectified = und(rgb_fisheye)  # 保留完整去畸变画面
    """

    def __init__(self, calib_path: str | Path, crop: int | None = None):
        d = json.loads(Path(calib_path).read_text())
        model = d.get("distortion_model")
        if model != "equidistant":
            raise ValueError(
                f"{calib_path}: 期望 equidistant(鱼眼) 模型, 实际 {model!r}"
            )
        self.K = np.array(d["camera_matrix"], dtype=np.float64)
        self.D = np.array(d["distortion_coeffs"], dtype=np.float64).reshape((4, 1))
        w, h = (int(x) for x in d["resolution"])
        self.in_size = (w, h)  # (width, height)
        self.crop = int(crop) if crop is not None else None
        if self.crop is not None and self.crop > min(w, h):
            raise ValueError(
                f"crop={self.crop} 超过标定分辨率最小边 {min(w, h)} ({calib_path})"
            )
        # 新内参 = K (原位去畸变, 与训练工具一致)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.K, self.in_size, cv2.CV_16SC2
        )
        self._warned_resize = False
        output = (
            f"裁剪 {self.crop}x{self.crop}" if self.crop is not None else "保留完整画面"
        )
        logger.info(f"腕部去畸变就绪: 标定 {w}x{h} -> {output} ({calib_path})")

    @property
    def out_shape(self) -> tuple[int, int, int]:
        if self.crop is None:
            return (self.in_size[1], self.in_size[0], 3)
        return (self.crop, self.crop, 3)

    def __call__(self, frame: NDArray) -> NDArray:
        h, w = frame.shape[:2]
        if (w, h) != self.in_size:
            # 源分辨率与标定不一致: 缩到标定分辨率再去畸变 (映射表按标定分辨率算)。
            if not self._warned_resize:
                logger.warning(
                    f"腕部源分辨率 {w}x{h} 与标定 {self.in_size[0]}x{self.in_size[1]} 不一致, 已缩放对齐"
                )
                self._warned_resize = True
            frame = cv2.resize(frame, self.in_size, interpolation=cv2.INTER_AREA)
        und = cv2.remap(
            frame, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        if self.crop is None:
            return und
        in_w, in_h = self.in_size
        x0 = (in_w - self.crop) // 2
        y0 = (in_h - self.crop) // 2
        return und[y0:y0 + self.crop, x0:x0 + self.crop]
