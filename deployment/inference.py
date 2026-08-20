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
模型推理入口 (policy 驱动) + 录像。

最小命令 (--match-policy 默认开, 硬件与任务自动对齐 checkpoint):
    python -m deployment.inference \
        --robot.type=realman_ugripper_dual \
        --policy.path=playground/results/models/xxx/checkpoints/005000/pretrained_model \
        --dataset.repo_id=eval_pen

--match-policy 会自动:
    - 触觉: 模型用触觉则 use_tactile=true, 否则 false (不连触觉)
    - 相机: 只保留模型实际消费的本地相机 (如模型 wrist_only 则丢掉 cam_top, 不连它)
    - single_task: 从 checkpoint 的 train_config.json -> 训练集 meta/tasks.parquet 自动取
关掉自动对齐: --match-policy=false (此时需手动 --robot.use_tactile / --dataset.single_task 等)

其余默认: 录像存 playground/eval/<repo_id> (video=True); num_episodes 用默认; 不推 hub。
"""

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from deployment._record_engine import RecordConfig, StickyHint, run_record  # noqa: E402
from vtla.engine.configs import parser

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig(RecordConfig):
    # 自动把机器人硬件 + 任务描述对齐到 checkpoint (默认开)
    match_policy: bool = True


def _task_from_checkpoint(pretrained_path: str) -> str | None:
    """从 checkpoint 的 train_config.json 找训练集, 读其 meta/tasks.parquet 的任务文字。

    仅当训练集恰好单任务时可靠; 多任务返回 None (需手动指定)。任何一步缺失也返回 None。
    """
    try:
        tc = Path(pretrained_path) / "train_config.json"
        if not tc.is_file():
            logger.warning(f"[match-policy] 未找到 {tc}, 无法自动获取 single_task")
            return None
        info = json.load(open(tc))
        ds = info.get("dataset", {}) or {}
        repo_id = ds.get("repo_id")
        # 候选根目录: train_config 里的 root (可能是训练机绝对路径) + 本地约定 playground/data/<repo_id>
        candidates = []
        if ds.get("root"):
            candidates.append(Path(ds["root"]))
        if repo_id:
            candidates.append(Path("playground/data") / repo_id)
        tasks_pq = next((c / "meta" / "tasks.parquet"
                         for c in candidates if (c / "meta" / "tasks.parquet").is_file()), None)
        if tasks_pq is None:
            logger.warning(f"[match-policy] 未找到训练集任务表 (试过: {[str(c) for c in candidates]})")
            return None
        import pandas as pd
        df = pd.read_parquet(tasks_pq)
        if len(df) != 1:
            logger.warning(f"[match-policy] 训练集有 {len(df)} 个任务, 无法自动选定, 请手动 --dataset.single_task")
            return None
        # 任务文字: 优先 'task' 列, 否则用索引 (LeRobot tasks.parquet 以任务文字为索引)
        task = str(df["task"].iloc[0]) if "task" in df.columns else str(df.index[0])
        return task
    except Exception as e:
        logger.warning(f"[match-policy] 解析训练集任务失败: {e}")
        return None


def _train_dataset_dirs(pretrained_path: str) -> tuple[list[Path], str]:
    """从 checkpoint 的 train_config.json 找训练集候选目录 + repo_id/root 文字 (用于名称兜底)。"""
    candidates: list[Path] = []
    name_hint = ""
    try:
        tc = Path(pretrained_path) / "train_config.json"
        if tc.is_file():
            ds = (json.load(open(tc)).get("dataset") or {})
            root, repo_id = ds.get("root"), ds.get("repo_id")
            name_hint = f"{root or ''} {repo_id or ''}"
            if root:
                candidates.append(Path(root))
            if repo_id:
                candidates.append(Path("playground/data") / repo_id)
    except Exception as e:
        logger.warning(f"[match-policy] 解析 train_config 失败: {e}")
    return candidates, name_hint


def _resolve_undistort(cfg: InferenceConfig) -> None:
    """把 robot.undistort_wrist=="auto" 按 checkpoint 解析为 true/false (分层: marker 优先, 名称兜底)。

    1) 训练集 meta/info.json 有 "undistort" 标记 -> 开启, 并采用其中的 crop;
    2) 训练集可访问但无标记 -> 关闭 (可靠判定: 该数据集未去畸变);
    3) 训练集不可访问 -> 看 train_config 的 repo_id/root 是否含 "undist";
    4) 显式 true/false 始终覆盖 auto。
    """
    if not hasattr(cfg.robot, "undistort_wrist"):
        return  # 该机器人不支持腕部去畸变, 跳过
    val = str(cfg.robot.undistort_wrist).lower()
    if val in ("true", "false"):
        logger.info(f"[match-policy] 腕部去畸变(显式): {val}")
        cfg.robot.undistort_wrist = val
        return

    candidates, name_hint = _train_dataset_dirs(cfg.policy.pretrained_path)
    info_path = next((c / "meta" / "info.json"
                      for c in candidates if (c / "meta" / "info.json").is_file()), None)
    enabled, crop, why = False, None, "默认关闭"
    if info_path is not None:
        try:
            marker = (json.load(open(info_path)).get("undistort") or None)
        except Exception:
            marker = None
        if marker:
            enabled, crop, why = True, marker.get("crop"), f"训练集标记 {info_path}"
        else:
            enabled, why = False, f"训练集无标记 {info_path}"
    elif "undist" in name_hint.lower():
        enabled, why = True, f"名称兜底('undist' in {name_hint.strip()!r})"

    cfg.robot.undistort_wrist = "true" if enabled else "false"
    if enabled and crop:
        cfg.robot.undistort_crop = int(crop)
    logger.info(f"[match-policy] 腕部去畸变(auto)={cfg.robot.undistort_wrist} "
                f"(crop={cfg.robot.undistort_crop}) <- {why}")


def _apply_match_policy(cfg: InferenceConfig) -> None:
    """按 checkpoint 把机器人硬件 + single_task 对齐到模型实际所需。"""
    in_feats = set(cfg.policy.input_features or {})
    uses = lambda sub: any(sub in k for k in in_feats)  # noqa: E731

    # 触觉: 模型 input_features 里有触觉键才开
    cfg.robot.use_tactile = uses("finger") or uses("tactile")

    # 本地相机 (cameras dict, 一般是 cam_top): 只留模型消费的
    if hasattr(cfg.robot, "cameras") and cfg.robot.cameras:
        kept = {n: c for n, c in cfg.robot.cameras.items()
                if f"observation.images.{n}" in in_feats}
        dropped = set(cfg.robot.cameras) - set(kept)
        cfg.robot.cameras = kept
        if dropped:
            logger.info(f"[match-policy] 模型不消费, 已不连相机: {sorted(dropped)}")

    # 任务描述: 仅当未手动指定时自动取。
    # 优先用 checkpoint 自带的 single_task (训练写入 config.json, 自包含, 不依赖数据集);
    # 取不到再回退到「从 train_config -> 训练集 tasks.parquet」查找 (兼容旧 checkpoint)。
    if cfg.dataset.single_task is None:
        task = getattr(cfg.policy, "single_task", None)
        if task:
            cfg.dataset.single_task = task
            logger.info(f"[match-policy] single_task <- checkpoint(config.json): {task!r}")
        else:
            task = _task_from_checkpoint(cfg.policy.pretrained_path)
            if task is not None:
                cfg.dataset.single_task = task
                logger.info(f"[match-policy] single_task <- 训练集 tasks.parquet: {task!r}")

    # 腕部去畸变: 按 checkpoint 自动判定 (消除训练-推理 gap)
    _resolve_undistort(cfg)

    # 动作空间: 按 checkpoint 的 action_mode 自动设 (relative_ee -> ee), 堵住"训练EE/推理joint"静默错配。
    _resolve_action_space(cfg)

    logger.info(f"[match-policy] 对齐结果: use_tactile={cfg.robot.use_tactile}, "
                f"cameras={list(getattr(cfg.robot, 'cameras', {}) or {})}, "
                f"undistort_wrist={getattr(cfg.robot, 'undistort_wrist', 'n/a')}, "
                f"action_space={getattr(cfg.robot, 'action_space', 'n/a')}")


def _resolve_action_space(cfg: InferenceConfig) -> None:
    """按 checkpoint 的 action_mode 对齐机器人动作空间, 并做硬校验防止带病起跑。

    - action_mode='relative_ee' -> robot.action_space='ee' (机器人发末端位姿, 走 rm_movep_canfd)
    - 否则                       -> 'joint'
    机器人不支持 action_space 字段时: 若 checkpoint 需要 ee, 直接报错 (该机器人无法执行 EE 动作,
    继续会把位姿当关节弧度下发 -> 撞机)。
    """
    needs_ee = getattr(cfg.policy, "action_mode", None) == "relative_ee"
    if not hasattr(cfg.robot, "action_space"):
        if needs_ee:
            raise ValueError(
                f"checkpoint 的 action_mode=relative_ee 需要 EE 动作空间, 但机器人 "
                f"'{getattr(cfg.robot, 'type', cfg.robot)}' 不支持 action_space 字段。"
                "请使用支持 EE 的机器人 (如 realman_ugripper_dual/realman_ugripper_left)。"
            )
        return
    cfg.robot.action_space = "ee" if needs_ee else "joint"
    logger.info(f"[match-policy] action_space <- action_mode={getattr(cfg.policy, 'action_mode', None)!r}: "
                f"{cfg.robot.action_space}")


@parser.wrap()
def inference(cfg: InferenceConfig):
    # 注意: policy 由 RecordConfig.__post_init__ 从 --policy.path 加载, 解析后即就绪
    if cfg.policy is None:
        raise ValueError("inference 需要模型: 请指定 --policy.path=.../pretrained_model")
    if cfg.teleop is not None:
        raise ValueError("inference 是纯推理入口, 不要 --teleop.*; 采数据请用 `python -m deployment.collect`")

    if cfg.match_policy:
        _apply_match_policy(cfg)

    if cfg.dataset.single_task is None:
        raise ValueError(
            "缺少 single_task: --match-policy 未能自动获取 (训练集缺失/多任务?), "
            "请手动 --dataset.single_task=\"...\""
        )

    # 录像默认存到 playground/eval/<repo_id 末段> (时间戳命名由调用方/bash 负责)
    if cfg.dataset.root is None:
        cfg.dataset.root = f"playground/eval/{cfg.dataset.repo_id.split('/')[-1]}"

    # 钉在底行的保存提示 (推理时容易被日志刷掉)
    hint = " \033[30;43m 推理中 ↑开始 | →保存 | ←重录 | ESC退出 \033[0m"
    with StickyHint(hint):
        return run_record(cfg)


def main():
    import faulthandler
    faulthandler.enable(file=sys.stderr, all_threads=True)
    inference()


if __name__ == "__main__":
    main()
