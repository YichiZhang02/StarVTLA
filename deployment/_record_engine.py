# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
录制引擎 (tac_infra deployment) —— 共享逻辑, 不是命令行入口。

采集 (deployment/collect.py, teleop 驱动) 与推理 (deployment/inference.py, policy 驱动)
两个入口都构建 RecordConfig 后调用本模块的 run_record(cfg)。这里集中: 录制主循环、
数据集创建 / episode 管理 / 键盘事件 / 处理管线 / 视频编码, 避免两个入口重复代码。

动作可来自遥操作 (主臂) 或策略 (模型推理)。触觉以 uint16 采集，并作为正式
video feature 保存到 videos/ 下的无损 FFV1/gbrp16le Matroska 文件。
请勿直接运行本文件; 用 collect.py / inference.py。
"""

# ============================================================================
# X11 线程安全初始化 - 必须在任何其他导入之前执行
# 解决 pynput (Xlib) + 多线程导致的 futex 崩溃问题
# ============================================================================
import sys
import os


def _init_x11_threads():
    """初始化 X11 多线程支持，防止 Xlib 在多线程环境下崩溃。"""
    if sys.platform.startswith('linux') and os.environ.get('DISPLAY'):
        try:
            import ctypes
            x11 = ctypes.CDLL('libX11.so.6')
            result = x11.XInitThreads()
            if result == 0:
                import logging
                logging.warning("XInitThreads() 返回 0，X11 多线程初始化可能失败")
        except OSError:
            pass
        except Exception as e:
            import logging
            logging.debug(f"X11 线程初始化跳过: {e}")


_init_x11_threads()
# ============================================================================

import contextlib
import logging
import math
import platform
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, Callable

import numpy as np

# ---- 硬件层 (deployment 自包含) ----
from deployment.hardware.tactile_sensors import TactileMkvWriter
from deployment.robots import Robot, RobotConfig, make_robot_from_config
from deployment.robots.rm_base_umi_dual import RmBaseUmiDual  # noqa: F401  注册 config 选项
from deployment.robots.rm_isf_umi_left import RmIsfUmiLeft  # noqa: F401  注册 config 选项
from deployment.robots.rm_isf_umi_right import RmIsfUmiRight  # noqa: F401  注册 config 选项
from deployment.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from deployment.teleoperators.rm_leader_dual import RmLeaderDual  # noqa: F401  注册 config 选项
from deployment.teleoperators.rm_leader_left import RmLeaderLeft  # noqa: F401  注册 config 选项
from deployment.teleoperators.rm_leader_right import RmLeaderRight  # noqa: F401  注册 config 选项
from tools.tactile_uint16_to_uint8 import tactile_uint16_to_uint8

# ---- 策略 / 数据集 / 处理管线 (复用本仓库 vtla) ----
from vtla.engine.configs import parser
from vtla.engine.configs.policies import PreTrainedConfig
from vtla.datasets.image_writer import safe_stop_image_writer
from vtla.datasets.lerobot_dataset import LeRobotDataset
from vtla.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from vtla.datasets.video_utils import VideoEncodingManager
from vtla.engine.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from vtla.frameworks.factory import make_policy, make_pre_post_processors
from vtla.frameworks.pretrained import PreTrainedPolicy
from vtla.frameworks.utils import make_robot_action
from vtla.engine.types import PolicyAction, RobotAction, RobotObservation
from vtla.engine.processor.pipeline import PolicyProcessorPipeline, RobotProcessorPipeline
from vtla.engine.processor.factory import make_default_processors
from vtla.engine.processor.rename_processor import rename_stats
from vtla.engine.utils.constants import ACTION, OBS_STR, HF_LEROBOT_HOME
from vtla.engine.common.control_utils import (
    init_keyboard_listener,
    is_headless,  # noqa: F401
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from vtla.engine.utils.device_utils import get_safe_torch_device
from vtla.engine.utils.utils import init_logging, log_say
from vtla.engine.utils.visualization_utils import init_rerun, log_rerun_data


class StickyHint:
    """把一行提示钉在终端最底行: 后台线程每 0.5s 重绘, 被其他输出刷掉也会马上回来。

    用 ANSI: 保存光标 -> 跳到底行 -> 清行 -> 写提示 -> 恢复光标。非 tty 则空操作。
    """

    def __init__(self, text: str):
        self.text = text
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _loop(self):
        while not self._stop.is_set():
            sys.stdout.write(f"\0337\033[999;1H\033[K{self.text}\0338")
            sys.stdout.flush()
            self._stop.wait(0.5)

    def __enter__(self):
        if sys.stdout.isatty():
            self._t = threading.Thread(target=self._loop, daemon=True, name="StickyHint")
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1.0)
        if sys.stdout.isatty():
            sys.stdout.write("\033[999;1H\033[K")
            sys.stdout.flush()


def busy_wait(seconds: float) -> None:
    """精确控频等待。Mac/Windows 上 time.sleep 不够精准，用忙等。"""
    if platform.system() in ("Darwin", "Windows"):
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        if seconds > 0:
            time.sleep(seconds)


def capture_home_action(robot: Robot) -> dict[str, float]:
    """Capture one fixed startup reset target after the robot connects."""
    config = getattr(robot, "config", None)
    joint_names: list[str] = getattr(robot, "JOINT_NAMES", [])
    sides = list(getattr(robot, "_arms", {}))
    gripper_name: str = getattr(robot, "GRIPPER_NAME", "gripper")
    home_gripper: float = getattr(config, "home_gripper", 1.0)
    if not joint_names or not sides:
        raise RuntimeError(
            f"robot {getattr(robot, 'name', type(robot).__name__)!r} does not expose "
            "JOINT_NAMES and active arms for startup home capture"
        )

    joint_keys = [f"{side}_{joint}" for side in sides for joint in joint_names]
    captured_joints = getattr(robot, "_home_joints", None)
    if isinstance(captured_joints, dict) and all(side in captured_joints for side in sides):
        source = {
            f"{side}_{joint}": float(captured_joints[side][index])
            for side in sides
            for index, joint in enumerate(joint_names)
        }
    else:
        # Fallback for robot implementations without a connection-time capture.
        source = robot.get_observation()
        logging.warning(
            "[home] robot 未提供连接时关节快照，退回到 connect() 后首次观测"
        )

    missing = [key for key in joint_keys if key not in source]
    if missing:
        raise RuntimeError(f"startup observation is missing home joint(s): {missing}")

    home = {key: float(source[key]) for key in joint_keys}
    for side in sides:
        home[f"{side}_{gripper_name}"] = float(home_gripper)
    logging.info(
        "[home] 固定复位目标来自机械臂连接时关节快照，仅捕获一次: joints=%s, gripper=%.3f",
        {key: home[key] for key in joint_keys},
        home_gripper,
    )
    return home


def move_to_home_smooth(
    robot: Robot,
    home_action: dict[str, float],
    fps: int,
    duration_s: float,
) -> bool:
    """从当前姿态余弦插值到 home，然后根据关节反馈等待实际到位。"""
    logging.info("[home] 本次复位使用固定启动目标: %s", dict(home_action))
    first_obs = robot.get_observation()
    start = {k: float(first_obs[k]) for k in home_action if k in first_obs}

    config = getattr(robot, "config", None)
    tolerance_deg = float(getattr(config, "home_joint_tolerance_deg", 1.0))
    use_degrees = bool(getattr(config, "use_degrees", False))
    tolerance = tolerance_deg if use_degrees else math.radians(tolerance_deg)
    settle_timeout_s = float(getattr(config, "home_settle_timeout_s", 2.0))
    joint_targets = {
        key: float(value) for key, value in home_action.items() if "joint" in key
    }

    steps = max(1, int(duration_s * fps))
    for i in range(1, steps + 1):
        t0 = time.perf_counter()
        progress = i / steps
        alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        target = {
            k: start.get(k, home_action[k]) * (1 - alpha) + home_action[k] * alpha
            for k in home_action
        }
        robot.send_action(target)
        busy_wait(1 / fps - (time.perf_counter() - t0))

    settle_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        obs = robot.get_observation()
        robot.send_action(home_action)
        errors = {
            key: abs(float(obs[key]) - target)
            for key, target in joint_targets.items()
            if key in obs
        }
        all_joints_at_home = len(errors) == len(joint_targets) and all(
            error <= tolerance for error in errors.values()
        )
        if all_joints_at_home:
            logging.info(
                "[home] 关节反馈已到位: max_error=%.3f°, tolerance=%.3f°",
                math.degrees(max(errors.values(), default=0.0))
                if not use_degrees
                else max(errors.values(), default=0.0),
                tolerance_deg,
            )
            return True
        if time.perf_counter() - settle_start >= settle_timeout_s:
            max_error = max(errors.values(), default=float("inf"))
            max_error_deg = (
                max_error if use_degrees else math.degrees(max_error)
            )
            logging.warning(
                "[home] 等待关节到位超时 %.2fs: max_error=%.3f°, tolerance=%.3f°",
                settle_timeout_s,
                max_error_deg,
                tolerance_deg,
            )
            return False
        busy_wait(1 / fps - (time.perf_counter() - t0))


def wait_for_episode_start(
    *,
    events: dict,
    episode_label: str,
    play_sounds: bool,
    on_prepared: Callable[[], None] | None = None,
) -> bool:
    """Wait for and consume an explicit start request from the operator."""
    events["start_episode"] = False
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["toggle_gripper"] = 0

    log_say(
        f"{episode_label}: ↑开始 | →复位并保存 | ←复位并重录 | ESC退出",
        play_sounds,
    )

    while not events["start_episode"] and not events["stop_recording"]:
        time.sleep(0.05)

    if events["stop_recording"]:
        return False

    events["start_episode"] = False
    if on_prepared is not None:
        on_prepared()
    return True


def wait_for_inflight_policy_action(fps: int) -> None:
    """Temporarily allow ten control periods for submitted policy motion to settle."""
    wait_frames = 10
    wait_s = wait_frames / fps
    logging.info(
        "[reset] 等待最后一个已下发模型动作完成: %.3fs (%d 帧, %d Hz)",
        wait_s,
        wait_frames,
        fps,
    )
    busy_wait(wait_s)


def reset_then_finalize_episode(
    *,
    robot: Robot,
    events: dict,
    reset_before_episode: bool,
    home_action: dict[str, float],
    fps: int,
    home_duration_s: float,
    play_sounds: bool,
    episode_label: str,
    finalize: Callable[[], None],
) -> bool:
    """Reset to the connection-time home, then save or discard an episode.

    Reset observations are intentionally not exposed to episode writers. A
    failed feedback confirmation retries and never finalizes the episode.
    """
    if events["stop_recording"]:
        return False
    if not reset_before_episode:
        finalize()
        return True
    if not home_action:
        raise RuntimeError("reset_before_episode=True requires a non-empty home action")

    while not events["stop_recording"]:
        log_say(f"{episode_label}: 机械臂按连接初始位姿复位中...", play_sounds)
        if move_to_home_smooth(robot, home_action, fps, home_duration_s):
            if events["stop_recording"]:
                return False
            log_say(f"{episode_label}: 复位已确认", play_sounds)
            finalize()
            return True
        log_say("复位未确认到位，保持当前 episode 并自动重试", play_sounds)

    return False


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}'.
    repo_id: str
    # A short but accurate description of the task performed during the recording.
    # 可选: collect 必填; inference 用 --match-policy 时从训练集自动获取。
    single_task: str | None = None
    # Root directory where the dataset will be stored.
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 600
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub. 默认 False: 本地采集/推理优先, 需要才显式开。
    push_to_hub: bool = False
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    video_encoding_batch_size: int = 1
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    # 保存模式：episode (标准 LeRobot 数据集) / stream (仅保存 stream/ 下视频 sidecar)
    save: str = "episode"
    # 统一统计开关 (stream 模式默认 False, episode 默认 True)
    compute_stats: bool | None = None

    def __post_init__(self):
        # single_task 的必填校验移到各入口 (collect 必填; inference 可由 --match-policy 自动填)。
        if self.fps <= 0:
            raise ValueError(f"`dataset.fps` must be a positive integer, got: {self.fps}")
        if self.save not in {"episode", "stream"}:
            raise ValueError(f"`dataset.save` must be one of ['episode', 'stream'], got: {self.save}")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    # Whether the robot returns home before every episode. Entry points set
    # this explicitly: collect=False, inference=True.
    reset_before_episode: bool = True

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@dataclass
class StreamPolicyMeta:
    """用于 stream 模式下 policy 初始化的轻量 ds_meta 占位对象。"""
    features: dict[str, dict[str, Any]]
    stats: dict[str, Any] | None = None


def resolve_compute_stats(save_mode: str, compute_stats: bool | None) -> bool:
    """统一解析统计开关：stream 默认 False，episode 默认 True。"""
    if compute_stats is not None:
        return compute_stats
    return save_mode == "episode"


def resolve_dataset_root(dataset_cfg: DatasetRecordConfig) -> Path:
    """解析数据集根目录，保证 stream 与 episode 的 root 规则一致。"""
    return Path(dataset_cfg.root) if dataset_cfg.root is not None else HF_LEROBOT_HOME / dataset_cfg.repo_id


class StreamVideoWriter:
    """stream 模式视频写入器：仅写固定相机，不生成任何 meta 文件。"""

    def __init__(
        self,
        stream_root: Path,
        fps: int,
        camera_keys: tuple[str, ...] = ("cam_top", "cam_right_wrist"),
    ):
        self.stream_root = Path(stream_root)
        self.fps = fps
        self.camera_keys = camera_keys
        self.stream_root.mkdir(parents=True, exist_ok=True)

        self._episode_dir: Path | None = None
        self._writers: dict[str, Any] = {}
        self._frame_sizes: dict[str, tuple[int, int]] = {}
        self._resize_warned_keys: set[str] = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def start_episode(self, episode_index: int) -> None:
        self.close_episode(discard=False)
        self._episode_dir = self.stream_root / f"episode-{episode_index:06d}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)

    def _make_writer(self, camera_key: str, frame: np.ndarray):
        import cv2

        if self._episode_dir is None:
            raise RuntimeError("Stream episode is not started")

        height, width = int(frame.shape[0]), int(frame.shape[1])
        self._frame_sizes[camera_key] = (width, height)
        video_path = self._episode_dir / f"{camera_key}.mp4"

        for fourcc_tag in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*fourcc_tag),
                float(self.fps),
                (width, height),
            )
            if writer.isOpened():
                return writer
            writer.release()

        raise RuntimeError(f"Failed to create stream video writer for {camera_key}: {video_path}")

    def _normalize_frame(self, frame: Any, camera_key: str) -> np.ndarray | None:
        import cv2

        if frame is None:
            return None

        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = frame.transpose(1, 2, 0)

        if frame.ndim != 3 or frame.shape[-1] not in (1, 3):
            logging.warning(
                f"Skip stream frame for {camera_key}: unsupported shape {getattr(frame, 'shape', None)}"
            )
            return None

        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                frame = np.clip(frame, 0.0, 1.0)
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        expected_size = self._frame_sizes.get(camera_key)
        if expected_size is not None:
            width, height = expected_size
            if frame.shape[1] != width or frame.shape[0] != height:
                if camera_key not in self._resize_warned_keys:
                    logging.warning(
                        f"Resize stream frame for {camera_key} from {frame.shape[1]}x{frame.shape[0]} "
                        f"to {width}x{height}"
                    )
                    self._resize_warned_keys.add(camera_key)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        return frame

    def add_observation(self, obs: dict[str, Any]) -> None:
        import cv2

        if self._episode_dir is None:
            raise RuntimeError("Stream episode is not started")

        for camera_key in self.camera_keys:
            frame = obs.get(camera_key)
            if frame is None:
                logging.warning(f"Missing frame for stream camera '{camera_key}', skipping this frame.")
                continue

            normalized_frame = self._normalize_frame(frame, camera_key)
            if normalized_frame is None:
                continue

            writer = self._writers.get(camera_key)
            if writer is None:
                writer = self._make_writer(camera_key, normalized_frame)
                self._writers[camera_key] = writer

            bgr_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)

    def close_episode(self, discard: bool = False) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        self._frame_sizes.clear()
        self._resize_warned_keys.clear()

        if discard and self._episode_dir is not None and self._episode_dir.exists():
            shutil.rmtree(self._episode_dir, ignore_errors=True)

        self._episode_dir = None

    def close(self) -> None:
        self.close_episode(discard=False)


def _tactile_camera_keys(robot: Robot) -> tuple[str, ...]:
    """Return raw uint16 tactile observation keys in stable hardware-feature order."""
    return tuple(
        key
        for key, feature in robot.observation_features.items()
        if isinstance(feature, tuple) and ("cam_finger" in key or "tactile" in key)
    )


def _with_external_tactile_video_features(
    features: dict[str, dict[str, Any]], robot: Robot, tactile_keys: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Mark tactile observations as externally encoded lossless dataset videos."""
    result = {key: value.copy() for key, value in features.items()}
    for camera_key in tactile_keys:
        dataset_key = f"{OBS_STR}.images.{camera_key}"
        feature = result.get(
            dataset_key,
            {
                "dtype": "video",
                "shape": robot.observation_features[camera_key],
                "names": ["height", "width", "channels"],
            },
        )
        feature.update(
            {
                "dtype": "video",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mkv",
                "external_video": True,
                "tactile_encoding": "tactile_u16_fixed_v1",
                "storage_dtype": "uint16",
            }
        )
        result[dataset_key] = feature
    return result


def _publish_stream_tactile_videos(
    dataset_root: Path, episode_index: int, paths: dict[str, Path]
) -> None:
    """Place stream-mode tactile episodes under the standard videos hierarchy."""
    chunk_index, file_index = divmod(int(episode_index), 1000)
    for camera_key, source in paths.items():
        destination = (
            dataset_root
            / "videos"
            / f"{OBS_STR}.images.{camera_key}"
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mkv"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Tactile stream video already exists: {destination}")
        shutil.move(str(source), str(destination))
    if paths:
        with contextlib.suppress(OSError):
            next(iter(paths.values())).parent.rmdir()


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


def build_drag_action(
    robot: Robot,
    observation: RobotObservation,
    gripper_target: float | None,
) -> RobotAction:
    """Build drag supervision from measured joints and the active gripper target."""
    missing = [key for key in robot.action_features if key not in observation]
    if missing:
        raise ValueError(
            "drag mode requires action features to be present in the measured observation; "
            f"missing: {missing}"
        )
    action = {key: float(observation[key]) for key in robot.action_features}
    if gripper_target is not None:
        for key in action:
            if "gripper" in key:
                action[key] = float(gripper_target)
    return action


def toggle_drag_gripper(
    robot: Robot,
    observation: RobotObservation,
    current_target: float | None,
    *,
    open_value: float,
    close_value: float,
) -> float:
    """Toggle all drag grippers and return the newly commanded target."""
    if current_target is None:
        measured = [float(v) for k, v in observation.items() if "gripper" in k]
        if not measured:
            raise ValueError("drag mode could not find a measured gripper position")
        currently_open = float(np.mean(measured)) >= (open_value + close_value) / 2.0
    else:
        currently_open = abs(current_target - open_value) <= abs(current_target - close_value)

    target = close_value if currently_open else open_value
    send_gripper = getattr(robot, "send_gripper_action", None)
    if not callable(send_gripper):
        raise TypeError(f"robot {robot.name!r} does not support independent gripper control")
    send_gripper(target)
    logging.info("[drag] 夹爪目标切换为 %.3f (%s)", target, "张开" if target == open_value else "闭合")
    return target


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    record_features: dict[str, dict[str, Any]] | None = None,
    stream_writer: StreamVideoWriter | None = None,
    tactile_writer: TactileMkvWriter | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    drag_mode: bool = False,
    drag_gripper_open_value: float = 1.0,
    drag_gripper_close_value: float = 0.0,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    drag_gripper_target: float | None = None
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        if drag_mode:
            toggle_count = int(events.get("toggle_gripper", 0))
            events["toggle_gripper"] = 0
            for _ in range(toggle_count):
                drag_gripper_target = toggle_drag_gripper(
                    robot,
                    obs,
                    drag_gripper_target,
                    open_value=drag_gripper_open_value,
                    close_value=drag_gripper_close_value,
                )

        # The authoritative tactile observation remains uint16 for the MKV writer. Policy,
        # display and optional inference recordings consume the versioned uint8 derivative.
        obs_for_processing = obs
        if tactile_writer is not None:
            obs_for_processing = obs.copy()
            for camera_key in tactile_writer.camera_keys:
                obs_for_processing[camera_key] = tactile_uint16_to_uint8(obs[camera_key])

        # Applies a pipeline to the model/standard-dataset observation.
        obs_processed = robot_observation_processor(obs_for_processing)

        # stream + policy 场景下 dataset 可能为 None，因此统一使用 features_for_frame
        features_for_frame = dataset.features if dataset is not None else record_features

        observation_frame = None
        if policy is not None or dataset is not None:
            if features_for_frame is None:
                raise ValueError(
                    "record_features is required when policy is enabled or dataset is None in record_loop."
                )
            observation_frame = build_dataset_frame(features_for_frame, obs_processed, prefix=OBS_STR)

        # Get action from either policy or teleop
        act_processed_policy: RobotAction | None = None
        act_processed_teleop: RobotAction | None = None
        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )

            if features_for_frame is None:
                raise ValueError("record_features is required for policy action conversion.")
            act_processed_policy = make_robot_action(action_values, features_for_frame)

        elif drag_mode:
            action_values = build_drag_action(robot, obs, drag_gripper_target)
        elif policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Applies a pipeline to the action, default is IdentityProcessor
        if drag_mode:
            robot_action_to_send = None
        elif policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot. Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        if robot_action_to_send is not None:
            robot.send_action(robot_action_to_send)

        if tactile_writer is not None:
            tactile_writer.add_observation(obs)

        # Write to dataset
        if dataset is not None:
            if features_for_frame is None:
                raise ValueError("record_features is required for dataset frame construction.")
            action_frame = build_dataset_frame(features_for_frame, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)
        elif stream_writer is not None:
            # stream 模式仅保存视频 sidecar，不写标准 dataset 帧
            stream_writer.add_observation(obs_processed)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


def run_record(cfg: RecordConfig) -> LeRobotDataset | None:
    """录制引擎主流程。由 collect.py / inference.py 构建好 RecordConfig 后调用。"""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    mode = getattr(cfg, "mode", "teleop")
    mode = getattr(mode, "value", mode)
    drag_mode = mode == "drag"
    # 解析保存模式与统计开关 (stream 默认不计算统计)
    save_mode = cfg.dataset.save
    compute_stats_enabled = resolve_compute_stats(save_mode, cfg.dataset.compute_stats)

    if save_mode == "stream" and compute_stats_enabled:
        logging.info("`dataset.save=stream` currently does not compute dataset stats, forcing stats off.")

    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = (
        make_teleoperator_from_config(cfg.teleop)
        if cfg.teleop is not None and not drag_mode
        else None
    )
    if drag_mode:
        for method_name in ("start_force_drag", "stop_force_drag", "send_gripper_action"):
            if not callable(getattr(robot, method_name, None)):
                raise TypeError(f"robot {robot.name!r} does not support drag method {method_name}()")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    all_dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    tactile_keys = _tactile_camera_keys(robot)
    dataset_features = _with_external_tactile_video_features(
        all_dataset_features, robot, tactile_keys
    )
    record_features = dataset_features
    if tactile_keys:
        logging.info(
            "Raw tactile features are stored as uint16 MKV under videos/ and indexed "
            "as dataset video features: %s",
            tactile_keys,
        )

    dataset: LeRobotDataset | None = None
    if save_mode == "episode":
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )

            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # episode 模式创建标准 LeRobotDataset
            # 推理模式 (有 policy) 用流式编码: 直接写 MP4, 不落 PNG 文件
            _inference_mode = cfg.policy is not None
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.robot_type,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                streaming_encoding=_inference_mode,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=0 if _inference_mode else cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )
    else:
        # stream 模式不创建 LeRobotDataset，只做 repo_id 合法性检查
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)

    tactile_writer = (
        TactileMkvWriter(
            root=resolve_dataset_root(cfg.dataset) / ".tactile_staging",
            fps=cfg.dataset.fps,
            camera_keys=tactile_keys,
            manifest_path=resolve_dataset_root(cfg.dataset) / "meta" / "tactile_encoding.json",
        )
        if tactile_keys
        else None
    )

    # 按当前模式加载策略；stream 模式使用轻量 ds_meta 占位
    if cfg.policy is not None:
        if dataset is not None:
            ds_meta_for_policy = dataset.meta
            dataset_stats = rename_stats(dataset.meta.stats, cfg.dataset.rename_map)
        else:
            ds_meta_for_policy = StreamPolicyMeta(features=record_features, stats=None)
            dataset_stats = rename_stats({}, cfg.dataset.rename_map)
        policy = make_policy(cfg.policy, ds_meta=ds_meta_for_policy)
    else:
        policy = None

    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=dataset_stats,
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

    robot.connect()
    startup_home_action: dict[str, float] = (
        capture_home_action(robot) if cfg.reset_before_episode else {}
    )
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    force_drag_active = False

    def start_force_drag() -> None:
        nonlocal force_drag_active
        if not drag_mode or force_drag_active:
            return
        robot.start_force_drag(
            precise=bool(getattr(cfg, "drag_force_precise", True)),
            mode=int(getattr(cfg, "drag_force_mode", 3)),
            singular_wall=bool(getattr(cfg, "drag_singular_wall", True)),
        )
        force_drag_active = True
        log_say("六维力拖动已开启", cfg.play_sounds)

    def stop_force_drag() -> None:
        nonlocal force_drag_active
        if not force_drag_active:
            return
        robot.stop_force_drag()
        force_drag_active = False
        log_say("六维力拖动已关闭", cfg.play_sounds)

    try:
        # A reset sends pose commands, so force drag must not own the arm at
        # the same time. Without reset, keep drag active across episodes.
        if drag_mode and not cfg.reset_before_episode:
            start_force_drag()

        if save_mode == "episode":
            if dataset is None:
                raise RuntimeError("Episode mode requires a valid dataset instance.")

            with VideoEncodingManager(dataset):
                recorded_episodes = 0
                _home_duration = getattr(getattr(robot, "config", None), "home_duration_s", 4.0)
                while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                    if not wait_for_episode_start(
                        events=events,
                        episode_label=(
                            f"Episode {dataset.num_episodes + 1}/{cfg.dataset.num_episodes}"
                        ),
                        play_sounds=cfg.play_sounds,
                        on_prepared=(
                            start_force_drag
                            if drag_mode and cfg.reset_before_episode
                            else None
                        ),
                    ):
                        continue

                    logging.info(f"开始录制 episode {dataset.num_episodes + 1}")
                    if tactile_writer is not None:
                        tactile_writer.start_episode(dataset.num_episodes)

                    # ── 主录制循环 ────────────────────────────────────────────
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        dataset=dataset,
                        record_features=record_features,
                        tactile_writer=tactile_writer,
                        control_time_s=cfg.dataset.episode_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        drag_mode=drag_mode,
                        drag_gripper_open_value=float(
                            getattr(cfg, "drag_gripper_open_value", 1.0)
                        ),
                        drag_gripper_close_value=float(
                            getattr(cfg, "drag_gripper_close_value", 0.0)
                        ),
                    )

                    if (
                        policy is not None
                        and cfg.reset_before_episode
                        and not events["stop_recording"]
                    ):
                        wait_for_inflight_policy_action(cfg.dataset.fps)

                    if drag_mode and cfg.reset_before_episode:
                        stop_force_drag()

                    episode_label = f"Episode {dataset.num_episodes + 1}"

                    # ── 左键: 先复位并确认，再丢弃本次 episode ───────────────
                    if events["rerecord_episode"]:
                        def discard_episode() -> None:
                            events["rerecord_episode"] = False
                            events["exit_early"] = False
                            if tactile_writer is not None:
                                tactile_writer.close_episode(discard=True)
                            dataset.clear_episode_buffer()

                        reset_then_finalize_episode(
                            robot=robot,
                            events=events,
                            reset_before_episode=cfg.reset_before_episode,
                            home_action=startup_home_action,
                            fps=cfg.dataset.fps,
                            home_duration_s=_home_duration,
                            play_sounds=cfg.play_sounds,
                            episode_label=episode_label,
                            finalize=discard_episode,
                        )
                        continue

                    # ── 右键/超时: 先复位并确认，再保存本次 episode ───────────
                    def save_episode() -> None:
                        if tactile_writer is not None:
                            tactile_paths = tactile_writer.close_episode(discard=False)
                            for camera_key, path in tactile_paths.items():
                                dataset.register_external_video(
                                    f"{OBS_STR}.images.{camera_key}", path
                                )
                        dataset.save_episode()
                        if tactile_writer is not None:
                            tactile_writer.cleanup_staging()

                    finalized = reset_then_finalize_episode(
                        robot=robot,
                        events=events,
                        reset_before_episode=cfg.reset_before_episode,
                        home_action=startup_home_action,
                        fps=cfg.dataset.fps,
                        home_duration_s=_home_duration,
                        play_sounds=cfg.play_sounds,
                        episode_label=episode_label,
                        finalize=save_episode,
                    )
                    if finalized:
                        events["exit_early"] = False
                        recorded_episodes += 1
        else:
            # stream 模式使用 sidecar 视频写入，不触碰 LeRobotDataset 计数逻辑
            stream_root = resolve_dataset_root(cfg.dataset) / "stream"
            with StreamVideoWriter(stream_root=stream_root, fps=cfg.dataset.fps) as stream_writer:
                episode_start = 0
                if cfg.resume and stream_root.exists():
                    existing = sorted(stream_root.glob("episode-*"))
                    if existing:
                        last_name = existing[-1].name
                        try:
                            episode_start = int(last_name.split("-", 1)[1]) + 1
                        except (ValueError, IndexError):
                            episode_start = len(existing)
                        logging.info(f"Stream resume: found {len(existing)} existing episodes, starting from episode {episode_start}")

                recorded_episodes = 0
                episode_index = episode_start
                _home_duration_s = getattr(getattr(robot, "config", None), "home_duration_s", 4.0)
                while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                    if not wait_for_episode_start(
                        events=events,
                        episode_label=f"Stream episode {episode_index + 1}",
                        play_sounds=cfg.play_sounds,
                        on_prepared=(
                            start_force_drag
                            if drag_mode and cfg.reset_before_episode
                            else None
                        ),
                    ):
                        continue

                    logging.info(f"开始录制 stream episode {episode_index + 1}")
                    stream_writer.start_episode(episode_index)
                    if tactile_writer is not None:
                        tactile_writer.start_episode(episode_index)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        dataset=None,
                        record_features=record_features,
                        stream_writer=stream_writer,
                        tactile_writer=tactile_writer,
                        control_time_s=cfg.dataset.episode_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        drag_mode=drag_mode,
                        drag_gripper_open_value=float(
                            getattr(cfg, "drag_gripper_open_value", 1.0)
                        ),
                        drag_gripper_close_value=float(
                            getattr(cfg, "drag_gripper_close_value", 0.0)
                        ),
                    )

                    if (
                        policy is not None
                        and cfg.reset_before_episode
                        and not events["stop_recording"]
                    ):
                        wait_for_inflight_policy_action(cfg.dataset.fps)

                    if drag_mode and cfg.reset_before_episode:
                        stop_force_drag()

                    episode_label = f"Stream episode {episode_index + 1}"

                    # ── 左键: 先复位并确认，再丢弃本次视频 ───────────────────
                    if events["rerecord_episode"]:
                        def discard_stream_episode() -> None:
                            events["rerecord_episode"] = False
                            events["exit_early"] = False
                            stream_writer.close_episode(discard=True)
                            if tactile_writer is not None:
                                tactile_writer.close_episode(discard=True)

                        reset_then_finalize_episode(
                            robot=robot,
                            events=events,
                            reset_before_episode=cfg.reset_before_episode,
                            home_action=startup_home_action,
                            fps=cfg.dataset.fps,
                            home_duration_s=_home_duration_s,
                            play_sounds=cfg.play_sounds,
                            episode_label=episode_label,
                            finalize=discard_stream_episode,
                        )
                        continue

                    # ── 右键/超时: 先复位并确认，再保存本次视频 ───────────────
                    def save_stream_episode() -> None:
                        stream_writer.close_episode(discard=False)
                        if tactile_writer is not None:
                            tactile_paths = tactile_writer.close_episode(discard=False)
                            _publish_stream_tactile_videos(
                                resolve_dataset_root(cfg.dataset), episode_index, tactile_paths
                            )
                            tactile_writer.cleanup_staging()

                    finalized = reset_then_finalize_episode(
                        robot=robot,
                        events=events,
                        reset_before_episode=cfg.reset_before_episode,
                        home_action=startup_home_action,
                        fps=cfg.dataset.fps,
                        home_duration_s=_home_duration_s,
                        play_sounds=cfg.play_sounds,
                        episode_label=episode_label,
                        finalize=save_stream_episode,
                    )
                    if finalized:
                        events["exit_early"] = False
                        recorded_episodes += 1
                        episode_index += 1
    finally:
        if force_drag_active:
            try:
                stop_force_drag()
            except Exception:
                logging.exception("停止六维力拖动失败")

        if tactile_writer is not None:
            try:
                tactile_writer.abort_episode()
            except Exception:
                logging.exception("清理触觉 episode writer 失败")

        # 优先停止策略推理线程
        if policy is not None and hasattr(policy, "stop"):
            policy.stop()

        log_say("Stop recording", cfg.play_sounds, blocking=True)

        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()

        if listener is not None:
            listener.stop()

    if cfg.dataset.push_to_hub:
        if save_mode == "episode" and dataset is not None:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        else:
            logging.info("Skipping push_to_hub in stream mode (stream outputs are non-standard sidecar videos).")

    log_say("Exiting", cfg.play_sounds)
    return dataset
