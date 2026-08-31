"""Asynchronous policy inference runtime.

The hardware worker is the sole owner of the Robot instance. It publishes one
coherent latest observation and accepts at most one unacknowledged command. A
separate inference worker continuously replaces a latest-completed-chunk slot.
The main process owns episode control, recording, and the immutable queue for a
chunk that has already been claimed for execution.
"""

from __future__ import annotations

import ctypes
import logging
import multiprocessing as mp
import queue
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import numpy as np

from deployment._record_engine import (
    StreamPolicyMeta,
    StreamVideoWriter,
    _publish_stream_tactile_videos,
    _tactile_camera_keys,
    _with_external_tactile_video_features,
    busy_wait,
    capture_home_action,
    reset_then_finalize_episode,
    resolve_compute_stats,
    resolve_dataset_root,
    wait_for_episode_start,
    wait_for_inflight_policy_action,
)
from deployment.hardware.tactile_sensors import TactileMkvWriter
from deployment.robots import RobotConfig, make_robot_from_config
from tools.tactile_uint16_to_uint8 import tactile_uint16_to_uint8
from vtla.datasets.image_writer import safe_stop_image_writer
from vtla.datasets.lerobot_dataset import LeRobotDataset
from vtla.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from vtla.datasets.video_utils import VideoEncodingManager
from vtla.engine.common.control_utils import (
    init_keyboard_listener,
    preprocess_policy_observation,
    predict_action_chunk,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from vtla.engine.processor.factory import make_default_processors
from vtla.engine.processor.rename_processor import rename_stats
from vtla.engine.utils.constants import ACTION, OBS_STR
from vtla.engine.utils.device_utils import get_safe_torch_device
from vtla.engine.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from vtla.engine.utils.utils import init_logging, log_say
from vtla.engine.utils.visualization_utils import init_rerun, log_rerun_data
from vtla.frameworks.factory import make_policy, make_pre_post_processors


@dataclass(frozen=True)
class _SharedArraySpec:
    key: str
    shape: tuple[int, ...]
    dtype: str
    buffer: Any


@dataclass(frozen=True)
class ObservationSnapshot:
    version: int
    timestamp: float
    values: dict[str, Any]


def _observation_dtype(key: str, feature: Any) -> np.dtype:
    if not isinstance(feature, tuple):
        return np.dtype(np.float64)
    if "cam_finger" in key or "tactile" in key:
        return np.dtype(np.uint16)
    if "cam" in key or "image" in key:
        return np.dtype(np.uint8)
    return np.dtype(np.float32)


def _ctype_for_dtype(dtype: np.dtype):
    return {
        np.dtype(np.uint8): ctypes.c_uint8,
        np.dtype(np.uint16): ctypes.c_uint16,
        np.dtype(np.float32): ctypes.c_float,
        np.dtype(np.float64): ctypes.c_double,
    }[dtype]


class SharedObservationBuffer:
    """A coherent observation ring whose public control view is latest-only."""

    def __init__(self, specs, lock, version, timestamps, history_size):
        self.specs: tuple[_SharedArraySpec, ...] = tuple(specs)
        self.lock = lock
        self.version = version
        self.timestamps = timestamps
        self.history_size = int(history_size)

    @classmethod
    def create(cls, observation_features: dict[str, Any], ctx, history_size: int = 1):
        if history_size <= 0:
            raise ValueError(f"history_size must be positive, got {history_size}")
        specs = []
        for key, feature in observation_features.items():
            shape = tuple(feature) if isinstance(feature, tuple) else ()
            dtype = _observation_dtype(key, feature)
            size = history_size * max(1, int(np.prod(shape, dtype=np.int64)))
            specs.append(
                _SharedArraySpec(
                    key=key,
                    shape=shape,
                    dtype=dtype.str,
                    buffer=ctx.RawArray(_ctype_for_dtype(dtype), size),
                )
            )
        return cls(
            specs,
            ctx.Lock(),
            ctx.Value("q", 0),
            ctx.RawArray(ctypes.c_double, history_size),
            history_size,
        )

    def _array(self, spec: _SharedArraySpec) -> np.ndarray:
        array = np.frombuffer(spec.buffer, dtype=np.dtype(spec.dtype))
        return (
            array.reshape((self.history_size, *spec.shape))
            if spec.shape
            else array.reshape((self.history_size,))
        )

    def _read_version_locked(self, version: int) -> ObservationSnapshot:
        index = (version - 1) % self.history_size
        values: dict[str, Any] = {}
        for spec in self.specs:
            value = self._array(spec)[index].copy()
            values[spec.key] = value.item() if not spec.shape else value
        timestamps = np.frombuffer(self.timestamps, dtype=np.float64)
        return ObservationSnapshot(
            version=version,
            timestamp=float(timestamps[index]),
            values=values,
        )

    def publish(self, observation: dict[str, Any]) -> int:
        missing = [spec.key for spec in self.specs if spec.key not in observation]
        if missing:
            raise KeyError(f"Observation is missing shared feature(s): {missing}")
        with self.lock:
            next_version = int(self.version.value) + 1
            index = (next_version - 1) % self.history_size
            for spec in self.specs:
                target_array = self._array(spec)
                value = np.asarray(observation[spec.key], dtype=np.dtype(spec.dtype))
                if value.shape != spec.shape:
                    raise ValueError(
                        f"Observation {spec.key!r} has shape {value.shape}, expected {spec.shape}"
                    )
                if spec.shape:
                    target_array[index, ...] = value
                else:
                    target_array[index] = value
            np.frombuffer(self.timestamps, dtype=np.float64)[index] = time.monotonic()
            self.version.value = next_version
            return next_version

    def read(self) -> ObservationSnapshot | None:
        with self.lock:
            version = int(self.version.value)
            if version == 0:
                return None
            return self._read_version_locked(version)

    def read_since(self, after_version: int) -> list[ObservationSnapshot]:
        """Return retained observations newer than ``after_version`` in order."""
        with self.lock:
            newest = int(self.version.value)
            if newest <= after_version:
                return []
            oldest_retained = max(1, newest - self.history_size + 1)
            first = max(oldest_retained, after_version + 1)
            return [
                self._read_version_locked(version)
                for version in range(first, newest + 1)
            ]


@dataclass(frozen=True)
class ActionChunkSnapshot:
    chunk_id: int
    generation: int
    observation_version: int
    inference_started_at: float
    inference_finished_at: float
    actions: np.ndarray


class SharedActionChunkSlot:
    """A capacity-one slot. Publishing may replace only unclaimed chunks."""

    def __init__(
        self,
        shape,
        buffer,
        lock,
        valid,
        chunk_id,
        generation,
        observation_version,
        inference_started_at,
        inference_finished_at,
    ):
        self.shape = tuple(shape)
        self.buffer = buffer
        self.lock = lock
        self.valid = valid
        self.chunk_id = chunk_id
        self.generation = generation
        self.observation_version = observation_version
        self.inference_started_at = inference_started_at
        self.inference_finished_at = inference_finished_at

    @classmethod
    def create(cls, shape: tuple[int, int], ctx):
        size = int(np.prod(shape, dtype=np.int64))
        return cls(
            shape=shape,
            buffer=ctx.RawArray(ctypes.c_float, size),
            lock=ctx.Lock(),
            valid=ctx.Value("b", False),
            chunk_id=ctx.Value("q", 0),
            generation=ctx.Value("q", 0),
            observation_version=ctx.Value("q", 0),
            inference_started_at=ctx.Value("d", 0.0),
            inference_finished_at=ctx.Value("d", 0.0),
        )

    def _array(self) -> np.ndarray:
        return np.frombuffer(self.buffer, dtype=np.float32).reshape(self.shape)

    def clear(self, generation: int) -> None:
        with self.lock:
            self.valid.value = False
            self.generation.value = generation

    def publish(
        self,
        actions: np.ndarray,
        *,
        generation: int,
        observation_version: int,
        inference_started_at: float,
        inference_finished_at: float,
    ) -> int:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != self.shape:
            raise ValueError(f"Action chunk has shape {actions.shape}, expected {self.shape}")
        with self.lock:
            self._array()[...] = actions
            self.chunk_id.value += 1
            self.generation.value = generation
            self.observation_version.value = observation_version
            self.inference_started_at.value = inference_started_at
            self.inference_finished_at.value = inference_finished_at
            self.valid.value = True
            return int(self.chunk_id.value)

    def claim(self, *, after_chunk_id: int, generation: int) -> ActionChunkSnapshot | None:
        with self.lock:
            if not self.valid.value:
                return None
            chunk_id = int(self.chunk_id.value)
            if chunk_id <= after_chunk_id or int(self.generation.value) != generation:
                return None
            return ActionChunkSnapshot(
                chunk_id=chunk_id,
                generation=int(self.generation.value),
                observation_version=int(self.observation_version.value),
                inference_started_at=float(self.inference_started_at.value),
                inference_finished_at=float(self.inference_finished_at.value),
                actions=self._array().copy(),
            )


class AsyncChunkScheduler:
    """Owns the immutable execution copy of a claimed action chunk."""

    def __init__(self, action_names: list[str]):
        self.action_names = list(action_names)
        self.execution_queue: deque[dict[str, float]] = deque()
        self.last_claimed_chunk_id = 0

    def accept(self, chunk: ActionChunkSnapshot | None, *, generation: int) -> bool:
        if chunk is None or self.execution_queue:
            return False
        if chunk.generation != generation or chunk.chunk_id <= self.last_claimed_chunk_id:
            return False
        if chunk.actions.shape[1] != len(self.action_names):
            raise ValueError(
                f"Chunk action dim {chunk.actions.shape[1]} != names {len(self.action_names)}"
            )
        self.execution_queue.extend(
            {
                name: float(row[index])
                for index, name in enumerate(self.action_names)
            }
            for row in chunk.actions
        )
        self.last_claimed_chunk_id = chunk.chunk_id
        return True

    def pop(self) -> dict[str, float] | None:
        return self.execution_queue.popleft() if self.execution_queue else None

    def reset(self) -> None:
        self.execution_queue.clear()


def _generation_value(generation) -> int:
    with generation.get_lock():
        return int(generation.value)


def _advance_generation(generation) -> int:
    with generation.get_lock():
        generation.value += 1
        return int(generation.value)


def _report_worker_error(name: str, error_queue, shutdown_event) -> None:
    error_queue.put((name, traceback.format_exc()))
    shutdown_event.set()


def _raise_worker_error(error_queue) -> None:
    try:
        name, details = error_queue.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError(f"{name} worker failed:\n{details}")


def _robot_io_worker(
    robot_cfg: RobotConfig,
    observation_buffer: SharedObservationBuffer,
    command_queue,
    response_queue,
    ready_queue,
    ready_event,
    error_queue,
    shutdown_event,
    fps: int,
    reset_before_episode: bool,
) -> None:
    robot = None
    try:
        init_logging()
        robot = make_robot_from_config(robot_cfg)
        robot.connect()
        home_action = capture_home_action(robot) if reset_before_episode else {}
        observation_buffer.publish(robot.get_observation())
        ready_queue.put(home_action)
        ready_event.set()

        observation_period = 1.0 / max(30, fps)
        next_observation_at = time.perf_counter()
        while not shutdown_event.is_set():
            timeout = max(0.0, min(0.05, next_observation_at - time.perf_counter()))
            try:
                command = command_queue.get(timeout=timeout)
            except queue.Empty:
                command = None

            if command is not None:
                request_id, action = command
                sent_action = robot.send_action(action)
                response_queue.put((request_id, sent_action))

            if time.perf_counter() >= next_observation_at:
                observation_buffer.publish(robot.get_observation())
                next_observation_at = time.perf_counter() + observation_period
    except Exception:
        _report_worker_error("robot_io", error_queue, shutdown_event)
    finally:
        if robot is not None and robot.is_connected:
            try:
                robot.disconnect()
            except Exception:
                logging.exception("异步 Robot I/O 进程断开机械臂失败")


def _inference_worker(
    policy_cfg,
    record_features,
    policy_stats,
    processor_stats,
    rename_map,
    robot_type: str,
    tactile_keys: tuple[str, ...],
    single_task: str,
    observation_buffer: SharedObservationBuffer,
    chunk_slot: SharedActionChunkSlot,
    generation,
    inference_enabled_event,
    ready_event,
    error_queue,
    shutdown_event,
) -> None:
    policy = None
    try:
        init_logging()
        ds_meta = StreamPolicyMeta(
            features=record_features,
            stats=policy_stats,
            robot_type=robot_type,
        )
        policy = make_policy(policy_cfg, ds_meta=ds_meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            dataset_stats=processor_stats,
            preprocessor_overrides={
                "device_processor": {"device": policy_cfg.device},
                "rename_observations_processor": {"rename_map": rename_map},
            },
        )
        _, _, robot_observation_processor = make_default_processors()
        device = get_safe_torch_device(policy.config.device)
        ready_event.set()

        active_generation = -1
        last_preprocessed_version = 0
        while not shutdown_event.is_set():
            if not inference_enabled_event.wait(timeout=0.05):
                continue

            current_generation = _generation_value(generation)
            if current_generation != active_generation:
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                active_generation = current_generation
                latest_at_start = observation_buffer.read()
                last_preprocessed_version = (
                    latest_at_start.version - 1 if latest_at_start is not None else 0
                )

            snapshots = observation_buffer.read_since(last_preprocessed_version)
            if not snapshots:
                shutdown_event.wait(0.002)
                continue

            observation_frames = []
            for snapshot in snapshots:
                raw_observation = snapshot.values
                observation_for_policy = raw_observation.copy()
                for camera_key in tactile_keys:
                    observation_for_policy[camera_key] = tactile_uint16_to_uint8(
                        raw_observation[camera_key]
                    )
                observation_processed = robot_observation_processor(observation_for_policy)
                observation_frames.append(
                    build_dataset_frame(
                        record_features, observation_processed, prefix=OBS_STR
                    )
                )
            last_preprocessed_version = snapshots[-1].version

            # Feed retained intermediate frames through the stateful preprocessing
            # pipeline so tactile windows keep control-rate spacing. The newest frame
            # is processed exactly once inside predict_action_chunk below.
            for observation_frame in observation_frames[:-1]:
                preprocess_policy_observation(
                    observation=observation_frame,
                    device=device,
                    preprocessor=preprocessor,
                    task=single_task,
                    robot_type=robot_type,
                )
            observation_frame = observation_frames[-1]
            snapshot = snapshots[-1]

            inference_started_at = time.monotonic()
            actions = predict_action_chunk(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot_type,
            )
            inference_finished_at = time.monotonic()
            if actions.shape[0] != 1:
                raise ValueError(f"Online async inference requires batch size 1, got {actions.shape[0]}")

            if (
                inference_enabled_event.is_set()
                and _generation_value(generation) == current_generation
            ):
                chunk_slot.publish(
                    actions[0].detach().to("cpu").numpy(),
                    generation=current_generation,
                    observation_version=snapshot.version,
                    inference_started_at=inference_started_at,
                    inference_finished_at=inference_finished_at,
                )
    except Exception:
        _report_worker_error("inference", error_queue, shutdown_event)
    finally:
        if policy is not None and hasattr(policy, "stop"):
            policy.stop()


class RobotIOClient:
    """Synchronous main-process facade over the single hardware owner process."""

    def __init__(
        self,
        *,
        config,
        robot_type: str,
        observation_buffer,
        command_queue,
        response_queue,
        error_queue,
        shutdown_event,
    ):
        self.config = config
        self.robot_type = robot_type
        self.observation_buffer = observation_buffer
        self.command_queue = command_queue
        self.response_queue = response_queue
        self.error_queue = error_queue
        self.shutdown_event = shutdown_event
        self._request_id = 0

    def get_observation(self) -> dict[str, Any]:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            _raise_worker_error(self.error_queue)
            snapshot = self.observation_buffer.read()
            if snapshot is not None:
                return snapshot.values
            time.sleep(0.005)
        raise TimeoutError("Timed out waiting for the first asynchronous robot observation")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.shutdown_event.is_set():
            _raise_worker_error(self.error_queue)
            raise RuntimeError("Robot I/O worker is shutting down")
        self._request_id += 1
        request_id = self._request_id
        try:
            self.command_queue.put((request_id, action), timeout=2.0)
        except queue.Full as exc:
            raise TimeoutError("Robot command mailbox did not drain") from exc

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _raise_worker_error(self.error_queue)
            try:
                response_id, sent_action = self.response_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if response_id != request_id:
                raise RuntimeError(
                    f"Robot response id mismatch: expected {request_id}, got {response_id}"
                )
            return sent_action
        raise TimeoutError(f"Robot action {request_id} was not acknowledged")


@safe_stop_image_writer
def async_record_loop(
    *,
    robot: RobotIOClient,
    events: dict,
    fps: int,
    generation: int,
    chunk_slot: SharedActionChunkSlot,
    action_names: list[str],
    robot_action_processor,
    robot_observation_processor,
    error_queue,
    shutdown_event,
    dataset: LeRobotDataset | None,
    record_features: dict[str, dict[str, Any]],
    tactile_keys: tuple[str, ...],
    tactile_writer: TactileMkvWriter | None,
    stream_writer: StreamVideoWriter | None,
    control_time_s: int | float,
    single_task: str,
    display_data: bool,
) -> None:
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should equal requested fps ({dataset.fps} != {fps})")

    scheduler = AsyncChunkScheduler(action_names)
    start_episode_t = time.perf_counter()
    timestamp = 0.0
    while True:
        start_loop_t = time.perf_counter()
        _raise_worker_error(error_queue)
        if shutdown_event.is_set():
            _raise_worker_error(error_queue)
            raise RuntimeError("Async worker stopped unexpectedly")
        if events["exit_early"]:
            if scheduler.execution_queue:
                logging.info(
                    "[async] operator interrupt: discard %d remaining action(s) from chunk %d",
                    len(scheduler.execution_queue),
                    scheduler.last_claimed_chunk_id,
                )
            events["exit_early"] = False
            break

        time_limit_reached = timestamp >= control_time_s
        if time_limit_reached and not scheduler.execution_queue:
            break

        raw_observation = robot.get_observation()
        observation_for_processing = raw_observation.copy()
        for camera_key in tactile_keys:
            observation_for_processing[camera_key] = tactile_uint16_to_uint8(
                raw_observation[camera_key]
            )
        observation_processed = robot_observation_processor(observation_for_processing)
        observation_frame = build_dataset_frame(
            record_features, observation_processed, prefix=OBS_STR
        )

        if not scheduler.execution_queue and not time_limit_reached:
            claimed = chunk_slot.claim(
                after_chunk_id=scheduler.last_claimed_chunk_id,
                generation=generation,
            )
            scheduler.accept(claimed, generation=generation)

        action_values = scheduler.pop()
        if action_values is not None:
            robot_action_to_send = robot_action_processor((action_values, raw_observation))
            sent_action = robot.send_action(robot_action_to_send)

            if tactile_writer is not None:
                tactile_writer.add_observation(raw_observation)
            if dataset is not None:
                action_frame = build_dataset_frame(record_features, sent_action, prefix=ACTION)
                dataset.add_frame({**observation_frame, **action_frame, "task": single_task})
            elif stream_writer is not None:
                stream_writer.add_observation(observation_processed)
            if display_data:
                log_rerun_data(observation=observation_processed, action=sent_action)

        busy_wait(1 / fps - (time.perf_counter() - start_loop_t))
        timestamp = time.perf_counter() - start_episode_t


def _prepare_recording(cfg, robot_template):
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )
    all_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot_template.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot_template.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    tactile_keys = _tactile_camera_keys(robot_template)
    record_features = _with_external_tactile_video_features(
        all_features, robot_template, tactile_keys
    )

    dataset = None
    if cfg.dataset.save == "episode":
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )
            if getattr(robot_template, "cameras", None):
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=(
                        cfg.dataset.num_image_writer_threads_per_camera
                        * len(robot_template.cameras)
                    ),
                )
            sanity_check_dataset_robot_compatibility(
                dataset,
                robot_template,
                cfg.dataset.fps,
                record_features,
            )
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot_template.robot_type,
                features=record_features,
                use_videos=cfg.dataset.video,
                streaming_encoding=True,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=0,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )
    else:
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
    policy_stats = dataset.meta.stats if dataset is not None else None
    processor_stats = rename_stats(policy_stats or {}, cfg.dataset.rename_map)
    return (
        dataset,
        record_features,
        tactile_keys,
        tactile_writer,
        robot_action_processor,
        robot_observation_processor,
        policy_stats,
        processor_stats,
    )


def _wait_worker_ready(name, process, ready_event, error_queue, timeout_s=600.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _raise_worker_error(error_queue)
        if ready_event.wait(timeout=0.1):
            return
        if not process.is_alive():
            _raise_worker_error(error_queue)
            raise RuntimeError(f"{name} worker exited before becoming ready")
    raise TimeoutError(f"Timed out waiting for {name} worker readiness")


def _begin_episode(generation, chunk_slot, inference_enabled_event) -> int:
    current = _advance_generation(generation)
    chunk_slot.clear(current)
    inference_enabled_event.set()
    return current


def _end_episode(generation, chunk_slot, inference_enabled_event) -> int:
    inference_enabled_event.clear()
    current = _advance_generation(generation)
    chunk_slot.clear(current)
    return current


def _stop_process(process, name: str, timeout_s: float = 15.0) -> None:
    if process is None:
        return
    process.join(timeout=timeout_s)
    if process.is_alive():
        logging.warning("%s worker did not stop in %.1fs; terminating it", name, timeout_s)
        process.terminate()
        process.join(timeout=5.0)
    process.close()


def run_async_record(cfg) -> LeRobotDataset | None:
    """Run latest-observation/latest-chunk inference without changing sync mode."""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.policy is None:
        raise ValueError("Async inference requires a policy")
    if cfg.teleop is not None:
        raise ValueError("Async inference does not support a teleoperator")
    if cfg.display_data:
        init_rerun(session_name="recording")

    compute_stats_enabled = resolve_compute_stats(
        cfg.dataset.save, cfg.dataset.compute_stats
    )
    if cfg.dataset.save == "stream" and compute_stats_enabled:
        logging.info("dataset.save=stream does not compute dataset stats; forcing stats off")

    robot_template = make_robot_from_config(cfg.robot)
    (
        dataset,
        record_features,
        tactile_keys,
        tactile_writer,
        robot_action_processor,
        robot_observation_processor,
        policy_stats,
        processor_stats,
    ) = _prepare_recording(cfg, robot_template)

    action_names = list(record_features[ACTION]["names"])
    action_steps = int(cfg.policy.n_action_steps)
    ctx = mp.get_context("spawn")
    tactile_history_size = 1
    if tactile_keys:
        tactile_frames = int(getattr(cfg.policy, "tactile_num_frames", 1) or 1)
        tactile_offset = int(getattr(cfg.policy, "tactile_frame_offset", 1) or 1)
        tactile_history_size = (tactile_frames - 1) * tactile_offset + 1
    logging.info(
        "[async] observation_history=%d, action_chunk=(%d, %d)",
        tactile_history_size,
        action_steps,
        len(action_names),
    )
    observation_buffer = SharedObservationBuffer.create(
        robot_template.observation_features,
        ctx,
        history_size=max(1, tactile_history_size),
    )
    chunk_slot = SharedActionChunkSlot.create(
        (action_steps, len(action_names)), ctx
    )
    generation = ctx.Value("q", 0)
    inference_enabled_event = ctx.Event()
    shutdown_event = ctx.Event()
    robot_ready_event = ctx.Event()
    inference_ready_event = ctx.Event()
    command_queue = ctx.Queue(maxsize=1)
    response_queue = ctx.Queue(maxsize=1)
    ready_queue = ctx.Queue(maxsize=1)
    error_queue = ctx.Queue()

    robot_process = ctx.Process(
        target=_robot_io_worker,
        name="async-robot-io",
        args=(
            cfg.robot,
            observation_buffer,
            command_queue,
            response_queue,
            ready_queue,
            robot_ready_event,
            error_queue,
            shutdown_event,
            cfg.dataset.fps,
            cfg.reset_before_episode,
        ),
    )
    inference_process = ctx.Process(
        target=_inference_worker,
        name="async-policy-inference",
        args=(
            cfg.policy,
            record_features,
            policy_stats,
            processor_stats,
            cfg.dataset.rename_map,
            robot_template.robot_type,
            tactile_keys,
            cfg.dataset.single_task,
            observation_buffer,
            chunk_slot,
            generation,
            inference_enabled_event,
            inference_ready_event,
            error_queue,
            shutdown_event,
        ),
    )

    listener = None
    robot_client = None
    startup_home_action: dict[str, float] = {}
    try:
        robot_process.start()
        _wait_worker_ready("robot_io", robot_process, robot_ready_event, error_queue)
        startup_home_action = ready_queue.get(timeout=2.0)
        robot_client = RobotIOClient(
            config=cfg.robot,
            robot_type=robot_template.robot_type,
            observation_buffer=observation_buffer,
            command_queue=command_queue,
            response_queue=response_queue,
            error_queue=error_queue,
            shutdown_event=shutdown_event,
        )

        inference_process.start()
        _wait_worker_ready(
            "inference", inference_process, inference_ready_event, error_queue
        )
        listener, events = init_keyboard_listener()

        if cfg.dataset.save == "episode":
            if dataset is None:
                raise RuntimeError("Episode mode requires a dataset")
            with VideoEncodingManager(dataset):
                recorded_episodes = 0
                home_duration = float(getattr(cfg.robot, "home_duration_s", 4.0))
                while (
                    recorded_episodes < cfg.dataset.num_episodes
                    and not events["stop_recording"]
                ):
                    if not wait_for_episode_start(
                        events=events,
                        episode_label=f"Episode {dataset.num_episodes + 1}/{cfg.dataset.num_episodes}",
                        play_sounds=cfg.play_sounds,
                    ):
                        continue
                    if tactile_writer is not None:
                        tactile_writer.start_episode(dataset.num_episodes)

                    episode_generation = _begin_episode(
                        generation, chunk_slot, inference_enabled_event
                    )
                    try:
                        async_record_loop(
                            robot=robot_client,
                            events=events,
                            fps=cfg.dataset.fps,
                            generation=episode_generation,
                            chunk_slot=chunk_slot,
                            action_names=action_names,
                            robot_action_processor=robot_action_processor,
                            robot_observation_processor=robot_observation_processor,
                            error_queue=error_queue,
                            shutdown_event=shutdown_event,
                            dataset=dataset,
                            record_features=record_features,
                            tactile_keys=tactile_keys,
                            tactile_writer=tactile_writer,
                            stream_writer=None,
                            control_time_s=cfg.dataset.episode_time_s,
                            single_task=cfg.dataset.single_task,
                            display_data=cfg.display_data,
                        )
                    finally:
                        _end_episode(generation, chunk_slot, inference_enabled_event)

                    if cfg.reset_before_episode and not events["stop_recording"]:
                        wait_for_inflight_policy_action(cfg.dataset.fps)

                    episode_label = f"Episode {dataset.num_episodes + 1}"
                    if events["rerecord_episode"]:
                        def discard_episode() -> None:
                            events["rerecord_episode"] = False
                            events["exit_early"] = False
                            if tactile_writer is not None:
                                tactile_writer.close_episode(discard=True)
                            dataset.clear_episode_buffer()

                        reset_then_finalize_episode(
                            robot=robot_client,
                            events=events,
                            reset_before_episode=cfg.reset_before_episode,
                            home_action=startup_home_action,
                            fps=cfg.dataset.fps,
                            home_duration_s=home_duration,
                            play_sounds=cfg.play_sounds,
                            episode_label=episode_label,
                            finalize=discard_episode,
                        )
                        continue

                    def save_episode() -> None:
                        if tactile_writer is not None:
                            paths = tactile_writer.close_episode(discard=False)
                            for camera_key, path in paths.items():
                                dataset.register_external_video(
                                    f"{OBS_STR}.images.{camera_key}", path
                                )
                        dataset.save_episode()
                        if tactile_writer is not None:
                            tactile_writer.cleanup_staging()

                    finalized = reset_then_finalize_episode(
                        robot=robot_client,
                        events=events,
                        reset_before_episode=cfg.reset_before_episode,
                        home_action=startup_home_action,
                        fps=cfg.dataset.fps,
                        home_duration_s=home_duration,
                        play_sounds=cfg.play_sounds,
                        episode_label=episode_label,
                        finalize=save_episode,
                    )
                    if finalized:
                        events["exit_early"] = False
                        recorded_episodes += 1
        else:
            stream_root = resolve_dataset_root(cfg.dataset) / "stream"
            with StreamVideoWriter(stream_root=stream_root, fps=cfg.dataset.fps) as stream_writer:
                episode_index = 0
                if cfg.resume and stream_root.exists():
                    existing = sorted(stream_root.glob("episode-*"))
                    if existing:
                        try:
                            episode_index = int(existing[-1].name.split("-", 1)[1]) + 1
                        except (ValueError, IndexError):
                            episode_index = len(existing)
                recorded_episodes = 0
                home_duration = float(getattr(cfg.robot, "home_duration_s", 4.0))
                while (
                    recorded_episodes < cfg.dataset.num_episodes
                    and not events["stop_recording"]
                ):
                    if not wait_for_episode_start(
                        events=events,
                        episode_label=f"Stream episode {episode_index + 1}",
                        play_sounds=cfg.play_sounds,
                    ):
                        continue
                    stream_writer.start_episode(episode_index)
                    if tactile_writer is not None:
                        tactile_writer.start_episode(episode_index)

                    episode_generation = _begin_episode(
                        generation, chunk_slot, inference_enabled_event
                    )
                    try:
                        async_record_loop(
                            robot=robot_client,
                            events=events,
                            fps=cfg.dataset.fps,
                            generation=episode_generation,
                            chunk_slot=chunk_slot,
                            action_names=action_names,
                            robot_action_processor=robot_action_processor,
                            robot_observation_processor=robot_observation_processor,
                            error_queue=error_queue,
                            shutdown_event=shutdown_event,
                            dataset=None,
                            record_features=record_features,
                            tactile_keys=tactile_keys,
                            tactile_writer=tactile_writer,
                            stream_writer=stream_writer,
                            control_time_s=cfg.dataset.episode_time_s,
                            single_task=cfg.dataset.single_task,
                            display_data=cfg.display_data,
                        )
                    finally:
                        _end_episode(generation, chunk_slot, inference_enabled_event)

                    if cfg.reset_before_episode and not events["stop_recording"]:
                        wait_for_inflight_policy_action(cfg.dataset.fps)

                    episode_label = f"Stream episode {episode_index + 1}"
                    if events["rerecord_episode"]:
                        def discard_stream_episode() -> None:
                            events["rerecord_episode"] = False
                            events["exit_early"] = False
                            stream_writer.close_episode(discard=True)
                            if tactile_writer is not None:
                                tactile_writer.close_episode(discard=True)

                        reset_then_finalize_episode(
                            robot=robot_client,
                            events=events,
                            reset_before_episode=cfg.reset_before_episode,
                            home_action=startup_home_action,
                            fps=cfg.dataset.fps,
                            home_duration_s=home_duration,
                            play_sounds=cfg.play_sounds,
                            episode_label=episode_label,
                            finalize=discard_stream_episode,
                        )
                        continue

                    def save_stream_episode() -> None:
                        stream_writer.close_episode(discard=False)
                        if tactile_writer is not None:
                            paths = tactile_writer.close_episode(discard=False)
                            _publish_stream_tactile_videos(
                                resolve_dataset_root(cfg.dataset), episode_index, paths
                            )
                            tactile_writer.cleanup_staging()

                    finalized = reset_then_finalize_episode(
                        robot=robot_client,
                        events=events,
                        reset_before_episode=cfg.reset_before_episode,
                        home_action=startup_home_action,
                        fps=cfg.dataset.fps,
                        home_duration_s=home_duration,
                        play_sounds=cfg.play_sounds,
                        episode_label=episode_label,
                        finalize=save_stream_episode,
                    )
                    if finalized:
                        events["exit_early"] = False
                        recorded_episodes += 1
                        episode_index += 1
    finally:
        inference_enabled_event.clear()
        shutdown_event.set()
        if tactile_writer is not None:
            try:
                tactile_writer.abort_episode()
            except Exception:
                logging.exception("清理异步触觉 writer 失败")
        _stop_process(inference_process if inference_process.pid else None, "inference")
        _stop_process(robot_process if robot_process.pid else None, "robot_io")
        if listener is not None:
            listener.stop()
        for ipc_queue in (command_queue, response_queue, ready_queue, error_queue):
            ipc_queue.close()
        log_say("Stop recording", cfg.play_sounds, blocking=True)

    if cfg.dataset.push_to_hub:
        if cfg.dataset.save == "episode" and dataset is not None:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        else:
            logging.info("Skipping push_to_hub in stream mode")
    log_say("Exiting", cfg.play_sounds)
    return dataset
