"""Asynchronous lossless uint16 tactile recording in FFV1 Matroska files."""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SENTINEL = object()
_FFV1_OPTIONS = {
    "level": "3",
    "coder": "0",
    "context": "0",
    "slicecrc": "1",
    "slices": "4",
}


def _validate_frame(frame: Any, camera_key: str) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint16 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"{camera_key}: expected HWC uint16 tactile frame, "
            f"got shape={array.shape} dtype={array.dtype}"
        )
    return np.ascontiguousarray(array)


class _AsyncFfv1Stream:
    def __init__(self, path: Path, fps: int, frame: np.ndarray, queue_size: int):
        self.path = path
        self.partial_path = path.with_suffix(path.suffix + ".partial")
        self.fps = int(fps)
        self.height, self.width = map(int, frame.shape[:2])
        self.frame_count = 0
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"ffv1-{path.stem}",
        )
        self._thread.start()

    def _run(self) -> None:
        import av

        container = None
        try:
            container = av.open(str(self.partial_path), mode="w", format="matroska")
            container.metadata["tactile_encoding"] = "tactile_u16_fixed_v1"
            container.metadata["channel_order"] = "depth,deform_x,deform_y"
            stream = container.add_stream("ffv1", rate=self.fps, options=_FFV1_OPTIONS)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "gbrp16le"
            stream.time_base = Fraction(1, self.fps)

            while True:
                item = self._queue.get()
                try:
                    if item is _SENTINEL:
                        break
                    video_frame = av.VideoFrame.from_ndarray(item, format="rgb48le")
                    video_frame.pts = self.frame_count
                    video_frame.time_base = Fraction(1, self.fps)
                    for packet in stream.encode(video_frame):
                        container.mux(packet)
                    self.frame_count += 1
                finally:
                    self._queue.task_done()

            for packet in stream.encode():
                container.mux(packet)
        except BaseException as exc:
            self._error = exc
        finally:
            if container is not None:
                try:
                    container.close()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"FFV1 tactile encoder failed for {self.path}") from self._error
        if not self._thread.is_alive() and self.frame_count == 0:
            raise RuntimeError(f"FFV1 tactile encoder exited unexpectedly for {self.path}")

    def add_frame(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"{self.path.stem}: tactile shape changed from "
                f"{(self.height, self.width)} to {frame.shape[:2]}"
            )
        copied = frame.copy(order="C")
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(copied, timeout=0.2)
                return
            except queue.Full:
                continue

    def finish(self, discard: bool = False) -> int:
        while self._thread.is_alive():
            try:
                self._queue.put(_SENTINEL, timeout=0.2)
                break
            except queue.Full:
                self._raise_if_failed()
        self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"FFV1 tactile encoder failed for {self.path}") from self._error
        if discard:
            self.partial_path.unlink(missing_ok=True)
        else:
            if self.frame_count == 0:
                raise RuntimeError(f"Cannot finalize empty tactile stream {self.path}")
            self.partial_path.replace(self.path)
        return self.frame_count


class TactileMkvWriter:
    """Write one lossless uint16 MKV per tactile camera and episode."""

    def __init__(
        self,
        root: Path | str,
        fps: int,
        camera_keys: tuple[str, ...],
        queue_size: int = 8,
        manifest_path: Path | str | None = None,
    ):
        self.root = Path(root)
        self.manifest_path = (
            Path(manifest_path) if manifest_path is not None else self.root / "manifest.json"
        )
        self.fps = int(fps)
        self.camera_keys = tuple(camera_keys)
        self.queue_size = int(queue_size)
        self.root.mkdir(parents=True, exist_ok=True)
        self._episode_dir: Path | None = None
        self._streams: dict[str, _AsyncFfv1Stream] = {}
        self._write_root_manifest()

    def _write_root_manifest(self) -> None:
        manifest = {
            "schema": "starvtla_tactile_mkv_v1",
            "encoding": "tactile_u16_fixed_v1",
            "container": "matroska",
            "codec": "ffv1",
            "pixel_format": "gbrp16le",
            "input_pixel_format": "rgb48le",
            "dtype": "<u2",
            "layout": "HWC",
            "channels": ["depth", "deform_x", "deform_y"],
            "physical_scale": 1000,
            "deform_offset": 30000,
            "fps": self.fps,
            "camera_keys": list(self.camera_keys),
        }
        path = self.manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError(f"Existing tactile manifest is incompatible: {path}")
            return
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def start_episode(self, episode_index: int) -> None:
        if self._episode_dir is not None:
            raise RuntimeError("A tactile episode is already active")
        episode_dir = self.root / f"episode-{int(episode_index):06d}"
        if episode_dir.exists():
            raise FileExistsError(f"Tactile episode already exists: {episode_dir}")
        episode_dir.mkdir(parents=True)
        self._episode_dir = episode_dir
        self._streams = {}

    def add_observation(self, observation: dict[str, Any]) -> None:
        if self._episode_dir is None:
            raise RuntimeError("No active tactile episode")
        for camera_key in self.camera_keys:
            if camera_key not in observation:
                raise KeyError(f"Missing tactile observation '{camera_key}'")
            frame = _validate_frame(observation[camera_key], camera_key)
            stream = self._streams.get(camera_key)
            if stream is None:
                stream = _AsyncFfv1Stream(
                    self._episode_dir / f"{camera_key}.mkv",
                    self.fps,
                    frame,
                    self.queue_size,
                )
                self._streams[camera_key] = stream
            stream.add_frame(frame)

    def close_episode(self, discard: bool = False) -> dict[str, Path]:
        if self._episode_dir is None:
            return {}
        episode_dir = self._episode_dir
        try:
            counts: dict[str, int] = {}
            errors: list[BaseException] = []
            for key, stream in self._streams.items():
                try:
                    counts[key] = stream.finish(discard=discard)
                except BaseException as exc:
                    errors.append(exc)
            if errors:
                raise RuntimeError(
                    f"Failed to finalize {len(errors)} tactile MKV stream(s) in {episode_dir}"
                ) from errors[0]
            if not discard:
                missing = set(self.camera_keys) - set(counts)
                if missing:
                    raise RuntimeError(f"Tactile episode is missing streams: {sorted(missing)}")
                if len(set(counts.values())) != 1:
                    raise RuntimeError(f"Tactile stream frame counts differ: {counts}")
                paths = {key: episode_dir / f"{key}.mkv" for key in counts}
            else:
                paths = {}
        finally:
            self._streams = {}
            self._episode_dir = None
            if discard:
                shutil.rmtree(episode_dir, ignore_errors=True)
        return paths

    def abort_episode(self) -> None:
        self.close_episode(discard=True)
        self.cleanup_staging()

    def cleanup_staging(self) -> None:
        """Remove the staging root after all encoded files have been published."""
        try:
            self.root.rmdir()
        except OSError:
            pass
