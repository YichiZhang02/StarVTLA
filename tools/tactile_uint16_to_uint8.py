#!/usr/bin/env python
"""Convert canonical uint16 tactile frames and datasets to linear uint8 MP4."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

TACTILE_UINT16_ENCODING = "tactile_u16_fixed_v1"
TACTILE_UINT8_ENCODING = "tactile_u8_linear_v1"

# This file is the single source of truth for the model/display quantization range.
DEPTH_MIN = 0
DEPTH_MAX = 1000
DEFORM_CENTER = 30000
DEFORM_MAX = 1000

_H264_RGB_OPTIONS = {
    "crf": "0",
    "preset": "medium",
    "g": "1",
}


def tactile_uint16_to_uint8(frame: NDArray) -> NDArray[np.uint8]:
    """Return the standard linear uint8 tactile view.

    HWC uint16 input is quantized with the fixed project-wide ranges. HWC uint8
    input is already processed and is returned unchanged.
    """
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"Expected HWC 3-channel tactile frame, got shape={array.shape} dtype={array.dtype}"
        )
    if array.dtype == np.uint8:
        return array
    if array.dtype != np.uint16:
        raise ValueError(
            f"Expected uint16 or uint8 tactile frame, got shape={array.shape} dtype={array.dtype}"
        )

    result = np.empty(array.shape, dtype=np.uint8)
    depth = np.clip(array[..., 0].astype(np.int64), DEPTH_MIN, DEPTH_MAX)
    result[..., 0] = np.rint(
        255.0 * (depth - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
    ).astype(np.uint8)

    delta = np.clip(array[..., 1:].astype(np.int64) - DEFORM_CENTER, -DEFORM_MAX, DEFORM_MAX)
    result[..., 1:] = np.where(
        delta < 0,
        128.0 - np.rint(128.0 * np.abs(delta) / DEFORM_MAX),
        128.0 + np.rint(127.0 * delta / DEFORM_MAX),
    ).astype(np.uint8)
    return result


def _video_glob(root: Path, template: str, video_key: str) -> list[Path]:
    pattern = re.sub(r"\{video_key(?::[^}]*)?\}", video_key, template)
    pattern = re.sub(r"\{[^}]+\}", "*", pattern)
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _frame_rate(stream: av.video.stream.VideoStream) -> Fraction:
    rate = stream.average_rate or stream.base_rate
    if rate is None:
        raise ValueError("Tactile video has no frame rate")
    return Fraction(rate.numerator, rate.denominator)


def _convert_video(path_str: str, output_str: str) -> tuple[str, str, int]:
    path = Path(path_str)
    output = Path(output_str)
    partial = output.with_name(f".{output.stem}.uint8.partial.mp4")
    frame_count = 0
    try:
        with av.open(str(path), mode="r") as input_container:
            input_stream = input_container.streams.video[0]
            rate = _frame_rate(input_stream)
            width = input_stream.codec_context.width
            height = input_stream.codec_context.height

            with av.open(str(partial), mode="w", format="mp4") as output_container:
                output_container.metadata["tactile_encoding"] = TACTILE_UINT8_ENCODING
                output_container.metadata["channel_order"] = "depth,deform_x,deform_y"
                output_stream = output_container.add_stream(
                    "libx264rgb", rate=rate, options=_H264_RGB_OPTIONS
                )
                output_stream.width = width
                output_stream.height = height
                output_stream.pix_fmt = "bgr24"

                time_base = Fraction(rate.denominator, rate.numerator)
                for frame_count, input_frame in enumerate(input_container.decode(input_stream), start=1):
                    packed = input_frame.to_ndarray(format="rgb48le")
                    encoded = tactile_uint16_to_uint8(packed)
                    output_frame = av.VideoFrame.from_ndarray(encoded, format="rgb24")
                    output_frame.pts = frame_count - 1
                    output_frame.time_base = time_base
                    for packet in output_stream.encode(output_frame):
                        output_container.mux(packet)

                for packet in output_stream.encode():
                    output_container.mux(packet)

        if frame_count == 0:
            raise ValueError(f"Tactile video has no decodable frames: {path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output)
        return str(path), str(output), frame_count
    finally:
        partial.unlink(missing_ok=True)


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = set()
    if "frames_cache" in names:
        ignored.add("frames_cache")
    if "contact_std.npz" in names:
        ignored.add("contact_std.npz")
    return ignored


def _write_uint8_manifest(root: Path, info: dict, tactile_keys: list[str]) -> None:
    raw_camera_keys = [key.removeprefix("observation.images.") for key in tactile_keys]
    manifest = {
        "schema": "starvtla_tactile_mkv_v1",
        "encoding": TACTILE_UINT8_ENCODING,
        "source_encoding": TACTILE_UINT16_ENCODING,
        "container": "mp4",
        "codec": "h264",
        "encoder": "libx264rgb",
        "lossless": True,
        "pixel_format": "gbrp",
        "input_pixel_format": "rgb24",
        "dtype": "|u1",
        "layout": "HWC",
        "channels": ["depth", "deform_x", "deform_y"],
        "depth_min": DEPTH_MIN,
        "depth_max": DEPTH_MAX,
        "deform_center": DEFORM_CENTER,
        "deform_max": DEFORM_MAX,
        "fps": info["fps"],
        "camera_keys": raw_camera_keys,
        "authoritative": False,
    }
    manifest_path = root / "meta" / "tactile_encoding.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _patch_info(root: Path, info: dict, tactile_keys: list[str]) -> None:
    for key in tactile_keys:
        feature = info["features"][key]
        feature["tactile_encoding"] = TACTILE_UINT8_ENCODING
        feature["storage_dtype"] = "uint8"
        template = feature.get("video_path", info.get("video_path"))
        if template:
            feature["video_path"] = str(Path(template).with_suffix(".mp4"))
        video_info = feature.setdefault("info", {})
        video_info["video.codec"] = "h264"
        video_info["video.pix_fmt"] = "gbrp"
        video_info["video.channels"] = 3
    (root / "meta" / "info.json").write_text(
        json.dumps(info, indent=4) + "\n", encoding="utf-8"
    )
    _write_uint8_manifest(root, info, tactile_keys)


def _uint16_tactile_keys(info: dict) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("tactile_encoding") == TACTILE_UINT16_ENCODING
    ]


def _conversion_tasks(
    root: Path, info: dict, tactile_keys: list[str]
) -> list[tuple[str, Path, Path]]:
    tasks: list[tuple[str, Path, Path]] = []
    for key in tactile_keys:
        feature = info["features"][key]
        template = feature.get("video_path", info.get("video_path"))
        if not template:
            raise ValueError(f"No video path template for tactile feature {key}")
        paths = _video_glob(root, template, key)
        if not paths:
            raise FileNotFoundError(f"No tactile videos found for {key} using {template!r}")
        for path in paths:
            output = path.with_suffix(".mp4")
            if output != path and output.exists():
                raise FileExistsError(output)
            tasks.append((key, path, output))
    return tasks


def _run_conversion_tasks(
    tasks: list[tuple[str, Path, Path]], jobs: int
) -> dict[str, int]:
    counts: dict[str, int] = {}
    generated: list[Path] = []
    try:
        if int(jobs) <= 1:
            converted = (
                (key, *_convert_video(str(path), str(output)))
                for key, path, output in tasks
            )
            for key, source, output, frames in converted:
                generated.append(Path(output))
                counts[key] = counts.get(key, 0) + frames
                print(f"[tactile uint8] {source} -> {output}: {frames} frames", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=int(jobs)) as executor:
                future_to_key = {
                    executor.submit(_convert_video, str(path), str(output)): key
                    for key, path, output in tasks
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    source, output, frames = future.result()
                    generated.append(Path(output))
                    counts[key] = counts.get(key, 0) + frames
                    print(f"[tactile uint8] {source} -> {output}: {frames} frames", flush=True)
    except BaseException:
        for output in generated:
            output.unlink(missing_ok=True)
        raise

    for _key, source, output in tasks:
        if source != output:
            source.unlink()
    return counts


def _drop_derived_caches(root: Path) -> None:
    shutil.rmtree(root / "frames_cache", ignore_errors=True)
    (root / "meta" / "contact_std.npz").unlink(missing_ok=True)


def _convert_root(root: Path, info: dict, jobs: int) -> dict[str, int]:
    tactile_keys = _uint16_tactile_keys(info)
    if not tactile_keys:
        return {}
    tasks = _conversion_tasks(root, info, tactile_keys)
    counts = _run_conversion_tasks(tasks, jobs)
    _patch_info(root, info, tactile_keys)
    _drop_derived_caches(root)
    return counts


def convert_tactile_dataset(src: Path | str, dst: Path | str, jobs: int = 4) -> dict[str, int]:
    """Create a dataset copy whose canonical uint16 tactile videos are linear uint8 MP4."""
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)
    if src == dst:
        raise ValueError("Source and destination datasets must be different")
    if dst.exists():
        raise FileExistsError(dst)

    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dst.name}.", dir=dst.parent) as temp_dir:
        work_root = Path(temp_dir) / dst.name
        shutil.copytree(src, work_root, ignore=_copy_ignore)
        counts = _convert_root(work_root, info, jobs)
        work_root.replace(dst)
    return counts


def convert_tactile_dataset_in_place(root: Path | str, jobs: int = 4) -> dict[str, int]:
    """Replace canonical uint16 tactile MKV files with linear uint8 lossless-RGB MP4 files."""
    root = Path(root).resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return _convert_root(root, info, jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert tactile videos to linear uint8 RGB MP4.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--root", type=Path, help="Dataset to convert in place")
    mode.add_argument("--src", type=Path, help="Source dataset for copy mode")
    parser.add_argument("--dst", type=Path, help="Destination dataset for copy mode")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel tactile video workers")
    args = parser.parse_args()

    if args.root is not None:
        if args.dst is not None:
            parser.error("--dst cannot be used with --root")
        counts = convert_tactile_dataset_in_place(args.root, jobs=args.jobs)
        destination = args.root
    else:
        if args.dst is None:
            parser.error("--dst is required with --src")
        counts = convert_tactile_dataset(args.src, args.dst, jobs=args.jobs)
        destination = args.dst
    if counts:
        print(f"Converted tactile streams: {counts}")
    else:
        print("No uint16 tactile streams found; nothing changed.")
    print(f"Saved uint8 dataset: {destination}")


if __name__ == "__main__":
    main()
