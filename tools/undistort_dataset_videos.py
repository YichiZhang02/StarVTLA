#!/usr/bin/env python
"""Create a fisheye-undistorted copy of a LeRobot (v3.0) dataset (wrist cameras only).

The UMI wrist cameras are recorded as full-frame fisheye (1920x1080, Kalibr equidistant /
OpenCV fisheye model). For training we want the rectilinear center crop the policies consume
(896x896) — i.e. the `..._umistyle` -> `..._umistyle_undist` transform. Doing it offline keeps
decode cheap and the geometry identical across runs.

Pipeline per wrist frame (matches ugripper/zxd_fisheye/undistort_wrist.py):
    1. cv2.fisheye undistort with new camera matrix = K  (undistort in place, no extra zoom)
    2. center-crop a CROP x CROP square out of the undistorted frame  (no resize)

What it does (non-destructive — writes a new dataset copy):
  - copies meta/ and data/ verbatim (parquet, stats, episodes, tasks, etc.)
  - undistorts + crops only the wrist cameras, preserving the exact frame count / fps /
    timestamps, with a dense keyframe interval (small GOP) so random-access seeks stay cheap
  - COPIES every other camera video verbatim — in particular the tactile finger cams, stored
    lossless 16-bit (ffv1 / gbrp16le, .mkv); re-encoding those would corrupt them
  - patches meta/info.json so each undistorted feature's shape / height / width / codec match

Per-channel image stats in meta/stats.json are kept as-is: an undistort + center crop is a
geometric warp that preserves per-channel mean/std to well within training tolerance (the same
rationale by which downscale_dataset_videos.py keeps stats across a resize).

Speed: the work per wrist frame is decode -> colour-convert -> fisheye remap -> re-encode, all
CPU-bound. `--gpu-decode auto` (default) offloads the H.264/HEVC/AV1 *decode* to NVDEC (cuvid)
and downloads NV12 (half the bytes of BGR), converting to BGR with cv2 — this frees CPU cores
under parallel `--jobs`. Encode stays on libx264 and remap stays on cv2: the A100 has no NVENC
unit, and remap needs a CUDA-enabled OpenCV build we don't assume. NVDEC decode is only used when
the source resolution equals the calibration resolution (no resize needed); other files fall back
to CPU automatically. The NV12->BGR path differs from the CPU swscale path by ~1 grey level (well
under codec noise); pass `--gpu-decode off` for byte-for-byte parity with older CPU-only runs.

Calibration JSON (Kalibr/OpenCV fisheye, e.g. tools/calib/x5_left_intrinsics.json):
    {
        "distortion_model": "equidistant",
        "camera_matrix":     [[fx,0,cx],[0,fy,cy],[0,0,1]],
        "distortion_coeffs": [k1,k2,k3,k4],
        "resolution":        [width, height]
    }
Defaults to the bundled tools/calib/x5_{left,right}_intrinsics.json for the two wrist keys.

Usage:
    # quick visual test: one frame per wrist cam -> comparison PNGs + a short clip
    python tools/undistort_dataset_videos.py --src playground/data/rm_umi_dual_pen_open --test

    # full dataset
    python tools/undistort_dataset_videos.py \
        --src playground/data/rm_umi_dual_pen_open \
        --dst playground/data/rm_umi_dual_pen_open_undist

Then train against --dataset.root=<dst> (same repo_id works).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

# ffmpeg encoder name -> the codec tag stored in info.json (matches LeRobot's convention).
_CODEC_TAG = {
    "libx264": "h264",
    "libx264rgb": "h264",
    "libx265": "hevc",
    "libsvtav1": "av1",
    "h264_nvenc": "h264",
    "hevc_nvenc": "hevc",
}

# Source codec (as ffprobe reports it) -> NVDEC (cuvid) decoder for GPU-offloaded decode.
# Only decode is GPU-offloaded: the A100 has no NVENC unit, and the fisheye remap needs a
# CUDA-enabled OpenCV build (not assumed), so color-convert + remap + encode stay on CPU.
_CUVID = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "av1": "av1_cuvid",
    "vp9": "vp9_cuvid",
    "vp8": "vp8_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "mpeg2video": "mpeg2_cuvid",
}

# Wrist cameras undistorted by default -> bundled intrinsics. Everything else is copied verbatim.
DEFAULT_CALIB = {
    "observation.images.left_cam_wrist": HERE / "calib" / "x5_left_intrinsics.json",
    "observation.images.right_cam_wrist": HERE / "calib" / "x5_right_intrinsics.json",
}


def _load_calibration(path: str) -> tuple[list, list, tuple[int, int]]:
    """Return (camera_matrix, distortion_coeffs, (w, h)) as plain lists/ints (picklable)."""
    d = json.loads(Path(path).read_text())
    model = d.get("distortion_model")
    if model != "equidistant":
        raise SystemExit(f"{path}: expected equidistant (fisheye) model, got {model!r}")
    K = d["camera_matrix"]
    D = d["distortion_coeffs"]
    w, h = (int(x) for x in d["resolution"])
    return K, D, (w, h)


def _build_maps(K_list, D_list, in_size):
    """Undistort maps; new camera matrix = K (in-place undistort, no extra zoom)."""
    K = np.asarray(K_list, dtype=np.float64)
    D = np.asarray(D_list, dtype=np.float64).reshape((4, 1))
    return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, in_size, cv2.CV_16SC2)


def _center_crop(img: np.ndarray, crop: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = (w - crop) // 2
    y0 = (h - crop) // 2
    return img[y0:y0 + crop, x0:x0 + crop]


def _read_first_frame(path: str, in_size: tuple[int, int]) -> np.ndarray:
    """Decode the first frame as BGR at in_size via ffmpeg (handles av1/hevc software)."""
    in_w, in_h = in_size
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-frames:v", "1",
         "-vf", f"scale={in_w}:{in_h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(in_h, in_w, 3).copy()


def _probe_stream(path: str) -> tuple[float, str, int, int]:
    """Return (fps, codec_name, width, height) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,codec_name,width,height",
         "-of", "default=nokey=1:noprint_wrappers=1", path],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    # ffprobe prints the entries in the order requested: codec_name width height r_frame_rate.
    codec, w, h, rate = (out + ["", "0", "0", "30"])[:4]
    if "/" in rate:
        n, d = rate.split("/")
        fps = float(n) / float(d) if float(d) else 30.0
    else:
        fps = float(rate or 30.0)
    return fps, codec, int(w or 0), int(h or 0)


def _nvdec_functional(sample_src: str, dec_name: str) -> bool:
    """One-shot check that NVDEC can actually decode this codec on this box (A100 has no NVENC
    but does have NVDEC; a build listing the decoder does not guarantee a capable device)."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
             "-c:v", dec_name, "-i", sample_src, "-map", "0:v:0", "-frames:v", "1",
             "-f", "null", "-"],
            capture_output=True,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _decode_remap_encode(src, dst, map1, map2, in_w, in_h, crop, fps, gop, crf,
                         codec, scale_flags, src_codec, use_gpu):
    """Run one decode -> remap -> encode pipeline. Returns (frames_written, dec_rc, enc_rc).

    Split out of _undistort_encode so a failed GPU decode can be retried on CPU with the same
    remap maps (see the per-file fallback in _undistort_encode)."""
    if use_gpu:
        dec = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
             "-c:v", _CUVID[src_codec], "-i", src, "-map", "0:v:0",
             "-vf", "hwdownload,format=nv12", "-f", "rawvideo", "-pix_fmt", "nv12", "-"],
            stdout=subprocess.PIPE,
        )
        frame_bytes = in_w * in_h * 3 // 2  # NV12 = 1.5 bytes/pixel
    else:
        dec = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", src, "-map", "0:v:0",
             "-vf", f"scale={in_w}:{in_h}:flags={scale_flags}",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE,
        )
        frame_bytes = in_w * in_h * 3
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{crop}x{crop}", "-r", f"{fps}",
         "-i", "-", "-c:v", codec, "-crf", str(crf), "-g", str(gop),
         "-pix_fmt", "yuv420p", "-an", "-vsync", "0", dst],
        stdin=subprocess.PIPE,
    )
    frames = 0
    while True:
        raw = dec.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            break
        if use_gpu:
            yuv = np.frombuffer(raw, np.uint8).reshape(in_h * 3 // 2, in_w)
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        else:
            frame = np.frombuffer(raw, np.uint8).reshape(in_h, in_w, 3)
        out = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        enc.stdin.write(np.ascontiguousarray(out).tobytes())
        frames += 1
    dec.stdout.close()
    dec.wait()
    enc.stdin.close()
    enc.wait()
    return frames, dec.returncode, enc.returncode


def _undistort_encode(args: tuple) -> tuple[str, bool, str]:
    """Undistort + center-crop one video into dst. Returns (src_path, ok, message).

    gpu_decode offloads the H.264/HEVC/AV1 decode to NVDEC (cuvid) and downloads frames as NV12
    (half the bytes of BGR); the colour-convert (cv2), fisheye remap (cv2) and encode (libx264)
    stay on CPU. gpu_codecs is the set of source codecs NVDEC was verified to handle on this box;
    only files in that set (and at the calibration resolution) take the GPU path, everything else
    decodes on CPU. If a GPU decode still produces zero frames it is retried on CPU rather than
    silently written as an empty stub.
    """
    (src, dst, K_list, D_list, in_size, crop, gop, crf, codec, scale_flags, gpu_codecs) = args
    in_w, in_h = in_size
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    try:
        map1, map2 = _build_maps(K_list, D_list, in_size)
        # Crop the maps to the output window so remap produces the crop x crop frame directly
        # (no full-frame warp of pixels we discard). Pixel-identical to remap-then-center-crop.
        x0, y0 = (in_w - crop) // 2, (in_h - crop) // 2
        map1 = np.ascontiguousarray(map1[y0:y0 + crop, x0:x0 + crop])
        map2 = np.ascontiguousarray(map2[y0:y0 + crop, x0:x0 + crop])
        fps, src_codec, src_w, src_h = _probe_stream(src)
        # NVDEC path only when no resize is needed (maps are computed at the calibration
        # resolution) and this codec was verified NVDEC-decodable on this box.
        use_gpu = (src_codec in gpu_codecs and (src_w, src_h) == (in_w, in_h)
                   and src_codec in _CUVID)
        frames, dec_rc, enc_rc = _decode_remap_encode(
            src, dst, map1, map2, in_w, in_h, crop, fps, gop, crf,
            codec, scale_flags, src_codec, use_gpu)
        # Defensive: a GPU decode that still yielded nothing means NVDEC choked on this file
        # despite the codec passing the probe. Retry once on CPU before giving up so we never
        # emit a zero-frame stub.
        if frames == 0 and use_gpu:
            frames, dec_rc, enc_rc = _decode_remap_encode(
                src, dst, map1, map2, in_w, in_h, crop, fps, gop, crf,
                codec, scale_flags, src_codec, use_gpu=False)
        if enc_rc != 0:
            return src, False, "ffmpeg encode returned non-zero"
        if frames == 0:
            return src, False, "produced 0 frames (decode failed)"
    except Exception as e:  # noqa: BLE001
        return src, False, str(e)
    return src, True, ""


def _copy_verbatim(args: tuple[str, str]) -> tuple[str, bool, str]:
    """Copy one video file unchanged. Returns (src_path, ok, message)."""
    src, dst = args
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as e:  # noqa: BLE001
        return src, False, str(e)
    return src, True, ""


def _frame_count(path: Path) -> int | None:
    """Video frame count via ffprobe. Uses -count_packets (demux only, no decode): for these CFR
    videos the packet count equals the frame count, and it is ~500x faster than -count_frames,
    which decodes every frame single-threaded (minutes per HEVC file). Returns None on failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return int(out)
    except Exception:  # noqa: BLE001
        return None


def copy_non_video(src: Path, dst: Path) -> None:
    """Copy everything except videos/ and frames_cache/ verbatim."""
    skip = {"videos", "frames_cache"}
    dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        if entry.name in skip:
            continue
        target = dst / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)


def patch_info_json(dst: Path, targets: list[str], crop: int, codec: str,
                    calib_map: dict[str, str]) -> int:
    """Update undistorted features' shape/codec in meta/info.json + write an "undistort" marker.

    The marker lets inference auto-detect that this dataset is undistorted (so the live wrist
    camera must be undistorted too); see deployment/inference.py _resolve_undistort.
    """
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    tag = _CODEC_TAG.get(codec, codec)
    n = 0
    for key, ft in info.get("features", {}).items():
        if key not in targets or ft.get("dtype") != "video":
            continue
        ch = ft["shape"][2] if len(ft.get("shape", [])) == 3 else 3
        ft["shape"] = [crop, crop, ch]
        vinfo = ft.setdefault("info", {})
        vinfo["video.height"] = crop
        vinfo["video.width"] = crop
        vinfo["video.codec"] = tag
        n += 1
    info["undistort"] = {
        "model": "equidistant",
        "crop": int(crop),
        "cameras": {cam: Path(calib_map[cam]).name for cam in targets},
    }
    info_path.write_text(json.dumps(info, indent=4))
    return n


def _resolve_calib(values: list[str] | None, cameras: list[str]) -> dict[str, str]:
    """--calib: unset -> bundled defaults; single PATH -> all cams; repeated cam=PATH."""
    if not values:
        return {c: str(DEFAULT_CALIB[c]) for c in cameras if c in DEFAULT_CALIB}
    if len(values) == 1 and "=" not in values[0]:
        return {c: values[0] for c in cameras}
    out: dict[str, str] = {}
    for v in values:
        cam, _, path = v.partition("=")
        out[cam] = path
    return out


def run_test(src: Path, calib_map: dict[str, str], crop: int) -> int:
    """One frame per wrist cam -> original / undistorted-with-cropbox / final PNGs + short clip."""
    out_dir = HERE / "undistort_test"
    out_dir.mkdir(exist_ok=True)
    for cam, jpath in calib_map.items():
        K, D, (W, H) = _load_calibration(jpath)
        map1, map2 = _build_maps(K, D, (W, H))
        vids = sorted((src / "videos" / cam).rglob("*.mp4"))
        if not vids:
            print(f"  [{cam}] no .mp4 found, skip")
            continue
        try:
            frame = _read_first_frame(str(vids[0]), (W, H))
        except Exception as e:  # noqa: BLE001
            print(f"  [{cam}] cannot read {vids[0]}: {e}, skip")
            continue
        und = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        x0, y0 = (W - crop) // 2, (H - crop) // 2
        box = und.copy()
        cv2.rectangle(box, (x0, y0), (x0 + crop, y0 + crop), (0, 0, 255), 4)
        tag = cam.split(".")[-1]
        cv2.imwrite(str(out_dir / f"{tag}_0_original.png"), frame)
        cv2.imwrite(str(out_dir / f"{tag}_1_undistorted_cropbox.png"), box)
        cv2.imwrite(str(out_dir / f"{tag}_2_final_{crop}x{crop}.png"), _center_crop(und, crop))
        clip = out_dir / f"{tag}_clip_{crop}x{crop}.mp4"
        _undistort_encode((str(vids[0]), str(clip), K, D, (W, H), crop, 4, 18, "libx264", "lanczos", False))
        print(f"  [{cam}] {vids[0].name} -> {tag}_*.png + clip")
    print(f"\nTest outputs in {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Source dataset root (unchanged).")
    ap.add_argument("--dst", type=Path, default=None, help="Destination dataset root (created).")
    ap.add_argument("--calib", nargs="+", default=None,
                    help="unset = bundled tools/calib; or single calib.json; or repeated cam=calib.json.")
    ap.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help=(
            "Wrist camera keys to undistort (others copied verbatim). "
            "Unset = use whichever bundled-default wrist cameras exist in the dataset."
        ),
    )
    ap.add_argument("--crop", type=int, default=896, help="Center-crop square size (default 896).")
    ap.add_argument("--gop", type=int, default=4, help="Keyframe interval; small = fast seeks (default 4).")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality, lower = better/larger (default 18).")
    ap.add_argument("--codec", default="libx264", help="ffmpeg video encoder (default libx264).")
    ap.add_argument("--scale-flags", default="lanczos", help="swscale kernel for input scaling (default lanczos).")
    ap.add_argument("--gpu-decode", choices=["auto", "on", "off"], default="auto",
                    help="Offload video decode to NVDEC/cuvid ('auto' = use it when a capable GPU "
                         "and NVDEC decoder exist; encode/remap always stay on CPU).")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel ffmpeg/copy workers (default 8).")
    ap.add_argument("--overwrite", action="store_true", help="Redo files that already exist in dst.")
    ap.add_argument("--verify", action="store_true",
                    help="ffprobe-check that each undistorted frame count matches the source "
                         "(fast: demux-only packet count).")
    ap.add_argument("--test", action="store_true", help="Quick visual test only (no dataset written).")
    args = ap.parse_args()

    src = args.src.resolve()
    if not (src / "meta" / "info.json").is_file():
        ap.error(f"{src} does not look like a LeRobot dataset (missing meta/info.json).")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        ap.error("ffmpeg/ffprobe not found on PATH.")

    info = json.loads((src / "meta" / "info.json").read_text())
    requested_cameras = args.cameras if args.cameras is not None else list(DEFAULT_CALIB)
    targets = [
        camera
        for camera in requested_cameras
        if info.get("features", {}).get(camera, {}).get("dtype") == "video"
    ]
    missing = [camera for camera in requested_cameras if camera not in targets]
    if args.cameras is not None and missing:
        ap.error(f"--cameras not present as video features in dataset: {missing}")
    calib_map = _resolve_calib(args.calib, targets)
    for cam in targets:
        if cam not in calib_map:
            ap.error(f"no --calib provided for {cam}")
        _load_calibration(calib_map[cam])  # validate early

    if args.test:
        return run_test(src, {c: calib_map[c] for c in targets}, args.crop)

    if args.dst is None:
        ap.error("--dst is required (or pass --test for a visual check).")
    dst = args.dst.resolve()
    if src == dst:
        ap.error("--src and --dst must differ.")
    video_root = src / "videos"
    if not video_root.is_dir():
        ap.error(f"No videos/ directory under {src}.")

    print(f"Source: {src}\nDest:   {dst}")
    print(f"Undistort targets ({len(targets)}): {sorted(targets)}")
    print(f"  -> center-crop {args.crop}x{args.crop}, codec={args.codec}, gop={args.gop}, crf={args.crf}")
    print("Copying meta/ and data/ verbatim (skipping videos/, frames_cache/) ...")
    copy_non_video(src, dst)
    n_feats = patch_info_json(dst, targets, args.crop, args.codec, calib_map)
    print(f"Patched {n_feats} video feature(s) in meta/info.json.")

    # Plan per-file work: undistort target wrist cams, copy everything else (incl. tactile) verbatim.
    enc_jobs: list[tuple] = []
    copy_jobs: list[tuple] = []
    calib_cache = {cam: _load_calibration(calib_map[cam]) for cam in targets}
    for vid in sorted(video_root.rglob("*")):
        if not vid.is_file():
            continue
        cam = vid.relative_to(video_root).parts[0]
        if cam in targets:
            out = (dst / vid.relative_to(src)).with_suffix(".mp4")
            if out.exists() and not args.overwrite:
                continue
            K, D, in_size = calib_cache[cam]
            enc_jobs.append([str(vid), str(out), K, D, in_size, args.crop,
                             args.gop, args.crf, args.codec, args.scale_flags])
        else:
            out = dst / vid.relative_to(src)  # preserve extension (e.g. tactile .mkv)
            if out.exists() and not args.overwrite:
                continue
            copy_jobs.append((str(vid), str(out)))

    # Resolve GPU-decode capability once per distinct source codec (a mixed-codec dataset can hold
    # e.g. mostly hevc plus a few av1; probing only the first file would wrongly enable NVDEC for
    # a codec this box can't hardware-decode, silently producing empty videos). Only codecs that
    # pass the probe take the GPU path; the rest decode on CPU. Encode + remap always run on CPU.
    gpu_codecs: set[str] = set()
    if args.gpu_decode != "off" and enc_jobs:
        codecs = {}
        for j in enc_jobs:
            _, c, _, _ = _probe_stream(j[0])
            codecs.setdefault(c, j[0])  # first sample file seen for this codec
        bad_codecs = []
        for c, sample in codecs.items():
            dec_name = _CUVID.get(c)
            if dec_name and _nvdec_functional(sample, dec_name):
                gpu_codecs.add(c)
            else:
                bad_codecs.append(c)
        if gpu_codecs:
            print(f"GPU decode: on for codec(s) {sorted(gpu_codecs)} (NVDEC; encode/remap on CPU).")
        if bad_codecs:
            if args.gpu_decode == "on":
                ap.error(f"--gpu-decode=on but NVDEC unavailable for codec(s) {sorted(bad_codecs)}.")
            print(f"GPU decode: CPU fallback for codec(s) {sorted(bad_codecs)} "
                  f"(no working NVDEC decoder).")
    for j in enc_jobs:
        j.append(gpu_codecs)

    print(f"Undistorting {len(enc_jobs)} wrist video(s); copying {len(copy_jobs)} other video(s) "
          f"verbatim with {args.jobs} worker(s) ...")
    failures: list[tuple[str, str]] = []
    total = len(enc_jobs) + len(copy_jobs)
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(_undistort_encode, j): ("und", j) for j in enc_jobs}
        futs.update({pool.submit(_copy_verbatim, (j[0], j[1])): ("cp", j) for j in copy_jobs})
        for fut in as_completed(futs):
            srcf, ok, msg = fut.result()
            done += 1
            kind = futs[fut][0]
            rel = Path(srcf).relative_to(video_root)
            if ok:
                print(f"  [{done}/{total}] {kind:3s} ok  {rel}")
            else:
                failures.append((srcf, msg))
                print(f"  [{done}/{total}] {kind:3s} FAIL {rel}: {msg}")

    if args.verify and not failures:
        print("Verifying undistorted frame counts (ffprobe) ...")
        for j in enc_jobs:
            cs, cd = _frame_count(Path(j[0])), _frame_count(Path(j[1]))
            if cs is not None and cs != cd:
                failures.append((j[0], f"frame count {cd} != source {cs}"))
                print(f"  MISMATCH {Path(j[1]).name}: {cd} != {cs}")

    if failures:
        print(f"\nDONE WITH {len(failures)} FAILURE(S):", file=sys.stderr)
        for f, m in failures:
            print(f"  {f}: {m}", file=sys.stderr)
        return 1

    print(f"Done. Train with --dataset.root={dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
