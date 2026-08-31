# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Open-loop evaluation: load one episode (HDF5 + cam_front/cam_high mp4),
# run policy at each chunk start with demo observations, compare predicted
# actions to ground truth and plot.  Supports optional tactile videos.
#
# Usage (same env as franka_server):
# cd /path/to/cosmos-policy

# export FRANKA_COSMOS_CONFIG=cosmos_predict2_2b_480p_franka_shave_cucumber_20260321
# export FRANKA_COSMOS_CKPT=/path/to/checkpoints/iter_000003000
# export FRANKA_DATASET_STATS_PATH=/path/to/dataset_statistics_franka.json
# export FRANKA_T5_EMBEDDINGS_PATH=/path/to/t5_embeddings.pkl

# Future pred + mask-IoU (saves pred/gt PNGs and future_pred_iou_summary.json):
#   --future_pred_eval
#
# uv run --extra cu128 --group libero --python 3.10 python -m cosmos_policy.experiments.robot.franka.run_franka_openloop \
#   --hdf5 /path/to/train/episode_0.hdf5 \
#   --cam_front /path/to/train/episode_0_cam_front.mp4 \
#   --cam_high /path/to/train/episode_0_cam_high.mp4 \
#   --tactile_left /path/to/train/episode_0_tactile_rectify_left.mp4 \
#   --tactile_right /path/to/train/episode_0_tactile_rectify_right.mp4 \
#   --out_dir ./openloop_out

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cosmos_policy.datasets.aloha_dataset import load_video_as_images
from cosmos_policy.datasets.dataset_utils import decode_single_jpeg_frame, resize_images
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)

DATA_ROOT = "/path/to/tactile_data_hupai/hupai"

CONFIG_NAME = os.environ.get("FRANKA_COSMOS_CONFIG", "cosmos_predict2_2b_480p_franka_hupai_tactile")
CKPT_PATH = os.environ.get("FRANKA_COSMOS_CKPT", "")
CONFIG_FILE = os.environ.get("FRANKA_COSMOS_CONFIG_FILE", "cosmos_policy/config/config.py")
DATASET_STATS_PATH = os.environ.get("FRANKA_DATASET_STATS_PATH", f"{DATA_ROOT}/dataset_statistics_franka.json")
T5_EMBEDDINGS_PATH = os.environ.get("FRANKA_T5_EMBEDDINGS_PATH", f"{DATA_ROOT}/t5_embeddings.pkl")
CHUNK_SIZE = int(os.environ.get("FRANKA_CHUNK_SIZE", "20"))
NUM_DENOISING_STEPS = int(os.environ.get("FRANKA_NUM_DENOISING_STEPS", "100"))
COSMOS_IMAGE_SIZE = 224


def _to_uint8_hw3(img: Any) -> np.ndarray:
    """(H,W,3) uint8 from model output (may be (1,H,W,3) or torch)."""
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    a = np.asarray(img)
    if a.ndim == 4:
        if a.shape[0] == 1:
            a = a[0]
        else:
            a = a[0]
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(np.round(a), 0, 255).astype(np.uint8)
    return a


def _rgb_to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    y = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    return y


def _otsu_threshold_from_hist(gray_flat: np.ndarray) -> int:
    """Otsu threshold in [0,255] from 1D uint8 samples (numpy only)."""
    hist = np.bincount(gray_flat.astype(np.int64).ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 127
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    # Between-class variance; avoid div by zero
    denom = omega * (1.0 - omega) + 1e-10
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    return int(np.argmax(sigma_b2))


def _laplacian_abs(gray: np.ndarray) -> np.ndarray:
    """|∇²I| on interior pixels; float32, same shape as gray."""
    g = gray.astype(np.float32)
    out = np.zeros_like(g, dtype=np.float32)
    out[1:-1, 1:-1] = np.abs(
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    )
    return out


def edge_texture_iou(
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    percentile: float = 88.0,
) -> float:
    """
    IoU of high-|Laplacian| regions (texture / dot boundaries), weakly sensitive to global color.
    Joint percentile on pred+GT Laplacian magnitudes defines one threshold for both masks.
    """
    gp = _rgb_to_gray_u8(pred_rgb).astype(np.float32)
    gg = _rgb_to_gray_u8(gt_rgb).astype(np.float32)
    if gp.shape != gg.shape:
        raise ValueError(f"Shape mismatch pred {gp.shape} vs gt {gg.shape}")
    lp = _laplacian_abs(gp)
    lg = _laplacian_abs(gg)
    flat = np.concatenate([lp.ravel(), lg.ravel()])
    t = float(np.percentile(flat, percentile))
    mp = lp >= t
    mg = lg >= t
    inter = np.logical_and(mp, mg).sum(dtype=np.int64)
    union = np.logical_or(mp, mg).sum(dtype=np.int64)
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def dark_region_inv_otsu_iou(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    """
    IoU of "dark" regions after value inversion (255 - gray), joint Otsu on inv.
    GelSight black dots become high in inv; helps compare dot-field without matching hue.
    """
    gp = _rgb_to_gray_u8(pred_rgb)
    gg = _rgb_to_gray_u8(gt_rgb)
    if gp.shape != gg.shape:
        raise ValueError(f"Shape mismatch pred {gp.shape} vs gt {gg.shape}")
    inv_p = (255 - gp.astype(np.int16)).clip(0, 255).astype(np.uint8)
    inv_g = (255 - gg.astype(np.int16)).clip(0, 255).astype(np.uint8)
    t = _otsu_threshold_from_hist(np.concatenate([inv_p.ravel(), inv_g.ravel()]))
    mp = inv_p >= t
    mg = inv_g >= t
    inter = np.logical_and(mp, mg).sum(dtype=np.int64)
    union = np.logical_or(mp, mg).sum(dtype=np.int64)
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def foreground_mask_iou(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    """
    IoU of binary foreground masks from a joint Otsu threshold on grayscale.
    Both pred and gt contribute to the histogram so one shared threshold is used.
    """
    gp = _rgb_to_gray_u8(pred_rgb)
    gg = _rgb_to_gray_u8(gt_rgb)
    if gp.shape != gg.shape:
        raise ValueError(f"Shape mismatch pred {gp.shape} vs gt {gg.shape}")
    t = _otsu_threshold_from_hist(np.concatenate([gp.ravel(), gg.ravel()]))
    mp = gp >= t
    mg = gg >= t
    inter = np.logical_and(mp, mg).sum(dtype=np.int64)
    union = np.logical_or(mp, mg).sum(dtype=np.int64)
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def _save_png(path: str, rgb: np.ndarray) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except Exception:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(rgb)
        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def build_franka_cfg(use_tactile: bool = False):
    return SimpleNamespace(
        suite="franka",
        config=CONFIG_NAME,
        ckpt_path=CKPT_PATH,
        config_file=CONFIG_FILE,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        use_wrist_image=True,
        num_wrist_images=1,
        use_third_person_image=True,
        num_third_person_images=1,
        use_tactile=use_tactile,
        use_jpeg_compression=True,
        trained_with_image_aug=True,
        flip_images=False,
        use_variance_scale=False,
        chunk_size=CHUNK_SIZE,
        num_denoising_steps_action=NUM_DENOISING_STEPS,
    )


def load_episode(
    hdf5_path: str,
    cam_front_path: str,
    cam_high_path: str,
    tactile_left_path: str = None,
    tactile_right_path: str = None,
):
    """Load qpos (T, 6), action (T, 7), task_name, video frames, and optional tactile frames."""
    def _maybe_load_tactile_from_hdf5(f: h5py.File, resize_size: int):
        """
        Try to load tactile left/right frames from HDF5. Supports:
        - Stored as uint8 arrays (T,H,W,3)
        - Stored as per-frame JPEG bytes (T,) that need decoding
        Returns (left, right) or (None, None) if not found.
        """

        def _read_first_existing(keys):
            for k in keys:
                if k in f:
                    return k, f[k]
            return None, None

        left_key, left_ds = _read_first_existing(
            [
                "observations/tactile_left",
                "observations/tactile_left_image",
                "observations/tactile_left_rgb",
                "observations/tactile/left",
                "tactile_left",
            ]
        )
        right_key, right_ds = _read_first_existing(
            [
                "observations/tactile_right",
                "observations/tactile_right_image",
                "observations/tactile_right_rgb",
                "observations/tactile/right",
                "tactile_right",
            ]
        )
        if left_ds is None or right_ds is None:
            return None, None

        def _load_frames(ds):
            arr = ds[()]
            # Case 1: already uint8 frames (T,H,W,C)
            if isinstance(arr, np.ndarray) and arr.ndim == 4:
                frames = np.asarray(arr)
                if frames.dtype != np.uint8:
                    frames = np.clip(frames, 0, 255).astype(np.uint8)
                if resize_size is not None:
                    frames = resize_images(frames, resize_size)
                return frames

            # Case 2: JPEG bytes per frame (T,)
            if isinstance(arr, np.ndarray) and arr.ndim == 1:
                frames_list = []
                for item in arr:
                    frame = decode_single_jpeg_frame(item)
                    frames_list.append(frame)
                frames = np.stack(frames_list, axis=0).astype(np.uint8)
                if resize_size is not None:
                    frames = resize_images(frames, resize_size)
                return frames

            raise ValueError(f"Unsupported tactile dataset shape: {getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")

        tactile_left_local = _load_frames(left_ds)
        tactile_right_local = _load_frames(right_ds)
        print(f"Loaded tactile from HDF5: left={left_key} right={right_key} T={tactile_left_local.shape[0]}")
        return tactile_left_local, tactile_right_local

    with h5py.File(hdf5_path, "r") as f:
        qpos = np.array(f["observations/qpos"][:], dtype=np.float32)
        actions_gt = np.array(f["action"][:], dtype=np.float32)
        task_name = f.attrs.get("task_name", "unknown")
        if isinstance(task_name, bytes):
            task_name = task_name.decode("utf-8")
        tactile_left_h5, tactile_right_h5 = _maybe_load_tactile_from_hdf5(f, resize_size=COSMOS_IMAGE_SIZE)
    T = qpos.shape[0]
    assert actions_gt.shape[0] == T and actions_gt.shape[1] == 7, f"action shape {actions_gt.shape}"
    assert qpos.shape[1] == 6, f"qpos (state) shape {qpos.shape}, expected 6-dim pose"

    cam_front = load_video_as_images(cam_front_path, resize_size=COSMOS_IMAGE_SIZE)
    cam_high = load_video_as_images(cam_high_path, resize_size=COSMOS_IMAGE_SIZE)
    T = min(T, len(cam_front), len(cam_high))
    qpos = qpos[:T]
    actions_gt = actions_gt[:T]
    cam_front = cam_front[:T]
    cam_high = cam_high[:T]

    tactile_left, tactile_right = tactile_left_h5, tactile_right_h5
    if tactile_left is not None and tactile_right is not None:
        T = min(T, len(tactile_left), len(tactile_right))
        qpos = qpos[:T]
        actions_gt = actions_gt[:T]
        cam_front = cam_front[:T]
        cam_high = cam_high[:T]
        tactile_left = tactile_left[:T]
        tactile_right = tactile_right[:T]
    elif tactile_left_path and tactile_right_path:
        tactile_left = load_video_as_images(tactile_left_path, resize_size=COSMOS_IMAGE_SIZE)
        tactile_right = load_video_as_images(tactile_right_path, resize_size=COSMOS_IMAGE_SIZE)
        T = min(T, len(tactile_left), len(tactile_right))
        qpos = qpos[:T]
        actions_gt = actions_gt[:T]
        cam_front = cam_front[:T]
        cam_high = cam_high[:T]
        tactile_left = tactile_left[:T]
        tactile_right = tactile_right[:T]
        print(f"Loaded tactile videos: left={len(tactile_left)}, right={len(tactile_right)}")

    return qpos, actions_gt, task_name, cam_front, cam_high, tactile_left, tactile_right


def run_openloop(
    cfg,
    model,
    dataset_stats,
    qpos,
    cam_front,
    cam_high,
    instruction,
    out_dir: str,
    tactile_left=None,
    tactile_right=None,
    future_pred_eval: bool = False,
    frame_index_offset: int = 0,
):
    """Run policy at t=0, chunk_size, 2*chunk_size, ...

    If future_pred_eval is True and the model returns future_image_predictions, saves per-modality
    predicted vs GT frames under out_dir/future_prediction_eval/ and writes future_pred_iou_summary.json
    with mask-IoU (joint Otsu on grayscale).

    frame_index_offset: added to chunk indices in filenames, titles, and IoU JSON (episode-global step).

    Returns:
        pred_actions, get_action_times_s, iou_summary (dict, possibly empty).
    """
    T = qpos.shape[0]
    pred_actions = np.full((T, 7), np.nan, dtype=np.float32)
    get_action_times_s: list[float] = []
    iou_per_chunk: List[Dict[str, Any]] = []

    image_out_dir = os.path.join(out_dir, "future_images")
    os.makedirs(image_out_dir, exist_ok=True)
    eval_root = os.path.join(out_dir, "future_prediction_eval")
    pred_frames_dir = os.path.join(eval_root, "pred")
    gt_frames_dir = os.path.join(eval_root, "gt")

    start = 0
    while start < T:
        obs = {
            "primary_image": cam_front[start],
            "wrist_image": cam_high[start],
            "proprio": qpos[start],
        }
        if cfg.use_tactile and tactile_left is not None:
            obs["tactile_left_image"] = tactile_left[start]
            obs["tactile_right_image"] = tactile_right[start]

        try:
            t0 = time.perf_counter()
            action_return = get_action(
                cfg,
                model,
                dataset_stats,
                obs,
                instruction,
                seed=0,
                randomize_seed=False,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
            get_action_times_s.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"get_action failed at start={start + frame_index_offset}: {e}")
            import traceback
            traceback.print_exc()
            break
        chunk = action_return["actions"]
        chunk_len = min(len(chunk), T - start)
        for i in range(chunk_len):
            a = chunk[i]
            pred_actions[start + i] = np.asarray(a, dtype=np.float32).ravel()[:7]

        future_image_predictions = action_return.get("future_image_predictions", None)
        if future_image_predictions is not None:
            gt_idx = min(start + CHUNK_SIZE, T - 1)
            g_start = start + frame_index_offset
            g_gt = gt_idx + frame_index_offset
            _save_image_comparison(
                future_image_predictions,
                "future_image",
                cam_front[gt_idx],
                os.path.join(image_out_dir, f"primary_start_{g_start:04d}.png"),
            )
            _save_image_comparison(
                future_image_predictions,
                "future_wrist_image",
                cam_high[gt_idx],
                os.path.join(image_out_dir, f"wrist_start_{g_start:04d}.png"),
            )
            if cfg.use_tactile and tactile_left is not None:
                _save_image_comparison(
                    future_image_predictions,
                    "future_tactile_left",
                    tactile_left[gt_idx],
                    os.path.join(image_out_dir, f"tactile_left_start_{g_start:04d}.png"),
                )
                _save_image_comparison(
                    future_image_predictions,
                    "future_tactile_right",
                    tactile_right[gt_idx],
                    os.path.join(image_out_dir, f"tactile_right_start_{g_start:04d}.png"),
                )

            if future_pred_eval:
                chunk_tag = f"start{g_start:04d}_gt{g_gt:04d}"
                iou_row: Dict[str, float] = {}
                texture_row: Dict[str, Dict[str, float]] = {}
                pairs: List[Tuple[str, str, np.ndarray]] = [
                    ("future_image", "primary", cam_front[gt_idx]),
                    ("future_wrist_image", "wrist", cam_high[gt_idx]),
                ]
                if cfg.use_tactile and tactile_left is not None:
                    pairs.extend(
                        [
                            ("future_tactile_left", "tactile_left", tactile_left[gt_idx]),
                            ("future_tactile_right", "tactile_right", tactile_right[gt_idx]),
                        ]
                    )
                for pred_key, short_name, gt_arr in pairs:
                    if pred_key not in future_image_predictions:
                        continue
                    try:
                        pred_u8 = _to_uint8_hw3(future_image_predictions[pred_key])
                        gt_u8 = np.asarray(gt_arr)
                        if gt_u8.dtype != np.uint8:
                            gt_u8 = np.clip(np.round(gt_u8), 0, 255).astype(np.uint8)
                        if pred_u8.shape[:2] != gt_u8.shape[:2]:
                            try:
                                from PIL import Image

                                pred_u8 = np.array(
                                    Image.fromarray(pred_u8).resize(
                                        (gt_u8.shape[1], gt_u8.shape[0]), Image.BICUBIC
                                    )
                                )
                            except Exception:
                                pass
                        iou = foreground_mask_iou(pred_u8, gt_u8)
                        iou_row[pred_key] = iou
                        if "tactile" in pred_key:
                            texture_row[pred_key] = {
                                "edge_texture_iou_p88": edge_texture_iou(
                                    pred_u8, gt_u8, percentile=88.0
                                ),
                                "dark_inv_otsu_iou": dark_region_inv_otsu_iou(pred_u8, gt_u8),
                            }
                        _save_png(
                            os.path.join(pred_frames_dir, f"{short_name}_{chunk_tag}.png"),
                            pred_u8,
                        )
                        _save_png(
                            os.path.join(gt_frames_dir, f"{short_name}_{chunk_tag}.png"),
                            gt_u8,
                        )
                    except Exception as ex:
                        print(f"[future_pred_eval] skip {pred_key} at {chunk_tag}: {ex}")
                if iou_row:
                    row = {
                        "chunk_start": g_start,
                        "gt_timestep": int(g_gt),
                        "iou": iou_row,
                    }
                    if texture_row:
                        row["texture_iou"] = texture_row
                    iou_per_chunk.append(row)

        start += CHUNK_SIZE

    iou_summary: Dict[str, Any] = {
        "definition": (
            "Mask IoU: joint Otsu threshold on grayscale histogram of pred+GT; "
            "binary masks are (gray >= t); IoU = |Mp∧Mg| / |Mp∨Mg|."
        ),
        "texture_definition": (
            "Tactile-only: (1) edge_texture_iou_p88 — IoU of masks where |Laplacian(gray)| "
            "exceeds the 88th percentile of pred+GT combined (dot edges / texture). "
            "(2) dark_inv_otsu_iou — IoU of binary masks from joint Otsu on (255−gray), "
            "emphasizing dark dots vs bright background."
        ),
        "per_chunk": iou_per_chunk,
        "mean_iou": {},
        "mean_texture_iou": {},
    }
    if iou_per_chunk:
        keys = set()
        for row in iou_per_chunk:
            keys.update(row["iou"].keys())
        for k in sorted(keys):
            vals = [row["iou"][k] for row in iou_per_chunk if k in row["iou"]]
            iou_summary["mean_iou"][k] = float(np.mean(vals)) if vals else 0.0
        # Mean tactile texture metrics
        tkeys = set()
        for row in iou_per_chunk:
            ti = row.get("texture_iou") or {}
            tkeys.update(ti.keys())
        for mod in sorted(tkeys):
            edge_vals = []
            dark_vals = []
            for row in iou_per_chunk:
                ti = row.get("texture_iou") or {}
                if mod not in ti:
                    continue
                edge_vals.append(ti[mod]["edge_texture_iou_p88"])
                dark_vals.append(ti[mod]["dark_inv_otsu_iou"])
            if edge_vals:
                iou_summary["mean_texture_iou"][mod] = {
                    "edge_texture_iou_p88": float(np.mean(edge_vals)),
                    "dark_inv_otsu_iou": float(np.mean(dark_vals)),
                }
        if future_pred_eval:
            os.makedirs(eval_root, exist_ok=True)
            summ_path = os.path.join(eval_root, "future_pred_iou_summary.json")
            with open(summ_path, "w", encoding="utf-8") as f:
                json.dump(iou_summary, f, indent=2)
            print(f"[future_pred_eval] Wrote {summ_path}")
            print("[future_pred_eval] mean IoU:", iou_summary["mean_iou"])
            if iou_summary["mean_texture_iou"]:
                print("[future_pred_eval] mean tactile texture IoU:", iou_summary["mean_texture_iou"])

    return pred_actions, get_action_times_s, iou_summary


def _save_image_comparison(predictions: dict, key: str, gt_image, out_path: str):
    """Save a side-by-side GT vs predicted image comparison (fixed titles for figures)."""
    pred_image = predictions.get(key, None)
    if pred_image is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gt_image)
        try:
            axes[0].set_title("Ground Truth", subtitle="")
        except (TypeError, AttributeError, ValueError):
            axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        axes[1].imshow(np.asarray(pred_image).astype(np.uint8))
        try:
            axes[1].set_title("Dream-Tac", subtitle="")
        except (TypeError, AttributeError, ValueError):
            axes[1].set_title("Dream-Tac")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Failed to save {out_path}: {e}")


def plot_pred_vs_gt(
    actions_gt: np.ndarray, pred_actions: np.ndarray, out_path: str, x_offset: int = 0
):
    """Plot each action dim: predicted vs ground truth over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, dim = actions_gt.shape
    valid = ~np.isnan(pred_actions[:, 0])
    n_valid = int(np.sum(valid))
    x = np.arange(T, dtype=np.float64) + float(x_offset)

    dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    fig, axes = plt.subplots(dim, 1, figsize=(12, 2 * dim), sharex=True)
    if dim == 1:
        axes = [axes]
    for d in range(dim):
        ax = axes[d]
        ax.plot(x, actions_gt[:, d], label="GT", color="C0", alpha=0.8)
        ax.plot(x, pred_actions[:, d], label="Pred", color="C1", alpha=0.8)
        ax.set_ylabel(dim_names[d])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step")
    plt.suptitle(f"Open-loop: Predicted vs Ground Truth (valid steps: {n_valid}/{T})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Franka open-loop: HDF5 + videos -> pred vs GT plot")
    ep = "episode_0"
    train_dir = f"{DATA_ROOT}/train"
    parser.add_argument("--hdf5", type=str, default=f"{train_dir}/{ep}.hdf5")
    parser.add_argument("--cam_front", type=str, default=f"{train_dir}/{ep}_cam_front.mp4")
    parser.add_argument("--cam_high", type=str, default=f"{train_dir}/{ep}_cam_high.mp4")
    parser.add_argument("--tactile_left", type=str, default=f"{train_dir}/{ep}_tactile_rectify_left.mp4")
    parser.add_argument("--tactile_right", type=str, default=f"{train_dir}/{ep}_tactile_rectify_right.mp4")
    parser.add_argument("--out_dir", type=str, default="./openloop_out")
    parser.add_argument(
        "--openloop-hard-residual-cache",
        action="store_true",
        help="Hard TeaCache-style block residual (5 denoise steps: full blocks only at steps 0 and 2; "
        "steps 1,3,4 reuse cached Δ through DiT blocks). Patched only for this script.",
    )
    parser.add_argument(
        "--compare-cache-timing",
        action="store_true",
        help="Run full openloop twice: baseline then hard-cache; print mean get_action times and speedup. "
        "Requires FRANKA_NUM_DENOISING_STEPS=5 for the cache path.",
    )
    parser.add_argument(
        "--future_pred_eval",
        action="store_true",
        help="Save predicted vs GT future frames (primary, wrist, tactile L/R) and compute IoU metrics; "
        "writes future_prediction_eval/ and future_pred_iou_summary.json. Tactile also gets texture-focused IoU: "
        "edge_texture_iou_p88 (|Laplacian| top 88%%) and dark_inv_otsu_iou (joint Otsu on 255−gray).",
    )
    parser.add_argument(
        "--frame_start",
        type=int,
        default=0,
        help="First timestep (inclusive) into the loaded episode; open-loop runs only on [frame_start, frame_end).",
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=-1,
        help="Exclusive end timestep; default -1 means use full episode length (clamped to T).",
    )
    args = parser.parse_args()

    for p in [args.hdf5, args.cam_front, args.cam_high]:
        if not os.path.isfile(p):
            print(f"Error: not a file: {p}")
            sys.exit(1)

    use_tactile = (
        args.tactile_left
        and args.tactile_right
        and os.path.isfile(args.tactile_left)
        and os.path.isfile(args.tactile_right)
    )
    if not use_tactile:
        print("Tactile MP4s not found or not specified; will try HDF5 tactile if present.")
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading episode...")
    qpos, actions_gt, task_name, cam_front, cam_high, tactile_left, tactile_right = load_episode(
        args.hdf5, args.cam_front, args.cam_high,
        args.tactile_left if use_tactile else None,
        args.tactile_right if use_tactile else None,
    )
    T = qpos.shape[0]
    use_tactile = tactile_left is not None and tactile_right is not None
    print(f"Episode length T={T}, task_name={task_name!r}, use_tactile={use_tactile}")

    fs = max(0, int(args.frame_start))
    fe = T if int(args.frame_end) < 0 else int(args.frame_end)
    fe = min(fe, T)
    if fs >= fe:
        print(f"Error: require frame_start < frame_end, got [{fs}, {fe}) with T={T}")
        sys.exit(1)
    if fs > 0 or fe < T:
        if int(args.frame_end) > T:
            print(f"Note: frame_end clamped from {args.frame_end} to {T} (episode length).")
        qpos = qpos[fs:fe]
        actions_gt = actions_gt[fs:fe]
        cam_front = cam_front[fs:fe]
        cam_high = cam_high[fs:fe]
        if tactile_left is not None:
            tactile_left = tactile_left[fs:fe]
        if tactile_right is not None:
            tactile_right = tactile_right[fs:fe]
        print(f"Timestep slice (global): [{fs}, {fe}) -> length {fe - fs}")
    frame_index_offset = fs

    print("Loading model (reusing FRANKA_* env)...")
    cfg = build_franka_cfg(use_tactile=use_tactile)
    cfg.ckpt_path = CKPT_PATH or cfg.ckpt_path
    if not CKPT_PATH or not os.path.exists(CKPT_PATH):
        print("Set FRANKA_COSMOS_CKPT to the checkpoint directory.")
        sys.exit(1)
    model, _ = get_model(cfg)
    dataset_stats = {}
    if DATASET_STATS_PATH and os.path.exists(DATASET_STATS_PATH):
        dataset_stats = load_dataset_stats(DATASET_STATS_PATH)
    else:
        cfg.unnormalize_actions = False
    if T5_EMBEDDINGS_PATH and os.path.exists(T5_EMBEDDINGS_PATH):
        init_t5_text_embeddings_cache(T5_EMBEDDINGS_PATH)

    def _print_get_action_timing(times: list[float], label: str) -> None:
        if not times:
            return
        arr = np.asarray(times, dtype=np.float64)
        prefix = f"{label} " if label else ""
        print(
            f"{prefix}get_action timing (s): n={len(arr)} "
            f"mean={arr.mean():.4f} std={arr.std(ddof=0):.4f} "
            f"min={arr.min():.4f} max={arr.max():.4f} total={arr.sum():.4f}"
        )

    if args.compare_cache_timing:
        if NUM_DENOISING_STEPS != 5:
            print("Error: --compare-cache-timing requires FRANKA_NUM_DENOISING_STEPS=5 (hard cache schedule is fixed for 5 steps).")
            sys.exit(1)
        from cosmos_policy.experiments.robot.openloop_hard_residual_cache import (
            apply_openloop_hard_residual_cache,
            remove_openloop_hard_residual_cache,
        )

        out_base = args.out_dir
        print("Running open-loop policy (baseline, no hard residual cache)...")
        pred_b, times_b, _ = run_openloop(
            cfg, model, dataset_stats, qpos, cam_front, cam_high,
            task_name, os.path.join(out_base, "baseline"),
            tactile_left=tactile_left, tactile_right=tactile_right,
            future_pred_eval=args.future_pred_eval,
            frame_index_offset=frame_index_offset,
        )
        _print_get_action_timing(times_b, "Baseline")

        apply_openloop_hard_residual_cache(model, num_denoising_steps=NUM_DENOISING_STEPS)
        print("Running open-loop policy (hard residual cache, 5-step: 2 full DiT block passes per sample)...")
        pred_c, times_c, _ = run_openloop(
            cfg, model, dataset_stats, qpos, cam_front, cam_high,
            task_name, os.path.join(out_base, "hard_cache"),
            tactile_left=tactile_left, tactile_right=tactile_right,
            future_pred_eval=args.future_pred_eval,
            frame_index_offset=frame_index_offset,
        )
        _print_get_action_timing(times_c, "Hard-cache")
        remove_openloop_hard_residual_cache(model)

        if times_b and times_c:
            mb, mc = float(np.mean(times_b)), float(np.mean(times_c))
            print(
                f"Mean get_action: baseline={mb:.4f}s, hard-cache={mc:.4f}s, "
                f"ratio={mb/mc:.3f}x (higher means cache is faster)"
            )

        plot_pred_vs_gt(
            actions_gt,
            pred_b,
            os.path.join(out_base, "baseline", "openloop_pred_vs_gt.png"),
            x_offset=frame_index_offset,
        )
        plot_pred_vs_gt(
            actions_gt,
            pred_c,
            os.path.join(out_base, "hard_cache", "openloop_pred_vs_gt.png"),
            x_offset=frame_index_offset,
        )
        pred_actions = pred_c
        get_action_times_s = times_c
    else:
        if args.openloop_hard_residual_cache:
            from cosmos_policy.experiments.robot.openloop_hard_residual_cache import apply_openloop_hard_residual_cache

            apply_openloop_hard_residual_cache(model, num_denoising_steps=cfg.num_denoising_steps_action)
            if cfg.num_denoising_steps_action != 5:
                print(
                    f"Note: hard block cache uses full schedule only when num_denoising_steps==5 "
                    f"(current {cfg.num_denoising_steps_action}); otherwise full DiT every step."
                )

        print("Running open-loop policy...")
        pred_actions, get_action_times_s, _iou_summary = run_openloop(
            cfg, model, dataset_stats, qpos, cam_front, cam_high,
            task_name, args.out_dir,
            tactile_left=tactile_left, tactile_right=tactile_right,
            future_pred_eval=args.future_pred_eval,
            frame_index_offset=frame_index_offset,
        )

        if get_action_times_s:
            _print_get_action_timing(get_action_times_s, "")

        out_plot = os.path.join(args.out_dir, "openloop_pred_vs_gt.png")
        plot_pred_vs_gt(actions_gt, pred_actions, out_plot, x_offset=frame_index_offset)

    valid = ~np.isnan(pred_actions[:, 0])
    if np.any(valid):
        mae = np.nanmean(np.abs(pred_actions - actions_gt))
        mae_per_dim = np.nanmean(np.abs(pred_actions - actions_gt), axis=0)
        print(f"MAE (over valid steps, last run): {mae:.6f}")
        print("MAE per dim:", " ".join(f"{x:.6f}" for x in mae_per_dim))
    print("Done.")


if __name__ == "__main__":
    main()
