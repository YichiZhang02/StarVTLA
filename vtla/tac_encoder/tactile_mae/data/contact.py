"""Contact-frame detection for tactile MAE.

A tactile frame is "in contact" when its image texture (standard deviation) is
high enough. On our gel sensors the discriminative signal is the per-channel std
(max over RGB channels) on the 0-255 scale: idle frames are nearly flat (~0.1),
contact frames rise to several units.

Per-frame scores are expensive to recompute, so they are cached per dataset
under ``<dataset_root>/<dataset_id>/meta/contact_std.npz`` keyed by absolute
frame ``index``.

Building the cache decodes every (kept) frame once, **sequentially per episode**:
all frames of an episode/camera are pulled in a single batched decode call
instead of one random-access seek-and-decode per frame. Random per-frame access
re-decodes from the preceding keyframe every time (~GOP/2 wasted frames each),
so grouping is ~1-2 orders of magnitude faster. ``stride`` decodes only every
N-th frame and nearest-fills the gaps -- contact std is temporally smooth at
30 Hz, so this is a faithful approximation of a per-frame gate.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from vtla.datasets.lerobot_dataset import LeRobotDataset
from vtla.datasets.video_utils import decode_tactile_video_frames_pyav, decode_video_frames


def contact_score(img):
    """img: CHW float tensor in [0, 1]. Max per-channel std on the 0-255 scale."""
    x = img * 255.0
    return float(max(x[c].std() for c in range(x.shape[0])))


def _std_per_channel_max(frames_uint8):
    """frames_uint8: (N, C, H, W) uint8. Returns (N,) max-over-channel std on 0-255 scale.

    Matches ``contact_score`` (torch default unbiased std) but vectorised over the
    whole batch of decoded frames at once.
    """
    x = frames_uint8.to(torch.float32)
    per_channel = x.flatten(2).std(dim=2)   # (N, C)
    return per_channel.amax(dim=1)          # (N,)


def _nearest_fill(sel_idx, sel_vals, length):
    """Expand strided samples to a per-frame array by nearest-neighbour fill.

    sel_idx: sorted frame indices that were actually scored (within [0, length)).
    sel_vals: their scores. Returns float32 array of size ``length``.
    """
    sel_idx = np.asarray(sel_idx)
    sel_vals = np.asarray(sel_vals, dtype=np.float32)
    if len(sel_idx) == 1:
        return np.full(length, sel_vals[0], dtype=np.float32)
    all_idx = np.arange(length)
    pos = np.clip(np.searchsorted(sel_idx, all_idx), 1, len(sel_idx) - 1)
    left, right = sel_idx[pos - 1], sel_idx[pos]
    take_right = (all_idx - left) > (right - all_idx)
    return np.where(take_right, sel_vals[pos], sel_vals[pos - 1]).astype(np.float32)


def load_or_compute_contact_std(dataset_root, dataset_id, camera_keys,
                                tolerance_s=0.1, video_backend="pyav", num_workers=8,
                                stride=1, force=False):
    """Return {camera: np.ndarray indexed by absolute frame index} of contact scores.

    stride: decode/score every N-th frame per episode and nearest-fill the rest
        (1 = score every frame). The on-disk cache format is identical regardless
        of stride; delete contact_std.npz to rebuild with a different stride.
    """
    meta_dir = os.path.join(dataset_root, dataset_id, "meta")
    cache_fp = os.path.join(meta_dir, "contact_std.npz")
    if os.path.exists(cache_fp) and not force:
        data = np.load(cache_fp)
        if all(c in data for c in camera_keys):
            return {c: data[c] for c in camera_keys}

    # The build is a one-time pass that decodes every (kept) frame once; FFV1 is
    # all-intra and decoding releases the GIL, so it scales with cores well past the
    # training dataloader's num_workers. Use the idle cores (capped to avoid thrash).
    pool_workers = max(int(num_workers), min(48, os.cpu_count() or 8))
    print(f"[contact] computing per-channel std cache for {dataset_id} "
          f"(stride={stride}, workers={pool_workers}, backend={video_backend}) ...")
    ds = LeRobotDataset(repo_id=dataset_id, root=os.path.join(dataset_root, dataset_id),
                        video_backend=video_backend, tolerance_s=tolerance_s)
    meta = ds.meta
    root = ds.root
    fps = meta.fps
    total = meta.total_frames
    n_eps = len(meta.episodes)
    arrays = {c: np.full(total, np.nan, dtype=np.float32) for c in camera_keys}

    def _process(ep_idx):
        ep = meta.episodes[ep_idx]
        length = int(ep["length"])
        from_abs = int(ep["dataset_from_index"])
        # Frame indices to actually decode: every `stride`-th, always incl. the last
        # so the nearest-fill is bounded on both ends.
        sel = list(range(0, length, stride))
        if sel[-1] != length - 1:
            sel.append(length - 1)
        for cam in camera_keys:
            from_ts = float(ep[f"videos/{cam}/from_timestamp"])
            video_path = root / meta.get_video_file_path(ep_idx, cam)
            query_ts = [from_ts + i / fps for i in sel]
            # One batched (sequential) decode for the whole episode segment.
            if meta.features[cam].get("tactile_encoding") == "tactile_u16_fixed_v1":
                frames = decode_tactile_video_frames_pyav(
                    video_path, query_ts, tolerance_s, return_uint8=True
                )
            else:
                frames = decode_video_frames(
                    video_path, query_ts, tolerance_s, video_backend, return_uint8=True
                )
            scores = _std_per_channel_max(frames).numpy()
            arrays[cam][from_abs:from_abs + length] = _nearest_fill(sel, scores, length)

    done = 0
    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        # Video decoding releases the GIL, so threads give real decode parallelism;
        # each episode writes a disjoint abs-index slice, so no locking is needed.
        for _ in pool.map(_process, range(n_eps)):
            done += 1
            if done % 50 == 0 or done == n_eps:
                print(f"[contact]   {dataset_id}: {done}/{n_eps} episodes")

    os.makedirs(meta_dir, exist_ok=True)
    np.savez(cache_fp, **arrays)
    print(f"[contact] saved {cache_fp}")
    return arrays
