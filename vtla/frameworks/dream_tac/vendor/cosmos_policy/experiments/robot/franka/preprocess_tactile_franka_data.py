# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Preprocess Franka/tactile dataset: HDF5 with puppet joint/gripper/pose and JPEG images.

Input (tactile_data layout):
    /dataset_path/<task_folder>/  e.g. insert_nut_into_screw/
        <session>/   e.g. insert_screw_202602010_01/
            0.hdf5, 1.hdf5, ...

Each HDF5 contains:
    - observations/images/cam_front, observations/images/cam_high  (JPEG bytes per frame)
    - observations/images/tactile_rectify_left, observations/images/tactile_rectify_right (optional, JPEG bytes)
    - puppet/pose (T, 6)  [end-effector pose: x, y, z, roll, pitch, yaw]
    - puppet/gripper (optional): if present, (T,) or (T, 1); if absent, gripper is filled with 1.0 (open).

Output (ALOHA-style layout for FrankaDataset):
    - qpos (state) = previous timestep's action pose only (no gripper) -> (T, 6).
      state[0] = action[0, :6]; state[t] = action[t-1, :6].
    - action = pose 6 + gripper 1 -> (T, 7)
    - observations/video_paths: cam_front, cam_high, and (if present) tactile_rectify_left, tactile_rectify_right; each resized and stored as MP4.

Task name: top-level folder name with underscores removed (e.g. insert_nut_into_screw -> insert nut into screw).

Usage:
uv run --extra cu128 --group libero --python 3.10 python cosmos_policy/experiments/robot/franka/preprocess_tactile_franka_data.py \
    --dataset_path /path/to/cut_banana/cut_banana_20260321 \
    --out_base_dir /path/to/attention_data_cut_banana \
    --percent_val 0.0
"""

import argparse
import json
import os
import random

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_video_from_images(images, video_path, fps=30):
    """Create MP4 from (num_frames, H, W, 3) RGB array."""
    if len(images) == 0:
        raise ValueError("No images provided")
    height, width, channels = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for img in images:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if channels == 3 else img
        out.write(img_bgr)
    out.release()


def decode_image_frame(raw):
    """Decode one frame: raw is uint8 array (JPEG bytes). Returns RGB numpy array."""
    if isinstance(raw, np.ndarray) and raw.dtype == np.uint8:
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode JPEG frame")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unexpected image type: {type(raw)}")


def load_franka_hdf5(demo_path):
    """
    Load one tactile Franka episode. State (qpos) is 6D pose only; action is 7D pose + gripper.

    Returns:
        qpos (T, 6): state = previous timestep's action pose only. qpos[0]=action[0,:6], qpos[t]=action[t-1,:6].
        action (T, 7): pose 6 + gripper 1.
        image_dict with keys cam_front, cam_high, and optionally tactile_rectify_left, tactile_rectify_right (each list of RGB arrays).
    """
    if not os.path.isfile(demo_path):
        raise FileNotFoundError(demo_path)

    with h5py.File(demo_path, "r") as root:
        cam_front_raw = root["observations/images/cam_front"][:]
        cam_high_raw = root["observations/images/cam_high"][:]
        T = len(cam_front_raw)

        # Optional tactile images: same key path in output HDF5
        optional_image_keys = ["tactile_rectify_left", "tactile_rectify_right"]
        optional_raw = {}
        for key in optional_image_keys:
            path = f"observations/images/{key}"
            if path in root:
                raw = root[path][:]
                if len(raw) == T:
                    optional_raw[key] = raw
                # else skip (length mismatch)

        pose = np.array(root["puppet/pose"][:], dtype=np.float32)

        # Normalize to (T, 6): some HDF5s use (T, n), others 1D flattened or (n, T)
        if pose.ndim == 1:
            pose = pose.reshape(-1, 6)
        elif pose.shape[1] != 6:
            pose = pose.T
        if pose.shape[0] != T:
            pose = pose[:T]
        pose = np.ascontiguousarray(pose).astype(np.float32)

        # Gripper: use puppet/gripper if present, else constant 1.0 (open)
        if "puppet/gripper" in root:
            gripper = np.array(root["puppet/gripper"][:], dtype=np.float32)
            if gripper.ndim == 1:
                gripper = gripper.reshape(-1, 1)
            if gripper.shape[0] != T:
                gripper = gripper[:T]
            if gripper.shape[1] != 1:
                gripper = gripper[:, :1]
        else:
            gripper = np.ones((T, 1), dtype=np.float32)

        # action (T, 7) = pose 6 + gripper 1
        action = np.concatenate([pose, gripper], axis=1).astype(np.float32)
        # qpos (state) = previous timestep's action pose only, no gripper (T, 6)
        qpos = np.zeros((T, 6), dtype=np.float32)
        qpos[0] = action[0, :6]
        qpos[1:] = action[:-1, :6]
        image_dict = {
            "cam_front": [decode_image_frame(cam_front_raw[i]) for i in range(T)],
            "cam_high": [decode_image_frame(cam_high_raw[i]) for i in range(T)],
        }
        for key, raw in optional_raw.items():
            image_dict[key] = [decode_image_frame(raw[i]) for i in range(T)]
    return qpos, action, image_dict


def load_and_preprocess_all_episodes(demo_paths, out_dataset_dir, split_name, task_name, args):
    metadata_list = []

    for idx, demo in enumerate(tqdm(demo_paths, desc=f"Processing {split_name} episodes")):
        qpos, action, image_dict = load_franka_hdf5(demo)
        episode_len = len(image_dict["cam_high"])
        # Process all image keys (cam_front, cam_high, and optionally tactile_rectify_left, tactile_rectify_right)
        cam_names = list(image_dict.keys())

        video_paths = {}
        for k in cam_names:
            resized = []
            for i in range(episode_len):
                arr = np.array(
                    Image.fromarray(image_dict[k][i]).resize(
                        (args.img_resize_size, args.img_resize_size), resample=Image.BICUBIC
                    )
                )
                resized.append(arr)
            resized = np.stack(resized)
            video_filename = f"episode_{idx}_{k}.mp4"
            video_path = os.path.join(out_dataset_dir, video_filename)
            create_video_from_images(resized, video_path, fps=args.video_fps)
            video_paths[k] = video_filename

        data_dict = {
            "qpos": qpos,
            "action": action,
            "video_paths": video_paths,
            "task_name": task_name,
        }
        save_franka_hdf5(out_dataset_dir, data_dict, idx)

        episode_metadata = {
            "original_file_path": demo,
            "preprocessed_episode_name": f"episode_{idx}.hdf5",
            "preprocessed_video_files": list(video_paths.values()),
            "split": split_name,
            "preprocessed_index": idx,
            "episode_length": episode_len,
            "task_name": task_name,
        }
        metadata_list.append(episode_metadata)

    return metadata_list


def save_franka_hdf5(out_dataset_dir, data_dict, episode_idx):
    """Save one preprocessed episode: qpos (T, 6) state=prev action pose only, action (T, 7) pose+gripper, video_paths, relative_action."""
    qpos = data_dict["qpos"]
    action = data_dict["action"]
    assert qpos.ndim == 2 and qpos.shape[1] == 6, f"qpos must be (T, 6), got {qpos.shape}"
    assert action.ndim == 2 and action.shape[1] == 7, f"action must be (T, 7), got {action.shape}"

    out_path = os.path.join(out_dataset_dir, f"episode_{episode_idx}.hdf5")
    with h5py.File(out_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        episode_len = qpos.shape[0]
        root.attrs["task_name"] = data_dict["task_name"]

        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        video_paths_group = obs.create_group("video_paths")
        for cam_name, video_path in data_dict["video_paths"].items():
            video_paths_group.create_dataset(cam_name, data=video_path.encode("utf-8"))

        root.create_dataset("action", data=action)

        relative_actions = np.zeros_like(action)
        relative_actions[:-1] = action[1:] - action[:-1]
        relative_actions[-1] = relative_actions[-2]
        root.create_dataset("relative_action", data=relative_actions)
    print(f"Saved: {out_path}")


def randomly_split_episode_paths(all_demo_paths, percent_val):
    random.shuffle(all_demo_paths)
    n = len(all_demo_paths)
    n_val = max(0, int(n * percent_val))
    n_train = n - n_val
    return all_demo_paths[:n_train], all_demo_paths[n_train:]


def main(args):
    os.makedirs(args.out_base_dir, exist_ok=True)
    # Task folder = basename of dataset_path (e.g. insert_nut_into_screw)
    task_folder = os.path.basename(args.dataset_path.rstrip("/"))
    task_name = task_folder.replace("_", "")  # 文件夹去掉_
    out_dataset_dir = os.path.join(args.out_base_dir, task_folder)
    os.makedirs(out_dataset_dir, exist_ok=True)

    all_demo_paths = []
    for root_dir, _dirs, files in os.walk(args.dataset_path):
        for f in files:
            if f.endswith(".hdf5"):
                all_demo_paths.append(os.path.join(root_dir, f))
    all_demo_paths.sort()
    if not all_demo_paths:
        raise SystemExit(f"No .hdf5 files under {args.dataset_path}")

    train_paths, val_paths = randomly_split_episode_paths(all_demo_paths, args.percent_val)
    print(f"Total episodes: {len(all_demo_paths)}; train: {len(train_paths)}, val: {len(val_paths)}; task_name: {task_name}")

    out_train = os.path.join(out_dataset_dir, "train")
    out_val = os.path.join(out_dataset_dir, "val")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_val, exist_ok=True)

    train_metadata = load_and_preprocess_all_episodes(train_paths, out_train, "train", task_name, args)
    val_metadata = load_and_preprocess_all_episodes(val_paths, out_val, "val", task_name, args)

    all_metadata = {
        "dataset_info": {
            "original_dataset_path": args.dataset_path,
            "preprocessed_dataset_path": out_dataset_dir,
            "task_name": task_name,
            "total_episodes": len(all_demo_paths),
            "train_episodes": len(train_paths),
            "val_episodes": len(val_paths),
            "validation_percentage": args.percent_val,
            "preprocessing_settings": {"img_resize_size": args.img_resize_size, "video_fps": args.video_fps},
        },
        "episode_mapping": train_metadata + val_metadata,
    }
    metadata_path = os.path.join(out_dataset_dir, "preprocessing_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Done. Metadata: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="/path/to/tactile_data/pick_eraser_and_erase_marker_from_whiteboard",
        help="Path to task folder containing (recursive) .hdf5 files.",
    )
    parser.add_argument("--out_base_dir", required=True, help="Base directory for preprocessed output.")
    parser.add_argument("--percent_val", type=float, default=0.05, help="Fraction of episodes for validation.")
    parser.add_argument("--img_resize_size", type=int, default=256, help="Resize images to this size (square).")
    parser.add_argument("--video_fps", type=int, default=25, help="FPS for output MP4 videos.")
    args = parser.parse_args()
    main(args)
