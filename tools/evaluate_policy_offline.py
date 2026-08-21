#!/usr/bin/env python

"""Evaluate a trained policy against a dataset without commanding a robot.

Each selected episode is replayed chronologically. Every evaluated observation is
passed to ``predict_action_chunk`` and compared with the dataset ground truth in:

1. the checkpoint's training action space (after unnormalization only), and
2. the robot command space (after the complete inference postprocessor).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/starvtla-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/starvtla-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm

from vtla.datasets.dataset_metadata import LeRobotDatasetMetadata
from vtla.datasets.factory import resolve_delta_timestamps
from vtla.datasets.lerobot_dataset import LeRobotDataset
from vtla.engine.configs import PreTrainedConfig
from vtla.engine.processor.converters import policy_action_to_transition, transition_to_policy_action
from vtla.engine.processor.normalize_processor import UnnormalizerProcessorStep
from vtla.engine.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from vtla.frameworks.factory import make_policy, make_pre_post_processors
from vtla.frameworks.sensor_routing import ACTION_ABSOLUTE_EE, ACTION_ABSOLUTE_QUAT
from vtla.frameworks.utils import populate_queues

LOGGER = logging.getLogger(__name__)

ACTION_SOURCE_BY_REPRESENTATION = {
    "joint": ACTION,
    "rot6d": ACTION_ABSOLUTE_EE,
    "quat": ACTION_ABSOLUTE_QUAT,
}


@dataclass
class EpisodeResult:
    episode_index: int
    observation_frames: list[int] = field(default_factory=list)
    target_frames: list[int] = field(default_factory=list)
    valid_chunks: list[np.ndarray] = field(default_factory=list)
    mode_predictions: list[np.ndarray] = field(default_factory=list)
    mode_ground_truth: list[np.ndarray] = field(default_factory=list)
    robot_predictions: list[np.ndarray] = field(default_factory=list)
    robot_ground_truth: list[np.ndarray] = field(default_factory=list)


def parse_episode_spec(spec: str, available: list[int]) -> list[int]:
    """Parse ``all``, comma-separated ids, and inclusive ranges such as ``0-3,7``."""
    if spec.strip().lower() == "all":
        return available

    selected: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending episode range: {token!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    unknown = sorted(selected - set(available))
    if unknown:
        raise ValueError(f"Dataset does not contain episode(s): {unknown}")
    if not selected:
        raise ValueError("No episodes were selected.")
    return sorted(selected)


def action_source_for_config(policy_cfg: PreTrainedConfig, features: dict[str, dict]) -> str:
    representation = getattr(policy_cfg, "action_representation", None)
    if representation not in ACTION_SOURCE_BY_REPRESENTATION:
        raise ValueError(
            f"Unsupported checkpoint action_representation={representation!r}; expected one of "
            f"{sorted(ACTION_SOURCE_BY_REPRESENTATION)}."
        )
    source = ACTION_SOURCE_BY_REPRESENTATION[representation]
    if source not in features:
        raise ValueError(
            f"action_mode={getattr(policy_cfg, 'action_mode', None)!r} requires dataset feature "
            f"{source!r}. Re-run scripts/process_joint_data.sh for this dataset."
        )
    return source


def default_output_dir(checkpoint: Path, dataset_root: Path) -> Path:
    if checkpoint.name == "pretrained_model" and checkpoint.parent.parent.name == "checkpoints":
        pretrained_dir = checkpoint.parents[2]
        checkpoint_id = checkpoint.parent.name
        return pretrained_dir / "offline_eval" / checkpoint_id / dataset_root.name

    checkpoint_dir = checkpoint.parent if checkpoint.name == "pretrained_model" else checkpoint
    return checkpoint_dir / "offline_eval" / dataset_root.name


def _apply_action_step(step: UnnormalizerProcessorStep, action: torch.Tensor) -> torch.Tensor:
    transition = policy_action_to_transition(action.clone())
    return transition_to_policy_action(step(transition))


def find_unnormalizer(postprocessor: Any) -> UnnormalizerProcessorStep:
    matches = [step for step in postprocessor.steps if isinstance(step, UnnormalizerProcessorStep)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one UnnormalizerProcessorStep in checkpoint postprocessor, got {len(matches)}."
        )
    return matches[0]


def _prepare_diffusion_predict(policy: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """Perform the observation-queue part of DiffusionPolicy.select_action only."""
    from vtla.frameworks.diffusion.modeling_diffusion import _resize_images_to_common

    prepared = dict(batch)
    prepared.pop(ACTION, None)
    if policy.config.image_features:
        prepared = _resize_images_to_common(
            prepared, policy.config.image_features, policy.config.resize_imgs_to
        )
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[key] for key in policy.config.image_features], dim=-4
        )
    policy._queues = populate_queues(policy._queues, prepared)
    return prepared


def prepare_predict_input(policy: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """Remove supervision and update any rollout state needed by predict_action_chunk."""
    model_input = {
        key: value
        for key, value in batch.items()
        if key != ACTION and not key.endswith("_is_pad")
    }
    if getattr(policy.config, "type", None) == "diffusion":
        return _prepare_diffusion_predict(policy, model_input)
    return model_input


def _scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def build_preprocessor_input(
    item: dict[str, Any],
    policy_cfg: PreTrainedConfig,
    metadata: LeRobotDatasetMetadata,
    action_source: str,
    include_ground_truth: bool,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """Build the same single-observation input used by online inference."""
    required_observations = set(policy_cfg.input_features or {})
    required_observations.add(OBS_STATE)  # raw joints are also the FK/relative-action anchor
    required_observations.update(policy_cfg.decoded_video_keys())

    batch: dict[str, Any] = {
        key: value
        for key, value in item.items()
        if key in required_observations
    }
    for key in metadata.camera_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
            batch[key] = value.to(dtype=torch.float32) / 255.0

    batch["task"] = item.get("task", getattr(policy_cfg, "single_task", None) or "")
    for key in ("index", "task_index", "episode_index"):
        if key in item:
            batch[key] = item[key]

    padding = None
    if include_ground_truth:
        ground_truth = item[action_source]
        if ground_truth.ndim != 2:
            raise ValueError(
                f"Expected chunked {action_source!r} with shape [T,D], got {tuple(ground_truth.shape)}."
            )
        batch[ACTION] = ground_truth.unsqueeze(0)
        source_padding_key = f"{action_source}_is_pad"
        padding = item.get(source_padding_key)
        if padding is None:
            padding = torch.zeros(ground_truth.shape[0], dtype=torch.bool)
        padding = padding.to(dtype=torch.bool)
        batch[f"{ACTION}_is_pad"] = padding.unsqueeze(0)
    return batch, padding


def compute_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    dimension_names: list[str],
) -> dict[str, Any]:
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Metric shape mismatch: {prediction.shape} vs {ground_truth.shape}")
    if prediction.size == 0:
        return {"count": 0}

    error = prediction.astype(np.float64) - ground_truth.astype(np.float64)
    abs_error = np.abs(error)
    per_dim_mae = abs_error.mean(axis=0)
    per_dim_rmse = np.sqrt(np.square(error).mean(axis=0))
    return {
        "count": int(prediction.shape[0]),
        "mae": float(abs_error.mean()),
        "l1": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "max_abs_error": float(abs_error.max()),
        "per_dimension": {
            name: {
                "mae": float(per_dim_mae[index]),
                "l1": float(per_dim_mae[index]),
                "rmse": float(per_dim_rmse[index]),
                "max_abs_error": float(abs_error[:, index].max()),
            }
            for index, name in enumerate(dimension_names)
        },
    }


def _flatten_valid_chunks(
    predictions: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if valid.shape != predictions.shape[:2]:
        raise ValueError(f"Padding mask shape {valid.shape} does not match chunks {predictions.shape}.")
    return predictions[valid], ground_truth[valid]


def _dimension_names(feature: dict[str, Any], width: int, prefix: str) -> list[str]:
    names = feature.get("names") or []
    if len(names) != width:
        return [f"{prefix}_{index}" for index in range(width)]
    return [str(name) for name in names]


def plot_comparison(
    path: Path,
    frames: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    names: list[str],
    title: str,
) -> None:
    rows = prediction.shape[-1]
    fig, axes = plt.subplots(rows, 1, figsize=(14, max(3.0, 2.1 * rows)), sharex=True)
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(frames, ground_truth[:, index], label="GT", linewidth=1.35, color="#1f77b4")
        axis.plot(frames, prediction[:, index], label="Prediction", linewidth=1.1, color="#d62728")
        dim_error = prediction[:, index] - ground_truth[:, index]
        axis.set_ylabel(names[index], rotation=0, ha="right", va="center")
        axis.set_title(
            f"MAE {np.abs(dim_error).mean():.5g} | RMSE {np.sqrt(np.square(dim_error).mean()):.5g}",
            fontsize=8,
            loc="right",
        )
        axis.grid(alpha=0.22)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Target frame in episode")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _episode_arrays(result: EpisodeResult) -> dict[str, np.ndarray]:
    if not result.mode_predictions:
        raise RuntimeError(f"Episode {result.episode_index} has no valid evaluated targets.")
    return {
        "observation_frames": np.asarray(result.observation_frames, dtype=np.int64),
        "target_frames": np.asarray(result.target_frames, dtype=np.int64),
        "chunk_valid": np.stack(result.valid_chunks),
        "action_mode_prediction_chunks": np.stack(result.mode_predictions),
        "action_mode_ground_truth_chunks": np.stack(result.mode_ground_truth),
        "robot_command_prediction_chunks": np.stack(result.robot_predictions),
        "robot_command_ground_truth_chunks": np.stack(result.robot_ground_truth),
    }


def _space_metrics(
    arrays: dict[str, np.ndarray], names: list[str], prediction_step: int
) -> dict[str, Any]:
    pred_key = "prediction_chunks"
    gt_key = "ground_truth_chunks"
    prediction = arrays[pred_key]
    ground_truth = arrays[gt_key]
    valid = arrays["chunk_valid"]
    selected_valid = valid[:, prediction_step]
    selected_prediction = prediction[selected_valid, prediction_step]
    selected_ground_truth = ground_truth[selected_valid, prediction_step]
    chunk_prediction, chunk_ground_truth = _flatten_valid_chunks(prediction, ground_truth, valid)
    return {
        "selected_prediction_step": compute_metrics(
            selected_prediction, selected_ground_truth, names
        ),
        "all_valid_chunk_steps": compute_metrics(chunk_prediction, chunk_ground_truth, names),
    }


def evaluate(args: argparse.Namespace) -> Path:
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {checkpoint / 'config.json'}")
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {dataset_root / 'meta' / 'info.json'}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    repo_id = args.dataset_repo_id or dataset_root.name
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    checkpoint_robot_type = getattr(policy_cfg, "robot_type", None)
    if not checkpoint_robot_type:
        raise ValueError("Checkpoint config is missing the required robot_type.")
    if not metadata.robot_type:
        raise ValueError("Dataset meta/info.json is missing the required robot_type.")
    if checkpoint_robot_type != metadata.robot_type:
        raise ValueError(
            f"robot_type mismatch: checkpoint={checkpoint_robot_type!r}, "
            f"dataset={metadata.robot_type!r}."
        )

    policy_cfg.device = args.device
    policy_cfg.pretrained_path = checkpoint
    action_source = action_source_for_config(policy_cfg, metadata.features)
    all_delta_timestamps = resolve_delta_timestamps(policy_cfg, metadata) or {}
    if action_source not in all_delta_timestamps:
        raise ValueError(f"Checkpoint did not define an action horizon for {action_source!r}.")
    action_delta_indices = list(policy_cfg.action_delta_indices or [])
    action_delta_timestamps = {action_source: all_delta_timestamps[action_source]}

    available_episodes = sorted(int(value) for value in metadata.episodes["episode_index"])
    episodes = parse_episode_spec(args.episodes, available_episodes)
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=action_delta_timestamps,
        return_uint8=True,
        use_video_keys=policy_cfg.decoded_video_keys(),
    )

    policy = make_policy(policy_cfg, ds_meta=metadata, for_training=False)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    unnormalizer = find_unnormalizer(postprocessor)

    prediction_step = (
        int(args.prediction_step)
        if args.prediction_step is not None
        else int(getattr(policy_cfg, "action_start_offset", 0))
    )
    chunk_size = len(action_delta_indices)
    if prediction_step < 0 or prediction_step >= chunk_size:
        raise ValueError(
            f"prediction_step={prediction_step} is outside checkpoint chunk [0, {chunk_size - 1}]."
        )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(checkpoint, dataset_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_names: list[str] | None = None
    robot_names: list[str] | None = None
    results: dict[int, EpisodeResult] = {}
    current_episode: int | None = None
    local_frame = 0

    LOGGER.info(
        "Evaluating episodes=%s action_mode=%s robot_type=%s prediction_step=%d",
        episodes,
        getattr(policy_cfg, "action_mode", None),
        checkpoint_robot_type,
        prediction_step,
    )

    progress = tqdm(range(len(dataset)), desc="Offline evaluation", unit="frame", dynamic_ncols=True)
    for dataset_index in progress:
        item = dataset[dataset_index]
        episode_index = _scalar_int(item["episode_index"])
        if episode_index != current_episode:
            current_episode = episode_index
            local_frame = 0
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            results[episode_index] = EpisodeResult(episode_index=episode_index)
            LOGGER.info("Episode %d", episode_index)
            progress.set_postfix(episode=episode_index, refresh=False)

        should_evaluate = local_frame % args.stride == 0
        raw_batch, padding = build_preprocessor_input(
            item,
            policy_cfg,
            metadata,
            action_source,
            include_ground_truth=should_evaluate,
        )
        processed = preprocessor(raw_batch)
        normalized_ground_truth = processed.pop(ACTION, None)
        model_input = prepare_predict_input(policy, processed)

        if should_evaluate:
            if normalized_ground_truth is None or padding is None:
                raise RuntimeError("Ground-truth action was lost during preprocessing.")
            with torch.inference_mode():
                normalized_prediction = policy.predict_action_chunk(model_input)
            if normalized_prediction.shape != normalized_ground_truth.shape:
                raise ValueError(
                    "Prediction/GT chunk mismatch: "
                    f"{tuple(normalized_prediction.shape)} vs {tuple(normalized_ground_truth.shape)}"
                )

            mode_prediction = _apply_action_step(unnormalizer, normalized_prediction)
            mode_ground_truth = _apply_action_step(unnormalizer, normalized_ground_truth)
            robot_prediction = postprocessor(normalized_prediction.clone())
            robot_ground_truth = postprocessor(normalized_ground_truth.clone())

            mode_prediction_np = mode_prediction.squeeze(0).detach().cpu().float().numpy()
            mode_ground_truth_np = mode_ground_truth.squeeze(0).detach().cpu().float().numpy()
            robot_prediction_np = robot_prediction.squeeze(0).detach().cpu().float().numpy()
            robot_ground_truth_np = robot_ground_truth.squeeze(0).detach().cpu().float().numpy()
            valid = ~padding.detach().cpu().numpy().astype(bool)

            if mode_names is None:
                mode_names = _dimension_names(
                    metadata.features[action_source], mode_prediction_np.shape[-1], "action"
                )
                robot_feature_key = (
                    ACTION
                    if policy_cfg.action_representation == "joint"
                    else ACTION_ABSOLUTE_EE
                )
                robot_feature = metadata.features.get(robot_feature_key, {})
                robot_names = _dimension_names(
                    robot_feature, robot_prediction_np.shape[-1], "robot_action"
                )

            result = results[episode_index]
            observation_frame = _scalar_int(item["frame_index"])
            target_frame = observation_frame + action_delta_indices[prediction_step]
            result.observation_frames.append(observation_frame)
            result.target_frames.append(target_frame)
            result.valid_chunks.append(valid)
            result.mode_predictions.append(mode_prediction_np)
            result.mode_ground_truth.append(mode_ground_truth_np)
            result.robot_predictions.append(robot_prediction_np)
            result.robot_ground_truth.append(robot_ground_truth_np)
        local_frame += 1

    if mode_names is None or robot_names is None:
        raise RuntimeError("No episode produced an evaluation sample.")

    metrics: dict[str, Any] = {"episodes": {}}
    all_mode_prediction: list[np.ndarray] = []
    all_mode_ground_truth: list[np.ndarray] = []
    all_robot_prediction: list[np.ndarray] = []
    all_robot_ground_truth: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []

    for episode_index, result in results.items():
        arrays = _episode_arrays(result)
        np.savez_compressed(output_dir / f"episode_{episode_index:06d}_predictions.npz", **arrays)

        mode_arrays = {
            "prediction_chunks": arrays["action_mode_prediction_chunks"],
            "ground_truth_chunks": arrays["action_mode_ground_truth_chunks"],
            "chunk_valid": arrays["chunk_valid"],
        }
        robot_arrays = {
            "prediction_chunks": arrays["robot_command_prediction_chunks"],
            "ground_truth_chunks": arrays["robot_command_ground_truth_chunks"],
            "chunk_valid": arrays["chunk_valid"],
        }
        metrics["episodes"][str(episode_index)] = {
            "evaluated_observations": int(arrays["observation_frames"].shape[0]),
            "valid_selected_targets": int(arrays["chunk_valid"][:, prediction_step].sum()),
            "action_mode_space": _space_metrics(mode_arrays, mode_names, prediction_step),
            "robot_command_space": _space_metrics(robot_arrays, robot_names, prediction_step),
        }

        selected_valid = arrays["chunk_valid"][:, prediction_step]
        if selected_valid.any():
            target_frames = arrays["target_frames"][selected_valid]
            plot_comparison(
                output_dir / f"episode_{episode_index:06d}_action_mode.png",
                target_frames,
                arrays["action_mode_prediction_chunks"][selected_valid, prediction_step],
                arrays["action_mode_ground_truth_chunks"][selected_valid, prediction_step],
                mode_names,
                f"Episode {episode_index} | action mode: {policy_cfg.action_mode}",
            )
            plot_comparison(
                output_dir / f"episode_{episode_index:06d}_robot_command.png",
                target_frames,
                arrays["robot_command_prediction_chunks"][selected_valid, prediction_step],
                arrays["robot_command_ground_truth_chunks"][selected_valid, prediction_step],
                robot_names,
                f"Episode {episode_index} | robot command space",
            )

        all_mode_prediction.append(arrays["action_mode_prediction_chunks"])
        all_mode_ground_truth.append(arrays["action_mode_ground_truth_chunks"])
        all_robot_prediction.append(arrays["robot_command_prediction_chunks"])
        all_robot_ground_truth.append(arrays["robot_command_ground_truth_chunks"])
        all_valid.append(arrays["chunk_valid"])

    combined_valid = np.concatenate(all_valid)
    metrics["overall"] = {
        "action_mode_space": _space_metrics(
            {
                "prediction_chunks": np.concatenate(all_mode_prediction),
                "ground_truth_chunks": np.concatenate(all_mode_ground_truth),
                "chunk_valid": combined_valid,
            },
            mode_names,
            prediction_step,
        ),
        "robot_command_space": _space_metrics(
            {
                "prediction_chunks": np.concatenate(all_robot_prediction),
                "ground_truth_chunks": np.concatenate(all_robot_ground_truth),
                "chunk_valid": combined_valid,
            },
            robot_names,
            prediction_step,
        ),
    }
    metrics["summary"] = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "dataset_repo_id": repo_id,
        "robot_type": checkpoint_robot_type,
        "policy_type": policy_cfg.type,
        "action_mode": policy_cfg.action_mode,
        "action_source": action_source,
        "action_mode_dimension_names": mode_names,
        "robot_command_dimension_names": robot_names,
        "episodes": episodes,
        "stride": args.stride,
        "prediction_step": prediction_step,
        "prediction_delta_frames": action_delta_indices[prediction_step],
        "seed": args.seed,
        "device": args.device,
        "inference_api": "predict_action_chunk",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved offline evaluation to %s", output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--episodes", default="all", help="all, comma-separated ids, or ranges (e.g. 0-3,7)")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every Nth frame; state still advances every frame",
    )
    parser.add_argument(
        "--prediction-step",
        type=int,
        default=None,
        help="Chunk index used for episode plots; defaults to action_start_offset",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output_dir = evaluate(args)
    print(output_dir)


if __name__ == "__main__":
    main()
