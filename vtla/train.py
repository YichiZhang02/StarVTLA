#!/usr/bin/env python

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
"""Train a VTLA policy with supervised fine-tuning."""

import dataclasses
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from vtla.datasets import EpisodeAwareSampler, make_dataset
from vtla.engine.common.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from vtla.engine.common.wandb_utils import WandBLogger
from vtla.engine.configs import parser
from vtla.engine.configs.train import TrainPipelineConfig
from vtla.engine.optim.factory import make_optimizer_and_scheduler
from vtla.engine.utils.import_utils import register_third_party_plugins, require_package
from vtla.engine.utils.logging_utils import AverageMeter, MetricsTracker
from vtla.engine.utils.random_utils import set_seed
from vtla.engine.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)
from vtla.engine.processor.relative_action_processor import route_ee_batch
from vtla.frameworks.factory import make_policy, make_pre_post_processors
from vtla.frameworks.pretrained import PreTrainedPolicy


def _scalar_metric(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item()) if value.numel() == 1 else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _component_losses(loss: torch.Tensor, output_dict: dict | None) -> tuple[float, float | None]:
    """Normalize policy-specific loss dictionaries for console training metrics."""
    outputs = output_dict or {}
    action_loss = next(
        (
            value
            for key in ("loss_action", "action_loss")
            if (value := _scalar_metric(outputs.get(key))) is not None
        ),
        float(loss.detach().item()),
    )
    video_loss = next(
        (
            value
            for key in ("loss_video", "video_loss")
            if (value := _scalar_metric(outputs.get(key))) is not None
        ),
        None,
    )
    return action_loss, video_loss


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    action_loss, video_loss = _component_losses(loss, output_dict)
    train_metrics.action_loss = action_loss
    if video_loss is not None:
        if "video_loss" not in train_metrics.metrics:
            action_meter = train_metrics.metrics["action_loss"]
            remaining_meters = {
                key: meter
                for key, meter in train_metrics.metrics.items()
                if key != "action_loss"
            }
            train_metrics.metrics = {
                "action_loss": action_meter,
                "video_loss": AverageMeter("video_loss", ":.3f"),
                **remaining_meters,
            }
        train_metrics.video_loss = video_loss
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """
    Main function to train a policy.

    This function orchestrates the supervised fine-tuning pipeline: logging, seeding,
    dataset/policy/optimizer setup, checkpoint resumption, training updates, metrics,
    checkpointing, and optional Hub upload.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    require_package("accelerate", extra="training")
    from accelerate import Accelerator

    cfg.validate()
    if cfg.policy is None:
        raise ValueError("vtla.train requires a policy config for SFT training.")

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active policy config's device is set to CPU.
        force_cpu = cfg.trainable_config.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    if is_main_process:
        logging.info("Creating policy")
    active_cfg = cfg.trainable_config
    if getattr(active_cfg, "type", None) == "fastwam":
        if active_cfg.load_text_encoder and is_main_process:
            logging.warning(
                "Ignoring load_text_encoder=True for FastWAM training; training always uses cached text contexts."
            )
        active_cfg.load_text_encoder = False
        if not cfg.resume:
            active_cfg.text_embedding_cache_dir = Path(dataset.root) / "text_embeddings" / "wan22"
            required_text_assets = [
                active_cfg.text_embedding_cache_dir / "manifest.json",
                active_cfg.text_embedding_cache_dir / "embeddings.safetensors",
            ]
            missing_text_assets = [path for path in required_text_assets if not path.is_file()]
            if missing_text_assets:
                raise FileNotFoundError(
                    "Missing dataset-local Wan2.2 text embeddings: "
                    f"{missing_text_assets}. Run `python tools/precompute_world_model_text_embeddings.py "
                    f"--dataset-root {dataset.root} --world-model wan22`."
                )
            if is_main_process:
                logging.info("Using dataset-local FastWAM text cache: %s", active_cfg.text_embedding_cache_dir)
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
        for_training=True,
    )

    # 把训练集任务文字写进 policy config, 使 checkpoint 自包含 (inference --match-policy 直接
    # 从 config.json 读 single_task, 不再依赖训练数据集是否还在)。仅单任务数据集写入。
    try:
        _tasks = list(dataset.meta.tasks.index)
        if len(_tasks) == 1:
            policy.config.single_task = str(_tasks[0])
            if is_main_process:
                logging.info(f"记录 single_task 到 checkpoint: {policy.config.single_task!r}")
        elif is_main_process:
            logging.info(f"数据集含 {len(_tasks)} 个任务, 不写单一 single_task (inference 需手动指定)")
    except Exception as _e:
        if is_main_process:
            logging.warning(f"未能记录 single_task 到 checkpoint: {_e}")

    if cfg.peft is not None:
        from peft import PeftModel

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish model creation before continuing
    accelerator.wait_for_everyone()

    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path
    # EE modes (episode_ee / absolute_ee / relative_ee), joint relative actions, and FastWAM
    # dataset-local text contexts change the
    # processor's feature dims and steps, so the pretrained/checkpoint processor must NOT be reused —
    # rebuild it from the current policy config (with the EE stats remap + pose relative/absolute
    # steps). This applies when resuming too: loading the saved processor would additionally prepend
    # the inference-only EpisodeEEPreprocessorStep (joints->EE FK, needs the robot SDK), which is
    # wrong for training where the dataset already supplies the EE columns. Model weights / optimizer
    # / scheduler / step are restored from the checkpoint independently of the processor.
    _ee_state_modes = ("episode_rot6d", "absolute_rot6d", "episode_quat", "absolute_quat",
                       "episode_ee", "absolute_ee")  # include legacy aliases
    _ee_action_modes = (
        "absolute_rot6d", "relative_rot6d", "absolute_quat", "relative_quat",
        "rot6d", "quat", "relative_ee",
    )
    _needs_rebuilt_processor = (
        getattr(active_cfg, "action_reference", "absolute") == "relative"
        or getattr(active_cfg, "state_mode", "absolute_joint") in _ee_state_modes
        or getattr(active_cfg, "action_mode", "absolute_joint") in _ee_action_modes
        or (getattr(active_cfg, "type", None) == "fastwam" and not cfg.resume)
    )
    if _needs_rebuilt_processor and processor_pretrained_path is not None:
        logging.warning(
            "Building processors from the current policy config instead of the pretrained processor "
            "because the active data contract requires dataset-specific state/action/text processing."
        )
        # The rebuilt-from-scratch processor would otherwise fall back to the HF hub tokenizer name
        # (the local path carried by the pretrained processor json is discarded here). For pi05, pull
        # the tokenizer straight from the pretrained model dir so the saved checkpoint stays offline-usable.
        if getattr(active_cfg, "type", None) == "pi05":
            _tokenizer_dir = processor_pretrained_path / "paligemma-3b-pt-224-tokenizer"
            if _tokenizer_dir.is_dir():
                active_cfg.paligemma_tokenizer_path = str(_tokenizer_dir)
            else:
                logging.warning(
                    f"Local paligemma tokenizer not found at {_tokenizer_dir}; the rebuilt processor "
                    "will fall back to the 'google/paligemma-3b-pt-224' HF hub name."
                )
        processor_pretrained_path = None

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if processor_pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        if getattr(active_cfg, "type", None) == "pi05":
            processor_kwargs["preprocessor_overrides"][
                "pi05_prepare_state_tokenizer_processor_step"
            ] = {
                "state_mode": getattr(active_cfg, "state_mode", "absolute_joint"),
                "max_state_dim": getattr(active_cfg, "max_state_dim", 32),
            }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=processor_pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(active_cfg, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=active_cfg.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()
    fastwam_visualization_samples = []

    train_metrics = {
        "action_loss": AverageMeter("action_loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        # Select joint vs EE columns as the canonical observation.state / action before the processor.
        batch = route_ee_batch(
            batch,
            getattr(active_cfg, "state_mode", "absolute_joint"),
            getattr(active_cfg, "action_mode", "absolute_joint"),
        )
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        if (
            getattr(active_cfg, "type", None) == "fastwam"
            and active_cfg.visualization_enabled
            and len(fastwam_visualization_samples) < active_cfg.visualization_num_samples
        ):
            from vtla.frameworks.fastwam.visualization import capture_samples

            fastwam_visualization_samples.extend(
                capture_samples(
                    batch,
                    tactile_keys=active_cfg.tactile_windowed_keys(),
                    num_samples=(
                        active_cfg.visualization_num_samples
                        - len(fastwam_visualization_samples)
                    ),
                )
            )
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
            progress_metrics = {
                "action_loss": f"{train_tracker.action_loss.val:.3f}",
            }
            if output_dict and "dino_alignment_loss" in output_dict:
                progress_metrics["dino_loss"] = f"{output_dict['dino_alignment_loss']:.3f}"
            if "video_loss" in train_tracker.metrics:
                progress_metrics["video_loss"] = f"{train_tracker.video_loss.val:.3f}"
            progress_metrics["lr"] = f"{train_tracker.lr.val:.1e}"
            progbar.set_postfix(progress_metrics)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_fastwam_visualization_step = (
            getattr(active_cfg, "type", None) == "fastwam"
            and active_cfg.visualization_enabled
            and len(fastwam_visualization_samples) >= active_cfg.visualization_num_samples
            and step % active_cfg.visualization_freq == 0
        )

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if is_fastwam_visualization_step:
            accelerator.wait_for_everyone()
            if is_main_process:
                if not fastwam_visualization_samples:
                    logging.warning("FastWAM visualization skipped because no sample was captured.")
                else:
                    try:
                        unwrapped_policy = accelerator.unwrap_model(policy)
                        visualization_summary = unwrapped_policy.generate_training_visualizations(
                            fastwam_visualization_samples,
                            output_dir=cfg.output_dir,
                            step=step,
                        )
                        for sample_metrics in visualization_summary["samples"]:
                            logging.info(
                                "FastWAM visualization sample %d saved to %s",
                                sample_metrics["sample_index"],
                                sample_metrics["image_path"],
                            )
                        logging.info(
                            "FastWAM visualization metrics saved to %s",
                            visualization_summary["metrics_path"],
                        )
                        if wandb_logger:
                            wandb_logger.log_dict(
                                visualization_summary["aggregate"], step=step, mode="eval"
                            )
                    except Exception:
                        logging.exception(
                            "FastWAM visualization failed at step %d; training will continue.",
                            step,
                        )
            accelerator.wait_for_everyone()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    save_training_state_dir=False,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if is_main_process:
        logging.info("End of training")

        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model)
            else:
                unwrapped_model.push_model_to_hub(cfg)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
