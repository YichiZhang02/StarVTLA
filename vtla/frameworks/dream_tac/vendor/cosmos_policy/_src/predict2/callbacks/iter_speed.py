# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import torch
import wandb
from torch import Tensor

from cosmos_policy._src.imaginaire.callbacks.every_n import EveryN
from cosmos_policy._src.imaginaire.model import ImaginaireModel
from cosmos_policy._src.imaginaire.trainer import ImaginaireTrainer
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.distributed import rank0_only
from cosmos_policy._src.imaginaire.utils.easy_io import easy_io

# Scalar loss components in output_batch (policy/cosmos model).
# - mse_loss / edm_loss: overall latent MSE and EDM-weighted loss (all frames, before mask).
# - demo_sample_*: only on demonstration (rollout_data_mask=0) samples.
#   - action_mse/l1: action prediction (main policy loss).
#   - future_proprio/image/wrist_image: future state prediction.
#   - value_mse/l1: value prediction (nan when no value targets).
# - world_model_sample_*: only on rollout world-model samples (future state).
# - value_function_sample_*: only on rollout value-function samples.
LOSS_KEYS_ORDER = [
    "mse_loss",
    "edm_loss",
    "demo_sample_action_mse_loss",
    "demo_sample_action_l1_loss",
    "demo_sample_future_proprio_mse_loss",
    "demo_sample_future_proprio_l1_loss",
    "demo_sample_future_wrist_image_mse_loss",
    "demo_sample_future_wrist_image_l1_loss",
    "demo_sample_future_image_mse_loss",
    "demo_sample_future_image_l1_loss",
    "demo_sample_future_tactile_left_mse_loss",
    "demo_sample_future_tactile_left_l1_loss",
    "demo_sample_future_tactile_right_mse_loss",
    "demo_sample_future_tactile_right_l1_loss",
    "demo_sample_value_mse_loss",
    "demo_sample_value_l1_loss",
    "world_model_sample_future_proprio_mse_loss",
    "world_model_sample_future_proprio_l1_loss",
    "world_model_sample_future_wrist_image_mse_loss",
    "world_model_sample_future_wrist_image_l1_loss",
    "world_model_sample_future_image_mse_loss",
    "world_model_sample_future_image_l1_loss",
    "world_model_sample_value_mse_loss",
    "world_model_sample_value_l1_loss",
    "value_function_sample_value_mse_loss",
    "value_function_sample_value_l1_loss",
]


def _format_loss_components(output_batch: dict) -> str:
    """Format scalar loss components from output_batch for logging."""
    parts = []
    for key in LOSS_KEYS_ORDER:
        if key not in output_batch:
            continue
        v = output_batch[key]
        if isinstance(v, Tensor) and v.numel() == 1:
            parts.append(f"{key}={v.item():.4f}")
    return " | ".join(parts) if parts else ""


def _loss_components_for_wandb(output_batch: dict) -> dict:
    """Build dict of scalar loss components for wandb.log; keys are train/loss_components/<name>."""
    out = {}
    for key in LOSS_KEYS_ORDER:
        if key not in output_batch:
            continue
        v = output_batch[key]
        if isinstance(v, Tensor) and v.numel() == 1:
            try:
                out[f"train/loss_components/{key}"] = v.item()
            except (ValueError, RuntimeError):
                pass  # skip nan/inf if needed
    return out


class IterSpeed(EveryN):
    """
    Args:
        hit_thres (int): Number of iterations to wait before logging.
        save_s3 (bool): Whether to save to S3.
        save_s3_every_log_n (int): Save to S3 every n log iterations, which means save_s3_every_log_n n * every_n global iterations.
    """

    def __init__(self, *args, hit_thres: int = 5, save_s3: bool = True, save_s3_every_log_n: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = None
        self.hit_counter = 0
        self.hit_thres = hit_thres
        self.save_s3 = save_s3
        self.save_s3_every_log_n = save_s3_every_log_n
        self.name = self.__class__.__name__
        self.last_hit_time = time.time()

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.hit_counter < self.hit_thres:
            loss_components = _format_loss_components(output_batch)
            log.info(
                f"Iteration {iteration}: "
                f"Hit counter: {self.hit_counter + 1}/{self.hit_thres} | "
                f"Loss: {loss.item():.4f} | "
                f"Time: {time.time() - self.last_hit_time:.2f}s"
            )
            if loss_components:
                log.info(f"  Loss components: {loss_components}")
            self.hit_counter += 1
            self.last_hit_time = time.time()
            #! useful for large scale training and avoid oom crash in the first two iterations!!!
            torch.cuda.synchronize()
            return
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    @rank0_only
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, Tensor],
        output_batch: dict[str, Tensor],
        loss: Tensor,
        iteration: int,
    ) -> None:
        if self.time is None:
            self.time = time.time()
            return
        cur_time = time.time()
        iter_speed = (cur_time - self.time) / self.every_n / self.step_size

        log.info(f"{iteration} : iter_speed {iter_speed:.2f} seconds per iteration | Loss: {loss.item():.4f}")
        loss_components = _format_loss_components(output_batch)
        if loss_components:
            log.info(f"  Loss components: {loss_components}")

        if wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            wandb_dict = {
                "timer/iter_speed": iter_speed,
                "sample_counter": sample_counter,
                "train/loss": loss.item(),
            }
            loss_component_dict = _loss_components_for_wandb(output_batch)
            if loss_component_dict:
                wandb_dict.update(loss_component_dict)
            wandb.log(wandb_dict, step=iteration)
        self.time = cur_time
        if self.save_s3:
            if iteration % (self.save_s3_every_log_n * self.every_n) == 0:
                easy_io.dump(
                    {
                        "iter_speed": iter_speed,
                        "iteration": iteration,
                    },
                    f"s3://rundir/{self.name}/iter_{iteration:09d}.yaml",
                )
