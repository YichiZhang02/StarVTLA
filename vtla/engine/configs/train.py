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
import builtins
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from vtla.engine.configs import parser
from vtla.engine.optim import LRSchedulerConfig, OptimizerConfig
from vtla.engine.utils.hub import HubMixin

from .default import DatasetConfig, PeftConfig, WandBConfig
from .policies import PreTrainedConfig

TRAIN_CONFIG_NAME = "train_config.json"


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    policy: PreTrainedConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Set to True to use deterministic cuDNN algorithms for reproducibility.
    # This disables cudnn.benchmark and may reduce training speed by ~10-20 percent.
    cudnn_deterministic: bool = False
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    steps: int = 100_000
    log_freq: int = 200
    tolerance_s: float = 1e-4
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    checkpoint_path: Path | None = field(init=False, default=None)

    @property
    def trainable_config(self) -> PreTrainedConfig:
        """Return the active policy config."""
        return self.policy  # type: ignore[return-value]

    def validate(self) -> None:
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")

        if policy_path:
            yaml_overrides = parser.get_yaml_overrides("policy")
            cli_overrides = parser.get_cli_overrides("policy") or []
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=yaml_overrides + cli_overrides
            )
            self.policy.pretrained_path = Path(policy_path)
        elif self.resume:
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )

            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )

            policy_dir = Path(config_path).parent
            if self.policy is not None:
                self.policy.pretrained_path = policy_dir
            self.checkpoint_path = policy_dir.parent

        if self.policy is None:
            raise ValueError("No policy is configured. Please specify one with `--policy.path`.")

        active_cfg = self.trainable_config
        if active_cfg.type == "starvla_groot_dinoalign":
            image_transforms = self.dataset.image_transforms
            color_temp = (
                tuple(image_transforms.color_temp)
                if image_transforms.color_temp is not None
                else None
            )
            if image_transforms.preset != "none" or color_temp != (0, 0):
                raise ValueError(
                    "starvla_groot_dinoalign requires augmentation_mode=none and "
                    "COLOR_TEMP_RANGE='[0,0]' so the DINO teacher receives an unaugmented "
                    "reference image. Got "
                    f"preset={image_transforms.preset!r}, color_temp={color_temp!r}."
                )
        if not self.job_name:
            self.job_name = f"{active_cfg.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = active_cfg.get_optimizer_preset()
            self.scheduler = active_cfg.get_scheduler_preset()

        if hasattr(active_cfg, "push_to_hub") and active_cfg.push_to_hub and not active_cfg.repo_id:
            raise ValueError("'repo_id' argument missing. Please specify it to push the model to the hub.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Keys for draccus pretrained-path loading."""
        return ["policy"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]  # because of the third-party library draccus uses Any as the return type

    def _save_pretrained(self, save_directory: Path) -> None:
        # 保存时把 cwd(仓库根) 下的绝对路径转相对, 使 train_config.json 跨机器可移植。
        # 只动 output_dir / dataset paths 这些已知会出现绝对路径的字段; dump 完恢复, 不影响内存中后续使用。
        cwd = Path.cwd()

        def _to_rel(v):
            if not v:
                return v, False
            p = Path(v)
            if not p.is_absolute():
                return v, False
            try:
                rel = p.relative_to(cwd)
            except ValueError:
                return v, False  # 不在仓库下 (如别的挂载点), 无法相对化, 保持原样
            return (rel if isinstance(v, Path) else str(rel)), True

        saved = {}
        new_out, changed = _to_rel(self.output_dir)
        if changed:
            saved["output_dir"] = self.output_dir
            self.output_dir = new_out
        ds = getattr(self, "dataset", None)
        if ds is not None and getattr(ds, "root", None):
            new_root, changed = _to_rel(ds.root)
            if changed:
                saved["dataset_root"] = ds.root
                ds.root = new_root
        if ds is not None and getattr(ds, "catalog_root", None):
            new_catalog_root, changed = _to_rel(ds.catalog_root)
            if changed:
                saved["dataset_catalog_root"] = ds.catalog_root
                ds.catalog_root = new_catalog_root
        if ds is not None and getattr(ds, "mixture_config", None):
            new_mixture_config, changed = _to_rel(ds.mixture_config)
            if changed:
                saved["dataset_mixture_config"] = ds.mixture_config
                ds.mixture_config = new_mixture_config
        try:
            with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
                draccus.dump(self, f, indent=4)
        finally:
            if "output_dir" in saved:
                self.output_dir = saved["output_dir"]
            if "dataset_root" in saved:
                self.dataset.root = saved["dataset_root"]
            if "dataset_catalog_root" in saved:
                self.dataset.catalog_root = saved["dataset_catalog_root"]
            if "dataset_mixture_config" in saved:
                self.dataset.mixture_config = saved["dataset_mixture_config"]

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=kwargs.pop("cli_args", []))
