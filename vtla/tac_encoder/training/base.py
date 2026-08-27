"""Protocol shared by the backbone-specific training recipes."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class StepOutput:
    loss: Tensor
    metrics: dict[str, Tensor]


class TrainingRecipe(ABC):
    model_id: str
    objective: str
    default_weight_decay: float = 0.05
    default_warmup_epochs: int = 1
    default_min_lr: float = 0.0

    @abstractmethod
    def build_model(self, args) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def step(self, model: nn.Module, images: Tensor, args) -> StepOutput:
        raise NotImplementedError

    def after_optimizer_step(self, model: nn.Module, step: int, total_steps: int) -> None:
        del model, step, total_steps

    def optimizer(self, model: nn.Module, args) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        weight_decay = (
            self.default_weight_decay if args.weight_decay is None else args.weight_decay
        )
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay, "WD_exclude": False},
                {"params": no_decay, "weight_decay": 0.0, "WD_exclude": True},
            ],
            lr=args.lr,
            betas=(0.9, 0.99),
        )

    def scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        args,
        steps_per_epoch: int,
    ) -> "WarmupCosineScheduler":
        return WarmupCosineScheduler(
            optimizer,
            total_steps=max(1, args.epochs * steps_per_epoch),
            warmup_steps=max(
                0,
                (self.default_warmup_epochs if args.warmup_epochs is None else args.warmup_epochs)
                * steps_per_epoch,
            ),
            start_lr=0.0,
            final_lr=self.default_min_lr if args.min_lr is None else args.min_lr,
        )

    @abstractmethod
    def encoder_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def trainer_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        raise NotImplementedError

    def restore_state(
        self,
        model: nn.Module,
        encoder_state: dict[str, Tensor],
        trainer_state: dict[str, Tensor],
    ) -> None:
        merged = {**encoder_state, **trainer_state}
        model.load_state_dict(merged, strict=True)

    @abstractmethod
    def save_visualization(
        self,
        model: nn.Module,
        dataset,
        indices: list[int],
        destination: Path,
        device: torch.device,
        args,
        autocast_dtype,
    ) -> None:
        raise NotImplementedError


def split_state_dict(
    model: nn.Module,
    encoder_prefixes: tuple[str, ...],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    encoder, trainer = {}, {}
    for key, value in model.state_dict().items():
        target = encoder if key.startswith(encoder_prefixes) else trainer
        target[key] = value
    return encoder, trainer


def require_fully_pretrained(model: nn.Module, report: dict[str, Any] | None) -> None:
    if report is None:
        raise ValueError("A pretrained checkpoint is required; from-scratch training is disabled")
    missing = []
    for key in report.get("missing_keys", []):
        try:
            value = model.get_parameter(key)
        except AttributeError:
            continue
        if value.requires_grad:
            missing.append(key)
    mismatched = list(report.get("shape_mismatch", {}))
    if missing or mismatched:
        raise ValueError(
            "Pretrained checkpoint does not cover every trainable parameter: "
            f"missing={missing[:12]}, shape_mismatch={mismatched[:12]}"
        )


class WarmupCosineScheduler:
    """Per-update warmup and cosine schedule used by the reference trainers."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        start_lr: float,
        final_lr: float,
        final_weight_decay: float | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.start_lr = float(start_lr)
        self.final_lr = float(final_lr)
        self.final_weight_decay = final_weight_decay
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.base_weight_decays = [float(group["weight_decay"]) for group in optimizer.param_groups]
        self.step_number = 0
        self._set_lrs(self._lrs_at(0))

    def _lrs_at(self, step: int) -> list[float]:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = step / self.warmup_steps
            return [self.start_lr + progress * (base - self.start_lr) for base in self.base_lrs]
        cosine_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / cosine_steps))
        return [
            self.final_lr
            + (base - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
            for base in self.base_lrs
        ]

    def _set_lrs(self, values: list[float]) -> None:
        for group, value in zip(self.optimizer.param_groups, values, strict=True):
            group["lr"] = value

    def step(self) -> None:
        self.step_number += 1
        self._set_lrs(self._lrs_at(self.step_number))
        if self.final_weight_decay is not None:
            progress = min(1.0, self.step_number / self.total_steps)
            for group, initial in zip(
                self.optimizer.param_groups, self.base_weight_decays, strict=True
            ):
                if group.get("WD_exclude", False):
                    continue
                group["weight_decay"] = self.final_weight_decay + (
                    initial - self.final_weight_decay
                ) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "start_lr": self.start_lr,
            "final_lr": self.final_lr,
            "final_weight_decay": self.final_weight_decay,
            "base_lrs": self.base_lrs,
            "base_weight_decays": self.base_weight_decays,
            "step_number": self.step_number,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in (
            "total_steps",
            "warmup_steps",
            "start_lr",
            "final_lr",
            "final_weight_decay",
            "base_lrs",
            "base_weight_decays",
            "step_number",
        ):
            setattr(self, key, state[key])
        self._set_lrs(self._lrs_at(self.step_number))
