"""Shared configuration constants for tactile backbone training."""

from __future__ import annotations

from dataclasses import asdict, dataclass


SUPPORTED_MODEL_IDS = ("anytouch1", "anytouch2", "sparsh_vjepa")


@dataclass(frozen=True)
class TactileDataConfig:
    image_size: int = 224
    num_frames: int = 4
    frame_stride: int = 2
    contact_enter_threshold: float = 8.0
    contact_exit_threshold: float = 6.0
    contact_debounce_frames: int = 3
    contact_smoothing_frames: int = 5
    contact_top_fraction: float = 0.01
    anchor_contact_policy: str = "any"

    @property
    def frame_offsets(self) -> tuple[int, ...]:
        return tuple(
            (index - self.num_frames + 1) * self.frame_stride
            for index in range(self.num_frames)
        )

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.num_frames <= 0 or self.frame_stride <= 0:
            raise ValueError("num_frames and frame_stride must be positive")
        if not 0 < self.contact_top_fraction <= 1:
            raise ValueError("contact_top_fraction must be in (0, 1]")
        if self.contact_exit_threshold >= self.contact_enter_threshold:
            raise ValueError("contact_exit_threshold must be smaller than contact_enter_threshold")
        if self.contact_debounce_frames <= 0:
            raise ValueError("contact_debounce_frames must be positive")
        if self.contact_smoothing_frames <= 0 or self.contact_smoothing_frames % 2 == 0:
            raise ValueError("contact_smoothing_frames must be a positive odd integer")
        if self.anchor_contact_policy not in {"any", "all"}:
            raise ValueError("anchor_contact_policy must be 'any' or 'all'")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["frame_offsets"] = list(self.frame_offsets)
        return result
