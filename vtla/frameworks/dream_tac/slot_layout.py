from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DreamTacSlotSpec:
    index: int
    name: str
    phase: str
    modality: str
    source_key: str | None = None


@dataclass(frozen=True)
class DreamTacSlotLayout:
    version: int
    slots: tuple[DreamTacSlotSpec, ...]
    temporal_compression_factor: int

    @property
    def state_t(self) -> int:
        return len(self.slots)

    @property
    def pixel_frames(self) -> int:
        return 1 + (self.state_t - 1) * self.temporal_compression_factor

    @property
    def action_index(self) -> int:
        return self._single_index("action", "action")

    @property
    def current_state_index(self) -> int | None:
        return self._optional_single_index("current", "state")

    @property
    def future_state_index(self) -> int | None:
        return self._optional_single_index("future", "state")

    @property
    def num_conditional_frames(self) -> int:
        return self.action_index

    @property
    def rgb_keys(self) -> tuple[str, ...]:
        return tuple(
            slot.source_key
            for slot in self.slots
            if slot.phase == "current" and slot.modality == "rgb" and slot.source_key is not None
        )

    @property
    def tactile_keys(self) -> tuple[str, ...]:
        return tuple(
            slot.source_key
            for slot in self.slots
            if slot.phase == "current" and slot.modality == "tactile" and slot.source_key is not None
        )

    def indices(self, *, phase: str, modality: str) -> tuple[int, ...]:
        return tuple(
            slot.index
            for slot in self.slots
            if slot.phase == phase and slot.modality == modality
        )

    @property
    def current_sensor_indices(self) -> tuple[int, ...]:
        return tuple(
            slot.index
            for slot in self.slots
            if slot.phase == "current" and slot.modality in {"rgb", "tactile"}
        )

    @property
    def future_rgb_indices(self) -> tuple[int, ...]:
        return self.indices(phase="future", modality="rgb")

    @property
    def future_tactile_indices(self) -> tuple[int, ...]:
        return self.indices(phase="future", modality="tactile")

    @property
    def tactile_indices(self) -> tuple[int, ...]:
        return self.indices(phase="current", modality="tactile") + self.future_tactile_indices

    @property
    def future_prediction_indices(self) -> tuple[int, ...]:
        indices = [self.action_index]
        if self.future_state_index is not None:
            indices.append(self.future_state_index)
        indices.extend(self.future_rgb_indices)
        indices.extend(self.future_tactile_indices)
        return tuple(indices)

    @property
    def fingerprint(self) -> str:
        payload = {
            "version": self.version,
            "temporal_compression_factor": self.temporal_compression_factor,
            "slots": [asdict(slot) for slot in self.slots],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def records(self) -> list[dict[str, str | int | None]]:
        return [asdict(slot) for slot in self.slots]

    def _single_index(self, phase: str, modality: str) -> int:
        indices = self.indices(phase=phase, modality=modality)
        if len(indices) != 1:
            raise ValueError(f"Expected one {phase}/{modality} slot, got {indices}.")
        return indices[0]

    def _optional_single_index(self, phase: str, modality: str) -> int | None:
        indices = self.indices(phase=phase, modality=modality)
        if len(indices) > 1:
            raise ValueError(f"Expected at most one {phase}/{modality} slot, got {indices}.")
        return indices[0] if indices else None


def compile_slot_layout(
    *,
    wrist_only: bool,
    wrist_camera_keys: list[str],
    top_camera_keys: list[str],
    tactile_mode: str,
    tactile_keys: list[str],
    state_mode: str,
    temporal_compression_factor: int = 4,
    version: int = 1,
) -> DreamTacSlotLayout:
    if temporal_compression_factor <= 0:
        raise ValueError("Dream-Tac temporal_compression_factor must be positive.")
    if tactile_mode == "encode":
        raise ValueError("Dream-Tac does not support tactile_mode='encode'; use 'none' or 'as_image'.")
    if tactile_mode not in {"none", "as_image"}:
        raise ValueError(f"Unsupported Dream-Tac tactile_mode={tactile_mode!r}.")

    def dedupe(keys: list[str]) -> list[str]:
        return list(dict.fromkeys(str(key) for key in keys if str(key)))

    wrist_keys = dedupe(wrist_camera_keys)
    top_keys = [] if wrist_only else dedupe(top_camera_keys)
    rgb_keys = dedupe(wrist_keys + top_keys)
    tactile_sensor_keys = dedupe(tactile_keys) if tactile_mode == "as_image" else []
    if not rgb_keys:
        raise ValueError("Dream-Tac requires at least one RGB camera slot.")
    overlap = sorted(set(rgb_keys) & set(tactile_sensor_keys))
    if overlap:
        raise ValueError(f"Dream-Tac RGB and tactile keys overlap: {overlap}.")

    slots: list[DreamTacSlotSpec] = []

    def append(name: str, phase: str, modality: str, source_key: str | None = None) -> None:
        slots.append(DreamTacSlotSpec(len(slots), name, phase, modality, source_key))

    append("blank", "special", "blank")
    use_state = state_mode != "none"
    if use_state:
        append("current_proprio", "current", "state")
    for key in rgb_keys:
        append(f"current_rgb:{key}", "current", "rgb", key)
    for key in tactile_sensor_keys:
        append(f"current_tactile:{key}", "current", "tactile", key)
    append("action_chunk", "action", "action")
    if use_state:
        append("future_proprio", "future", "state")
    for key in rgb_keys:
        append(f"future_rgb:{key}", "future", "rgb", key)
    for key in tactile_sensor_keys:
        append(f"future_tactile:{key}", "future", "tactile", key)

    return DreamTacSlotLayout(
        version=version,
        slots=tuple(slots),
        temporal_compression_factor=temporal_compression_factor,
    )


__all__ = ["DreamTacSlotLayout", "DreamTacSlotSpec", "compile_slot_layout"]
