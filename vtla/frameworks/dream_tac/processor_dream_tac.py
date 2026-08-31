from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from vtla.engine.configs import PipelineFeatureType, PolicyFeature
from vtla.engine.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from vtla.engine.types import EnvTransition, TransitionKey
from vtla.engine.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from vtla.frameworks.ee_processor_utils import make_ee_relative_steps, remap_ee_dataset_stats

from .configuration_dream_tac import DreamTacConfig
from .slot_layout import DreamTacSlotLayout, DreamTacSlotSpec


def _as_batch_time(image: torch.Tensor) -> torch.Tensor:
    image = torch.as_tensor(image)
    if image.ndim == 4:
        image = image.unsqueeze(1)
    if image.ndim != 5 or image.shape[2] != 3:
        raise ValueError(f"Dream-Tac image must be [B,T,3,H,W] or [B,3,H,W], got {image.shape}.")
    return image


def _resize_uint8(image: torch.Tensor, size: int) -> torch.Tensor:
    original_dtype = image.dtype
    flat = image.float().flatten(0, 1)
    flat = F.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False)
    image = flat.unflatten(0, (image.shape[0], image.shape[1]))
    if original_dtype != torch.uint8:
        image = image.clamp(0, 1).mul(255)
    return image.round().to(torch.uint8)


def tactile_self_attention_gate(
    streams: list[torch.Tensor], batch_size: int, device: torch.device | None = None
) -> torch.Tensor:
    """Aggregate the original contact-change gate over any number of tactile streams."""
    diffs = []
    for stream in streams:
        stream = _as_batch_time(stream)
        if stream.shape[1] < 2:
            diffs.append(torch.zeros(batch_size, device=stream.device))
            continue
        diff = (stream[:, 1].float() - stream[:, 0].float()).abs().mean(dim=(1, 2, 3))
        if stream.dtype == torch.uint8:
            diff = diff / 255.0
        diffs.append(diff)
    if diffs:
        raw = torch.stack(diffs).amax(dim=0)
    else:
        raw = torch.zeros(batch_size, device=device)
    z = (4.0 * (raw - 0.002) / (0.001 + 1e-6)).clamp(-30, 30)
    return 0.15 + 0.85 * torch.sigmoid(z)


@ProcessorStepRegistry.register(name="dream_tac_prepare_batch")
@dataclass
class DreamTacPrepareBatchStep(ProcessorStep):
    rgb_keys: list[str]
    tactile_keys: list[str]
    layout_version: int
    layout_records: list[dict[str, Any]]
    layout_fingerprint: str
    temporal_compression_factor: int
    state_dim: int | None
    action_dim: int
    image_size: int
    context_len: int
    text_dim: int
    prompt_template: str
    text_embedding_cache_dir: str | None = None
    text_embedding_cache_dirs: list[str] | None = None
    task_to_slot: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.layout = DreamTacSlotLayout(
            version=self.layout_version,
            slots=tuple(DreamTacSlotSpec(**record) for record in self.layout_records),
            temporal_compression_factor=self.temporal_compression_factor,
        )
        if self.layout.fingerprint != self.layout_fingerprint:
            raise ValueError("Dream-Tac processor slot layout fingerprint is invalid.")
        self.task_to_slot = dict(self.task_to_slot or {})
        self.text_embedding_cache_dirs = list(self.text_embedding_cache_dirs or [])
        self._state: dict[str, torch.Tensor] = {}
        if self.task_to_slot:
            return
        cache_dirs = [Path(path).expanduser() for path in self.text_embedding_cache_dirs]
        if self.text_embedding_cache_dir is not None:
            singular = Path(self.text_embedding_cache_dir).expanduser()
            if singular not in cache_dirs:
                cache_dirs.insert(0, singular)
        if not cache_dirs:
            return
        reference_hash = None
        for cache_dir in cache_dirs:
            manifest_path = cache_dir / "manifest.json"
            embedding_path = cache_dir / "embeddings.safetensors"
            missing = [path for path in (manifest_path, embedding_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Dream-Tac cache is incomplete in {cache_dir}: {missing}")
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            reference_hash = self._validate_manifest(manifest, cache_dir, reference_hash)
            source = load_file(str(embedding_path), device="cpu")
            for entry in manifest["tasks"]:
                task = str(entry["task"])
                source_slot = int(entry["slot"])
                context = source[f"context.{source_slot}"].detach().cpu()
                mask = source[f"mask.{source_slot}"].detach().cpu().bool()
                if task in self.task_to_slot:
                    target_slot = self.task_to_slot[task]
                    if not torch.equal(self._state[f"context.{target_slot}"], context):
                        raise ValueError(f"Dream-Tac task {task!r} has conflicting text embeddings.")
                    continue
                target_slot = len(self.task_to_slot)
                self.task_to_slot[task] = target_slot
                self._state[f"context.{target_slot}"] = context
                self._state[f"mask.{target_slot}"] = mask
        self.load_state_dict(self._state)

    def _validate_manifest(self, manifest: dict[str, Any], cache_dir: Path, reference_hash: str | None) -> str:
        if manifest.get("world_model") != "dream_tac":
            raise ValueError(f"Expected a dream_tac text cache in {cache_dir}.")
        if int(manifest.get("context_length", -1)) != self.context_len:
            raise ValueError(f"Dream-Tac context length mismatch in {cache_dir}.")
        if int(manifest.get("embedding_dim", -1)) != self.text_dim:
            raise ValueError(f"Dream-Tac embedding dimension mismatch in {cache_dir}.")
        if manifest.get("prompt_template") != self.prompt_template:
            raise ValueError(f"Dream-Tac prompt template mismatch in {cache_dir}.")
        if not isinstance(manifest.get("tasks"), list):
            raise ValueError(f"Dream-Tac cache has no task list: {cache_dir}.")
        model_hash = str(manifest.get("text_encoder_model_hash", ""))
        if reference_hash is not None and model_hash != reference_hash:
            raise ValueError("Dream-Tac caches were produced by different T5 checkpoints.")
        return model_hash

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        output = transition.copy()
        observation = dict(output.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(output.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        rgb = {
            key: _resize_uint8(_as_batch_time(observation[key]), self.image_size)
            for key in self.rgb_keys
        }
        batch_size = next(iter(rgb.values())).shape[0]
        tactile_raw = {key: _as_batch_time(observation[key]) for key in self.tactile_keys}
        tactile = {
            key: _resize_uint8(stream, self.image_size)
            for key, stream in tactile_raw.items()
        }
        gate = tactile_self_attention_gate(
            list(tactile_raw.values()), batch_size, device=next(iter(rgb.values())).device
        )

        tasks = complementary.get("task")
        if isinstance(tasks, str):
            tasks = [tasks]
        if not isinstance(tasks, (list, tuple)) or len(tasks) != batch_size:
            raise ValueError("Dream-Tac requires one task string per batch item.")

        if self.layout.current_state_index is None:
            current_state = future_state = torch.empty(batch_size, 0)
        else:
            state = torch.as_tensor(observation[OBS_STATE]).float()
            if state.ndim == 2:
                state = state.unsqueeze(1)
            if state.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Dream-Tac expected state dim {self.state_dim}, got {tuple(state.shape)}."
                )
            current_state = state[:, 0]
            future_state = state[:, -1]

        actions = output.get(TransitionKey.ACTION)
        normalized_actions = None
        if actions is not None:
            normalized_actions = torch.as_tensor(actions).float()
            if normalized_actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Dream-Tac expected action dim {self.action_dim}, "
                    f"got {tuple(normalized_actions.shape)}."
                )

        blank = torch.zeros_like(next(iter(rgb.values()))[:, 0])
        slot_images = []
        for slot in self.layout.slots:
            if slot.modality in {"blank", "state", "action"}:
                image = blank
            elif slot.modality == "rgb":
                frames = rgb[str(slot.source_key)]
                image = frames[:, 0] if slot.phase == "current" else frames[:, -1]
            elif slot.modality == "tactile":
                frames = tactile[str(slot.source_key)]
                if slot.phase == "current":
                    image = frames[:, 1] if frames.shape[1] >= 3 else frames[:, -1]
                else:
                    image = frames[:, -1]
            else:
                raise ValueError(f"Unknown Dream-Tac slot modality: {slot.modality!r}.")
            slot_images.append(image)
        frames = [slot_images[0].unsqueeze(1)]
        frames.extend(
            image.unsqueeze(1).expand(-1, self.temporal_compression_factor, -1, -1, -1)
            for image in slot_images[1:]
        )
        video = torch.cat(frames, dim=1).permute(0, 2, 1, 3, 4).contiguous()
        if video.shape[2] != self.layout.pixel_frames:
            raise RuntimeError(
                f"Dream-Tac sequence must contain {self.layout.pixel_frames} frames, got {video.shape}."
            )

        contexts = []
        for task in tasks:
            if str(task) not in self.task_to_slot:
                raise KeyError(f"Task is absent from Dream-Tac text cache: {task!r}.")
            slot = self.task_to_slot[str(task)]
            contexts.append(self._state[f"context.{slot}"])

        current_rgb = self.layout.indices(phase="current", modality="rgb")
        future_rgb = self.layout.future_rgb_indices
        current_tactile = self.layout.indices(phase="current", modality="tactile")
        future_tactile = self.layout.future_tactile_indices

        def item(indices: tuple[int, ...], offset: int, default: int = -1) -> int:
            return indices[offset] if len(indices) > offset else default

        index_names = {
            "current_proprio_latent_idx": (
                self.layout.current_state_index
                if self.layout.current_state_index is not None
                else -1
            ),
            "current_wrist_image_latent_idx": item(current_rgb, 0),
            "current_image_latent_idx": item(current_rgb, 1),
            "action_latent_idx": self.layout.action_index,
            "future_proprio_latent_idx": (
                self.layout.future_state_index
                if self.layout.future_state_index is not None
                else -1
            ),
            "future_wrist_image_latent_idx": item(future_rgb, 0),
            "future_image_latent_idx": item(future_rgb, 1),
            "future_tactile_left_latent_idx": item(future_tactile, 0),
            "future_tactile_right_latent_idx": item(future_tactile, 1),
        }
        indices = {
            name: torch.full((batch_size,), index, dtype=torch.long)
            for name, index in index_names.items()
        }
        prepared = {
            "dataset_name": "video_data",
            "video": video,
            "t5_text_embeddings": torch.stack(contexts),
            # Original Dream-Tac datasets pass an all-one T5 mask after zero-padding embeddings.
            "t5_text_mask": torch.ones(batch_size, self.context_len, dtype=torch.long),
            "fps": torch.full((batch_size,), 16, dtype=torch.float32),
            "padding_mask": torch.zeros(batch_size, 1, self.image_size, self.image_size),
            "image_size": torch.full((batch_size, 4), self.image_size, dtype=torch.float32),
            "num_conditional_frames": self.layout.num_conditional_frames,
            "proprio": current_state,
            "future_proprio": future_state,
            "tactile_self_attn_gate": gate,
            "rollout_data_mask": torch.zeros(batch_size, dtype=torch.long),
            "world_model_sample_mask": torch.zeros(batch_size, dtype=torch.long),
            "value_function_sample_mask": torch.zeros(batch_size, dtype=torch.long),
            "value_function_return": torch.full((batch_size,), -100.0),
            "value_latent_idx": torch.full((batch_size,), -1, dtype=torch.long),
            "task": list(tasks),
            "slot_layout_fingerprint": self.layout_fingerprint,
            "current_sensor_latent_indices": torch.tensor(
                self.layout.current_sensor_indices, dtype=torch.long
            ).expand(batch_size, -1),
            "future_rgb_latent_indices": torch.tensor(
                future_rgb, dtype=torch.long
            ).expand(batch_size, -1),
            "future_tactile_latent_indices": torch.tensor(
                future_tactile, dtype=torch.long
            ).expand(batch_size, -1),
            "tactile_latent_indices": torch.tensor(
                current_tactile + future_tactile, dtype=torch.long
            ).expand(batch_size, -1),
            **indices,
        }
        if normalized_actions is not None:
            prepared["actions"] = normalized_actions
            prepared[ACTION] = normalized_actions
        output[TransitionKey.OBSERVATION] = {}
        complementary.update(prepared)
        output[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return output

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "rgb_keys": self.rgb_keys,
            "tactile_keys": self.tactile_keys,
            "layout_version": self.layout_version,
            "layout_records": self.layout_records,
            "layout_fingerprint": self.layout_fingerprint,
            "temporal_compression_factor": self.temporal_compression_factor,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "image_size": self.image_size,
            "context_len": self.context_len,
            "text_dim": self.text_dim,
            "prompt_template": self.prompt_template,
            "text_embedding_cache_dir": self.text_embedding_cache_dir,
            "text_embedding_cache_dirs": self.text_embedding_cache_dirs,
            "task_to_slot": self.task_to_slot,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected_text = {key for slot in self.task_to_slot.values() for key in (f"context.{slot}", f"mask.{slot}")}
        missing = sorted(expected_text - set(state))
        if missing:
            raise ValueError(f"Dream-Tac processor state is missing tensors: {missing}")
        self._state = {
            key: value.detach().cpu()
            for key, value in state.items()
            if key in expected_text
        }
        for task, slot in self.task_to_slot.items():
            if self._state[f"context.{slot}"].shape != (self.context_len, self.text_dim):
                raise ValueError(f"Invalid Dream-Tac context shape for {task!r}.")


def make_dream_tac_pre_post_processors(
    config: DreamTacConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    dataset_stats = remap_ee_dataset_stats(dataset_stats, config)
    relative_step, absolute_step = make_ee_relative_steps(config)
    features = {**config.normalizer_input_features(), **config.output_features}
    layout = config.slot_layout()
    prepare = DreamTacPrepareBatchStep(
        rgb_keys=list(layout.rgb_keys),
        tactile_keys=config.resolved_tactile_keys(),
        layout_version=layout.version,
        layout_records=layout.records(),
        layout_fingerprint=layout.fingerprint,
        temporal_compression_factor=config.temporal_compression_factor,
        state_dim=(
            int(config.robot_state_feature.shape[0])
            if config.robot_state_feature is not None
            else None
        ),
        action_dim=int(config.action_feature.shape[0]),
        image_size=config.image_size,
        context_len=config.context_len,
        text_dim=config.text_dim,
        prompt_template=config.prompt_template,
        text_embedding_cache_dir=str(config.text_embedding_cache_dir) if config.text_embedding_cache_dir else None,
        text_embedding_cache_dirs=[str(path) for path in config.text_embedding_cache_dirs],
    )
    return (
        PolicyProcessorPipeline(
            steps=[
                RenameObservationsProcessorStep(rename_map={}),
                AddBatchDimensionProcessorStep(),
                relative_step,
                NormalizerProcessorStep(
                    features=features,
                    norm_map=config.normalization_mapping,
                    stats=dataset_stats,
                ),
                prepare,
                DeviceProcessorStep(device=config.device),
            ],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=[
                UnnormalizerProcessorStep(
                    features=config.output_features,
                    norm_map=config.normalization_mapping,
                    stats=dataset_stats,
                ),
                absolute_step,
                DeviceProcessorStep(device="cpu"),
            ],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
