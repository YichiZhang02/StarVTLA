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
from vtla.engine.utils.constants import OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from vtla.frameworks.ee_processor_utils import make_ee_relative_steps, remap_ee_dataset_stats
from vtla.frameworks.tactile_temporal_processor import TactileTemporalWindowStep

from .configuration_fastwam import FastWAMConfig


@ProcessorStepRegistry.register(name="fastwam_prepare_batch")
@dataclass
class FastWAMPrepareBatchStep(ProcessorStep):
    camera_keys: list[str]
    frame_indices: list[int]
    video_size: tuple[int, int]
    context_len: int
    text_dim: int
    prompt_template: str
    text_embedding_cache_dir: str | None = None
    text_embedding_cache_dirs: list[str] | None = None
    task_to_slot: dict[str, int] | None = None
    use_text_cache: bool = True
    camera_image_size: tuple[int, int] = (224, 224)
    use_proprio: bool = True

    def __post_init__(self) -> None:
        self.camera_image_size = tuple(int(size) for size in self.camera_image_size)
        self.video_size = (
            self.camera_image_size[0],
            self.camera_image_size[1] * len(self.camera_keys),
        )
        self.task_to_slot = dict(self.task_to_slot or {})
        self.text_embedding_cache_dirs = list(self.text_embedding_cache_dirs or [])
        self._text_context_state: dict[str, torch.Tensor] = {}
        if not self.use_text_cache:
            return
        # A saved processor carries the merged mapping and state file. Do not touch the original
        # training dataset paths before DataProcessorPipeline restores that state.
        if self.task_to_slot:
            return
        cache_dirs = [Path(path).expanduser() for path in self.text_embedding_cache_dirs]
        if self.text_embedding_cache_dir is not None:
            singular = Path(self.text_embedding_cache_dir).expanduser()
            if singular not in cache_dirs:
                cache_dirs.insert(0, singular)
        if not cache_dirs:
            return

        merged_state: dict[str, torch.Tensor] = {}
        task_sources: dict[str, Path] = {}
        reference_manifest = None
        for cache_dir in cache_dirs:
            manifest_path = cache_dir / "manifest.json"
            tensor_path = cache_dir / "embeddings.safetensors"
            if not manifest_path.is_file() or not tensor_path.is_file():
                raise FileNotFoundError(
                    f"FastWAM text cache requires {manifest_path} and {tensor_path}."
                )
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            self._validate_cache_manifest(manifest, cache_dir, reference_manifest)
            if reference_manifest is None:
                reference_manifest = manifest
            source_state = load_file(str(tensor_path), device="cpu")
            for entry in manifest["tasks"]:
                task = str(entry["task"])
                source_slot = int(entry["slot"])
                context = source_state[f"context.{source_slot}"]
                mask = source_state[f"mask.{source_slot}"]
                if task in self.task_to_slot:
                    slot = self.task_to_slot[task]
                    if not torch.equal(merged_state[f"context.{slot}"], context) or not torch.equal(
                        merged_state[f"mask.{slot}"], mask
                    ):
                        raise ValueError(
                            f"FastWAM task {task!r} has conflicting embeddings in "
                            f"{task_sources[task]} and {cache_dir}."
                        )
                    continue
                slot = len(self.task_to_slot)
                self.task_to_slot[task] = slot
                task_sources[task] = cache_dir
                merged_state[f"context.{slot}"] = context
                merged_state[f"mask.{slot}"] = mask
        self.load_state_dict(merged_state)

    def _validate_cache_manifest(
        self,
        manifest: dict[str, Any],
        cache_dir: Path,
        reference: dict[str, Any] | None,
    ) -> None:
        if manifest.get("world_model") != "wan22":
            raise ValueError(f"Expected a wan22 text cache in {cache_dir}, got {manifest.get('world_model')!r}.")
        if int(manifest.get("context_length", -1)) != self.context_len:
            raise ValueError(f"FastWAM text cache context length does not match policy config: {cache_dir}")
        if int(manifest.get("embedding_dim", -1)) != self.text_dim:
            raise ValueError(f"FastWAM text cache embedding dimension does not match policy config: {cache_dir}")
        if manifest.get("prompt_template") != self.prompt_template:
            raise ValueError(f"FastWAM text cache prompt template does not match policy config: {cache_dir}")
        if not isinstance(manifest.get("tasks"), list):
            raise ValueError(f"FastWAM text cache manifest has no tasks list: {cache_dir}")
        if reference is not None and manifest.get("text_encoder_model_hash") != reference.get(
            "text_encoder_model_hash"
        ):
            raise ValueError(f"FastWAM text caches use different text encoder model hashes: {cache_dir}")

    def _prepare_camera(self, image: torch.Tensor) -> torch.Tensor:
        image = torch.as_tensor(image)
        if image.ndim == 4:
            image = image.unsqueeze(1)
        if image.ndim != 5:
            raise ValueError(f"FastWAM camera tensor must be [B,T,C,H,W] or [B,C,H,W], got {image.shape}.")
        if image.shape[1] > 1:
            if max(self.frame_indices) >= image.shape[1]:
                raise ValueError(
                    f"FastWAM frame index {max(self.frame_indices)} exceeds input length {image.shape[1]}."
                )
            image = image[:, self.frame_indices]
        else:
            image = image.expand(-1, len(self.frame_indices), -1, -1, -1)
        original_dtype = image.dtype
        image = image.float()
        flat = image.flatten(0, 1)
        flat = F.interpolate(
            flat,
            size=self.camera_image_size,
            mode="bilinear",
            align_corners=False,
        )
        image = flat.unflatten(0, (image.shape[0], image.shape[1]))
        if original_dtype == torch.uint8:
            image = image / 127.5 - 1.0
        else:
            image = image * 2.0 - 1.0
        return image

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        output = transition.copy()
        observation = dict(output.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(output.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        cameras = [self._prepare_camera(observation[key]) for key in self.camera_keys]
        video = torch.cat(cameras, dim=-1).permute(0, 2, 1, 3, 4).contiguous()
        if tuple(video.shape[-2:]) != self.video_size:
            raise ValueError(f"FastWAM video shape mismatch: expected {self.video_size}, got {video.shape[-2:]}")

        state = observation.get(OBS_STATE)
        if self.use_proprio and state is None:
            raise ValueError(f"FastWAM requires {OBS_STATE}.")
        if state is not None:
            state = torch.as_tensor(state)
            if state.ndim == 2:
                state = state.unsqueeze(1)

        tasks = complementary.get("task")
        if isinstance(tasks, str):
            tasks = [tasks]
        if not isinstance(tasks, (list, tuple)):
            raise ValueError("FastWAM requires a task string for each batch item.")
        if len(tasks) != video.shape[0]:
            raise ValueError(f"FastWAM received {len(tasks)} tasks for batch size {video.shape[0]}.")
        context = complementary.get("context")
        context_mask = complementary.get("context_mask")
        if (context is None) != (context_mask is None):
            raise ValueError("FastWAM context and context_mask must be provided together.")
        if context is not None:
            context = torch.as_tensor(context)
            context_mask = torch.as_tensor(context_mask, dtype=torch.bool)
            if context.shape != (video.shape[0], self.context_len, self.text_dim):
                raise ValueError(f"Invalid FastWAM context shape: {tuple(context.shape)}")
            if context_mask.shape != (video.shape[0], self.context_len):
                raise ValueError(f"Invalid FastWAM context_mask shape: {tuple(context_mask.shape)}")
        elif self.use_text_cache and self._text_context_state:
            contexts = []
            masks = []
            for task in tasks:
                task = str(task)
                if task not in self.task_to_slot:
                    raise KeyError(
                        f"Task is absent from FastWAM wan22 text cache: {task!r}. "
                        f"Available tasks: {sorted(self.task_to_slot)}"
                    )
                slot = self.task_to_slot[task]
                contexts.append(self._text_context_state[f"context.{slot}"])
                masks.append(self._text_context_state[f"mask.{slot}"])
            context = torch.stack(contexts)
            context_mask = torch.stack(masks).bool()
            context = context.clone()
            context[~context_mask] = 0
            context_mask = torch.ones_like(context_mask)

        camera_pad_masks = []
        for camera_key in self.camera_keys:
            pad_mask = observation.get(f"{camera_key}_is_pad")
            if pad_mask is not None:
                pad_mask = torch.as_tensor(pad_mask, dtype=torch.bool)
                if pad_mask.ndim == 1:
                    pad_mask = pad_mask.unsqueeze(0)
                if pad_mask.shape[1] > 1:
                    pad_mask = pad_mask[:, self.frame_indices]
                else:
                    pad_mask = pad_mask.expand(-1, len(self.frame_indices))
                camera_pad_masks.append(pad_mask)
        if not camera_pad_masks:
            image_is_pad = torch.zeros(video.shape[0], len(self.frame_indices), dtype=torch.bool)
        else:
            image_is_pad = torch.stack(camera_pad_masks).any(dim=0)

        complementary.update({"video": video, "image_is_pad": image_is_pad})
        if state is not None:
            complementary["proprio"] = state
        if context is not None:
            complementary.update(
                {
                    "context": context,
                    "context_mask": context_mask,
                }
            )
        for camera_key in self.camera_keys:
            observation.pop(camera_key, None)
            observation.pop(f"{camera_key}_is_pad", None)
        observation.pop(OBS_STATE, None)
        output[TransitionKey.OBSERVATION] = observation
        output[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return output

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_keys": self.camera_keys,
            "frame_indices": self.frame_indices,
            "video_size": self.video_size,
            "camera_image_size": self.camera_image_size,
            "context_len": self.context_len,
            "text_dim": self.text_dim,
            "prompt_template": self.prompt_template,
            "text_embedding_cache_dir": self.text_embedding_cache_dir,
            "text_embedding_cache_dirs": self.text_embedding_cache_dirs,
            "task_to_slot": self.task_to_slot,
            "use_text_cache": self.use_text_cache,
            "use_proprio": self.use_proprio,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._text_context_state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if not self.use_text_cache:
            self._text_context_state = {}
            return
        expected_keys = {
            key
            for slot in self.task_to_slot.values()
            for key in (f"context.{slot}", f"mask.{slot}")
        }
        missing = sorted(expected_keys - set(state))
        if missing:
            raise ValueError(f"FastWAM text cache is missing tensors: {missing}")
        loaded = {key: value.detach().cpu() for key, value in state.items() if key in expected_keys}
        for task, slot in self.task_to_slot.items():
            context = loaded[f"context.{slot}"]
            mask = loaded[f"mask.{slot}"]
            if context.shape != (self.context_len, self.text_dim):
                raise ValueError(f"Invalid context shape for task {task!r}: {tuple(context.shape)}")
            if mask.shape != (self.context_len,):
                raise ValueError(f"Invalid mask shape for task {task!r}: {tuple(mask.shape)}")
        self._text_context_state = loaded


def make_fastwam_pre_post_processors(
    config: FastWAMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    dataset_stats = remap_ee_dataset_stats(dataset_stats, config)
    relative_step, absolute_step = make_ee_relative_steps(config)
    tactile_window_step = TactileTemporalWindowStep(
        tactile_keys=list(config.tactile_windowed_keys()),
        num_frames=config.tactile_num_frames,
        frame_offset=config.tactile_frame_offset,
    )
    features = {**config.normalizer_input_features(), **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        tactile_window_step,
        relative_step,
        NormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        FastWAMPrepareBatchStep(
            camera_keys=list(config.camera_keys),
            frame_indices=list(range(0, config.n_obs_steps, config.video_frame_stride)),
            video_size=config.video_size,
            camera_image_size=config.camera_image_size,
            context_len=config.context_len,
            text_dim=config.text_dim,
            prompt_template=config.prompt_template,
            text_embedding_cache_dir=(
                str(config.text_embedding_cache_dir) if config.text_embedding_cache_dir is not None else None
            ),
            text_embedding_cache_dirs=[str(path) for path in config.text_embedding_cache_dirs],
            use_text_cache=not config.load_text_encoder,
            use_proprio=config.state_mode != "none",
        ),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        absolute_step,
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline(steps=input_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
