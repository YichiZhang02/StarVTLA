#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Shared sensor-routing config knobs for all vtla policies.

Three knobs, unified across act / diffusion / pi05 / starvla_groot:

- ``wrist_only``: ``True`` uses only the wrist camera; ``False`` uses top + wrist.
- ``tactile_mode``: ``none`` (tactile unused) / ``as_image`` (tactile fed as
  extra image inputs) / ``encode`` (tactile through a dedicated encoder — reserved).
- ``state_mode``: ``none`` (no proprio state) / ``joint`` (joint angles) /
  ``ee`` (end-effector pose — reserved).

The bulk of the routing is feature selection performed at ``validate_features()``
time via the composable helpers below, so the model code only needs to consume
whatever VISUAL / STATE features survive. ``encode`` and ``ee`` are reserved and
raise ``NotImplementedError`` consistently across all policies.
"""

from dataclasses import dataclass, field

from vtla.engine.configs import FeatureType, PolicyFeature
from vtla.engine.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

VALID_TACTILE_MODES = ("none", "as_image", "encode")

# state_mode choices:
#   none           — no proprioceptive state fed to the model.
#   joint          — raw joint angles (default).
#   episode_rot6d  — EE pose relative to each episode's FIRST frame (T0^-1·Tt), rot6d rotation.
#   absolute_rot6d — EE pose in the robot base frame (Tt, no T0), rot6d rotation.
#   episode_quat   — same as episode_rot6d but rotation stored as quaternion [x,y,z,w].
#   absolute_quat  — same as absolute_rot6d but rotation stored as quaternion [x,y,z,w].
#
# Backward-compat aliases (still accepted, normalised to the canonical name on validation):
#   "episode_ee"  -> "episode_rot6d",  "absolute_ee" -> "absolute_rot6d"
VALID_STATE_MODES = (
    "none", "absolute_joint", "episode_joint",
    "episode_rot6d", "absolute_rot6d",
    "episode_quat",  "absolute_quat",
)
# Accepted for backward compat; map to canonical names in _normalise_ee_modes().
_STATE_MODE_ALIASES: dict[str, str] = {
    "joint":       "absolute_joint",
    "episode_ee":  "episode_rot6d",
    "absolute_ee": "absolute_rot6d",
}

# action_mode combines reference and representation, e.g. absolute_joint or relative_rot6d.
#
# Backward-compat alias: "relative_ee" -> "rot6d"
VALID_ACTION_MODES = (
    "absolute_joint", "relative_joint",
    "absolute_rot6d", "relative_rot6d",
    "absolute_quat", "relative_quat",
)
_ACTION_MODE_ALIASES: dict[str, str] = {
    "joint":       "absolute_joint",
    "rot6d":       "relative_rot6d",
    "quat":        "relative_quat",
    "relative_ee": "relative_rot6d",
}

VALID_TACTILE_ENCODERS = (None, "anytouch2", "native")
VALID_TACTILE_INSERT_LOCATIONS = ("encoder", "decoder")

# Dataset columns / stats keys added offline by tools/convert_joints_to_eepose.py.
# rot6d variants (original "_ee" suffix, kept for backward compat).
OBS_STATE_EPISODE_EE = OBS_STATE + "_episode_ee"    # observation.state_episode_ee  (rot6d)
OBS_STATE_EPISODE_JOINT = OBS_STATE + "_episode_joint"
ACTION_EPISODE_EE    = ACTION + "_episode_ee"        # action_episode_ee              (rot6d)
OBS_STATE_ABSOLUTE_EE = OBS_STATE + "_absolute_ee"  # observation.state_absolute_ee (rot6d)
ACTION_ABSOLUTE_EE   = ACTION + "_absolute_ee"       # action_absolute_ee             (rot6d)
ACTION_RELATIVE_EE   = ACTION + "_relative_ee"       # stats-only: St^-1·S_{t+k}     (rot6d)
# quat variants (new).
OBS_STATE_EPISODE_QUAT  = OBS_STATE + "_episode_quat"   # observation.state_episode_quat
ACTION_EPISODE_QUAT     = ACTION + "_episode_quat"       # action_episode_quat
OBS_STATE_ABSOLUTE_QUAT = OBS_STATE + "_absolute_quat"  # observation.state_absolute_quat
ACTION_ABSOLUTE_QUAT    = ACTION + "_absolute_quat"      # action_absolute_quat
ACTION_RELATIVE_QUAT    = ACTION + "_relative_quat"      # stats-only (quat)


@dataclass
class SensorRoutingMixin:
    """Mixin holding the shared sensor-routing fields + helper methods.

    Intended to be mixed in *before* ``PreTrainedConfig`` so its (all-defaulted)
    fields combine cleanly:

        @PreTrainedConfig.register_subclass("act")
        @dataclass
        class ACTConfig(SensorRoutingMixin, PreTrainedConfig): ...
    """

    # --- camera routing ---
    # top_camera_keys / wrist_camera_keys 均为列表，支持多路相机（如双臂 left/right wrist）。
    wrist_only: bool = False
    top_camera_keys: list[str] = field(
        default_factory=lambda: ["observation.images.cam_top"]
    )
    wrist_camera_keys: list[str] = field(
        default_factory=lambda: ["observation.images.cam_right_wrist"]
    )

    # --- tactile routing ---
    tactile_mode: str = "none"  # none | as_image | encode
    tactile_encoder_type: str | None = None  # None | anytouch2 | native (encode only)
    tactile_keys: list[str] = field(
        default_factory=lambda: [
            "observation.images.cam_finger0",
            "observation.images.cam_finger1",
        ]
    )
    # --- tactile encoder (tactile_mode="encode" only) ---
    # Path to a trained tactile-MAE checkpoint (.pth) or HF dir. The encoder arch /
    # sensor_id / image_size are read from the checkpoint automatically.
    tactile_encoder_path: str | None = None
    # Where the tactile tokens are injected, relative to each policy's
    # observation-encoder -> action-decoder structure:
    #   "encoder": tactile tokens enter the observation encoder with the other
    #              modalities (deep multimodal interaction).
    #   "decoder": tactile tokens are an extra condition queried by the action
    #              decoder only (does not pass through the VLM / obs encoder).
    # Ignored for Diffusion (no explicit encoder/decoder split) and only active
    # when tactile_mode="encode".
    tactile_insert_location: str = "decoder"  # encoder | decoder
    # Number of learnable query tokens emitted by the tactile-MAE encoder per tactile
    # image. Total tactile tokens = len(tactile_keys) * tactile_num_tokens.
    tactile_num_tokens: int = 8
    # --- tactile temporal window (independent of the RGB observation window) ---
    # Number of tactile frames fed to the policy per step, INCLUDING the current frame.
    # ``1`` (default) = current frame only = exact legacy behaviour. ``F>1`` stacks the
    # current frame plus ``F-1`` earlier frames along a leading time axis, giving the
    # policy short-horizon tactile history. Applies to both tactile_mode="as_image"
    # and "encode". Decoupled from the shared observation window / n_obs_steps.
    tactile_num_frames: int = 1
    # Spacing, in dataset frames, between two consecutive tactile frames. ``1`` = adjacent
    # frames; ``k`` = every k-th frame (wider temporal receptive field, same F frames).
    # The sampled tactile delta indices are ``[-(F-1)*offset, ..., -offset, 0]``.
    tactile_frame_offset: int = 1
    # By default the tactile-MAE encoder + query tokens are fine-tuned end-to-end with
    # the policy (the checkpoint is used as initialization). Set True to freeze the MAE
    # backbone and train only the query tokens + projection.
    freeze_tactile_encoder: bool = False

    # --- state / action routing ---
    # none | absolute_joint | episode_joint | episode_rot6d | absolute_rot6d |
    # episode_quat | absolute_quat
    # (aliases accepted: episode_ee→episode_rot6d, absolute_ee→absolute_rot6d)
    state_mode: str = "absolute_joint"
    # absolute_joint | relative_joint | absolute_rot6d | relative_rot6d |
    # absolute_quat | relative_quat
    action_mode: str = "absolute_joint"
    # Derived from the compound modes during validation. They are saved in checkpoints so tools
    # can inspect the data contract without parsing strings, but state_mode/action_mode remain the
    # user-facing CLI knobs.
    state_reference: str = "absolute"
    state_representation: str = "joint"
    action_reference: str = "absolute"
    action_representation: str = "joint"
    # Gripper commands remain absolute even when arm joints/poses are relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_feature_names: list[str] | None = None
    # Number of arms packed in the EE vectors.
    # rot6d: per arm = pos(3) + rot6d(6) + gripper(1) = 10 dims.
    # quat:  per arm = pos(3) + qx,qy,qz,qw(4) + gripper(1) = 8 dims.
    ee_num_arms: int = 2
    # Ordered names of the observation.state joints (populated by make_policy from ds_meta).
    # Required for EpisodeEEPreprocessorStep to locate joint/gripper indices at inference time.
    state_feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _dedupe(keys: list[str]) -> list[str]:
        seen, out = set(), []
        for k in keys:
            if k not in seen:
                out.append(k)
                seen.add(k)
        return out

    def selected_camera_keys(self) -> list[str]:
        """RGB cameras selected by ``wrist_only`` (top/wrist 各可多路)。"""
        keys = (
            list(self.wrist_camera_keys)
            if self.wrist_only
            else list(self.top_camera_keys) + list(self.wrist_camera_keys)
        )
        return self._dedupe(keys)

    def image_keys(self) -> list[str]:
        """All image keys fed to the model's vision path (cameras + tactile-as-image)."""
        keys = self.selected_camera_keys()
        if self.tactile_mode == "as_image":
            keys = keys + list(self.tactile_keys)
        return self._dedupe(keys)

    # Alias used by the VLM policies (pi05 / starvla_groot).
    def vlm_image_keys(self) -> list[str]:
        return self.image_keys()

    def tactile_encoder_keys(self) -> list[str]:
        """Tactile keys reserved for a dedicated encoder branch (``encode`` mode)."""
        return self._dedupe(list(self.tactile_keys)) if self.tactile_mode == "encode" else []

    def tactile_windowed(self) -> bool:
        """True when a multi-frame tactile history is requested (``tactile_num_frames > 1``)."""
        return self.tactile_mode in ("as_image", "encode") and self.tactile_num_frames > 1

    def tactile_delta_indices(self) -> list[int]:
        """Frame offsets sampled for each tactile key, oldest → current.

        ``[-(F-1)*offset, ..., -offset, 0]`` where ``F = tactile_num_frames`` and
        ``offset = tactile_frame_offset``. ``F == 1`` yields ``[0]`` (current frame only).
        These override the shared ``observation_delta_indices`` for tactile keys so the
        tactile temporal window is independent of the RGB observation window.
        """
        f = int(self.tactile_num_frames)
        off = int(self.tactile_frame_offset)
        return [-(f - 1 - i) * off for i in range(f)]

    def tactile_windowed_keys(self) -> list[str]:
        """Tactile image keys that receive the temporal window (both as_image and encode)."""
        if self.tactile_mode in ("as_image", "encode"):
            return self._dedupe(list(self.tactile_keys))
        return []

    def image_feature_keys_expanded(self) -> list[str]:
        """``image_features`` keys with windowed tactile-as-image keys expanded per frame.

        Preserves the ``image_features`` insertion order (what the vision models iterate),
        replacing each windowed tactile key with ``F`` per-frame keys ``<key>.f{i}``. When
        the tactile window is inactive (F == 1 or not as_image) this returns the plain
        ``image_features`` keys — i.e. exact legacy behaviour.
        """
        keys = list(self.image_features)
        if not (self.tactile_mode == "as_image" and self.tactile_num_frames > 1):
            return keys
        windowed = set(self.tactile_windowed_keys())
        f = int(self.tactile_num_frames)
        out: list[str] = []
        for key in keys:
            if key in windowed:
                out.extend(f"{key}.f{i}" for i in range(f))
            else:
                out.append(key)
        return out

    def decoded_video_keys(self) -> list[str]:
        """All camera videos this policy actually consumes (RGB + tactile-as-image + tactile-encode).

        Used to tell the dataset which video streams to decode, so unselected cameras (e.g. finger
        cams when ``tactile_mode='none'``) are not decoded every sample — a large data-loading win
        for fast models that would otherwise starve the GPU. Cameras not in this list are skipped.
        """
        return self._dedupe(self.image_keys() + self.tactile_encoder_keys())

    @property
    def image_features(self) -> dict:
        """RGB image features fed to the policy's vision backbone.

        Overrides ``PreTrainedConfig.image_features`` (the mixin precedes it in the
        MRO) to drop tactile-encoder keys in ``encode`` mode: those tactile images go
        through the dedicated tactile-MAE encoder, not the RGB vision path.
        """
        from vtla.engine.configs import FeatureType

        if not self.input_features:
            return {}
        tactile_keys = set(self.tactile_encoder_keys())
        return {
            key: ft
            for key, ft in self.input_features.items()
            if ft.type is FeatureType.VISUAL and key not in tactile_keys
        }

    def normalizer_input_features(self) -> dict:
        """``input_features`` for the dataset normalizer.

        In ``encode`` mode the tactile-encoder keys are dropped here so they are *not*
        normalized with dataset mean/std: the tactile-MAE encoder consumes raw [0, 1]
        images and applies its own (ImageNet) normalization internally.
        """
        feats = dict(self.input_features)
        if self.tactile_mode == "encode":
            for key in self.tactile_encoder_keys():
                feats.pop(key, None)
        return feats

    # ------------------------------------------------------------------
    # EE mode helpers
    # ------------------------------------------------------------------
    def _normalise_ee_modes(self) -> None:
        """Resolve backward-compat aliases for state_mode / action_mode (mutates in place)."""
        original_action_mode = self.action_mode
        self.state_mode = _STATE_MODE_ALIASES.get(self.state_mode, self.state_mode)
        self.action_mode = _ACTION_MODE_ALIASES.get(self.action_mode, self.action_mode)
        # Legacy PI05 configs represented relative joint action with a separate boolean.
        if original_action_mode == "joint" and getattr(self, "use_relative_actions", False):
            self.action_mode = "relative_joint"

        if self.state_mode == "none":
            self.state_reference, self.state_representation = "absolute", "none"
        elif "_" in self.state_mode:
            self.state_reference, self.state_representation = self.state_mode.split("_", 1)
        if "_" in self.action_mode:
            self.action_reference, self.action_representation = self.action_mode.split("_", 1)

    def ee_rot_mode(self) -> str:
        """Return the rotation format implied by ``state_mode`` / ``action_mode``.

        Returns ``"quat"`` when either mode uses quaternion EE; otherwise ``"rot6d"``.
        """
        if self.state_representation == "quat" or self.action_representation == "quat":
            return "quat"
        return "rot6d"

    def ee_per_arm_dim(self) -> int:
        """Return the packed dimension per arm based on the active EE rotation format."""
        from vtla.engine.utils.ee_transforms import per_arm_dim
        return per_arm_dim(self.ee_rot_mode())

    def ee_total_dim(self) -> int:
        """Return the total EE vector dimension (all arms)."""
        return self.ee_num_arms * self.ee_per_arm_dim()

    def is_ee_mode(self) -> bool:
        """True when any EE state or action mode is active."""
        return self.state_representation in ("rot6d", "quat") or self.action_representation in ("rot6d", "quat")

    # ------------------------------------------------------------------
    # Validation building blocks (call these from each config)
    # ------------------------------------------------------------------
    def validate_sensor_modes(self) -> None:
        """Enum validation + reserved-mode gating. Call from ``__post_init__``."""
        if self.tactile_mode not in VALID_TACTILE_MODES:
            raise ValueError(
                f"Invalid tactile_mode '{self.tactile_mode}'. Expected one of {VALID_TACTILE_MODES}."
            )
        # Normalise backward-compat aliases before checking.
        self._normalise_ee_modes()
        if self.state_mode not in VALID_STATE_MODES:
            raise ValueError(
                f"Invalid state_mode '{self.state_mode}'. "
                f"Expected one of {VALID_STATE_MODES} (aliases: {_STATE_MODE_ALIASES})."
            )
        if self.action_mode not in VALID_ACTION_MODES:
            raise ValueError(
                f"Invalid action_mode '{self.action_mode}'. "
                f"Expected one of {VALID_ACTION_MODES} (aliases: {_ACTION_MODE_ALIASES})."
            )
        # Action and model state are intentionally independent. Relative actions use a hidden raw
        # observation anchor, so state_mode='none' and mixed joint/EE representations are valid.
        if self.tactile_encoder_type not in VALID_TACTILE_ENCODERS:
            raise ValueError(
                f"Invalid tactile_encoder_type '{self.tactile_encoder_type}'. "
                f"Expected one of {VALID_TACTILE_ENCODERS}."
            )
        if self.tactile_insert_location not in VALID_TACTILE_INSERT_LOCATIONS:
            raise ValueError(
                f"Invalid tactile_insert_location '{self.tactile_insert_location}'. "
                f"Expected one of {VALID_TACTILE_INSERT_LOCATIONS}."
            )
        if self.tactile_mode == "encode" and not self.tactile_encoder_path:
            raise ValueError(
                "tactile_mode='encode' requires --policy.tactile_encoder_path to point at a "
                "trained tactile-MAE checkpoint (.pth) or HF directory."
            )
        if self.tactile_mode == "encode" and self.tactile_num_tokens < 1:
            raise ValueError(
                f"tactile_num_tokens must be >= 1, got {self.tactile_num_tokens}."
            )
        # Tactile temporal window (both as_image and encode).
        if self.tactile_num_frames < 1:
            raise ValueError(
                f"tactile_num_frames must be >= 1, got {self.tactile_num_frames}."
            )
        if self.tactile_frame_offset < 1:
            raise ValueError(
                f"tactile_frame_offset must be >= 1, got {self.tactile_frame_offset}."
            )
        if self.tactile_num_frames > 1 and self.tactile_mode == "none":
            raise ValueError(
                "tactile_num_frames > 1 requires tactile_mode in {'as_image', 'encode'} "
                f"(got tactile_mode='none')."
            )

    def require_visual_feature(self, key: str, purpose: str) -> None:
        if key not in self.input_features:
            available = [n for n, ft in self.input_features.items() if ft.type is FeatureType.VISUAL]
            raise ValueError(
                f"{type(self).__name__}: {purpose} key '{key}' is not present in input_features. "
                f"Available visual keys: {available}"
            )
        if self.input_features[key].type is not FeatureType.VISUAL:
            raise ValueError(
                f"{type(self).__name__}: {purpose} key '{key}' must be a visual feature, "
                f"got {self.input_features[key].type}."
            )

    def validate_routed_keys(self) -> None:
        """Check that selected cameras and (if used) tactile keys exist as VISUAL features."""
        for key in self.selected_camera_keys():
            self.require_visual_feature(key, "camera")
        if self.tactile_mode in ("as_image", "encode"):
            if not self.tactile_keys:
                raise ValueError(f"{type(self).__name__}: tactile_mode='{self.tactile_mode}' requires tactile_keys.")
            for key in self.tactile_keys:
                self.require_visual_feature(key, "tactile")

    def prune_unselected_visual_features(self, extra_keep: tuple[str, ...] = ()) -> None:
        """Drop VISUAL input features that are not part of the active routing.

        Keeps: selected cameras, tactile (as_image/encode), and any ``extra_keep``
        (e.g. empty-camera placeholders). Everything else VISUAL is removed so the
        model only sees the cameras the knobs selected.
        """
        keep = set(self.image_keys()) | set(self.tactile_encoder_keys()) | set(extra_keep)
        for key in list(self.input_features):
            ft = self.input_features[key]
            if ft.type is FeatureType.VISUAL and key not in keep:
                self.input_features.pop(key)

    def apply_state_mode(self, padded_state_dim: int | None = None) -> None:
        """Route the proprioceptive state according to ``state_mode``.

        The dataset carries the joint ``observation.state`` plus the EE variants
        ``observation.state_episode_ee`` / ``observation.state_absolute_ee`` (rot6d) and
        ``observation.state_episode_quat`` / ``observation.state_absolute_quat`` (quat).
        This method selects one as the canonical ``observation.state`` the model consumes
        and drops all unselected variants.

        - ``none``:          remove ``observation.state`` (all variants).
        - ``joint``:         keep joint ``observation.state``, drop the EE variants.
        - ``episode_rot6d``: use ``observation.state_episode_ee``  (rot6d, existing).
        - ``absolute_rot6d``:use ``observation.state_absolute_ee`` (rot6d, existing).
        - ``episode_quat``:  use ``observation.state_episode_quat``  (quat, new).
        - ``absolute_quat``: use ``observation.state_absolute_quat`` (quat, new).
        """
        # All EE-variant column keys (drop whichever are not selected).
        _all_ee_keys = (
            OBS_STATE_EPISODE_JOINT,
            OBS_STATE_EPISODE_EE, OBS_STATE_ABSOLUTE_EE,
            OBS_STATE_EPISODE_QUAT, OBS_STATE_ABSOLUTE_QUAT,
        )
        if self.state_mode == "none":
            self.input_features.pop(OBS_STATE, None)
            for k in _all_ee_keys:
                self.input_features.pop(k, None)
        elif self.state_mode == "absolute_joint":
            for k in _all_ee_keys:
                self.input_features.pop(k, None)
            if padded_state_dim is not None and OBS_STATE not in self.input_features:
                self.input_features[OBS_STATE] = PolicyFeature(
                    type=FeatureType.STATE, shape=(padded_state_dim,)
                )
        else:
            # EE mode: pick the canonical column for this (state_mode, rot_mode) pair.
            _ee_key_map = {
                "episode_joint": OBS_STATE_EPISODE_JOINT,
                "episode_rot6d":  OBS_STATE_EPISODE_EE,
                "absolute_rot6d": OBS_STATE_ABSOLUTE_EE,
                "episode_quat":   OBS_STATE_EPISODE_QUAT,
                "absolute_quat":  OBS_STATE_ABSOLUTE_QUAT,
            }
            ee_key = _ee_key_map[self.state_mode]
            for k in _all_ee_keys:
                if k != ee_key:
                    self.input_features.pop(k, None)
            ee_ft = self.input_features.pop(ee_key, None)
            if ee_ft is not None:
                # Dataset has the pre-computed column; rename it to canonical OBS_STATE.
                self.input_features.pop(OBS_STATE, None)
                self.input_features[OBS_STATE] = ee_ft
            elif OBS_STATE not in self.input_features:
                raise ValueError(
                    f"state_mode='{self.state_mode}' requires either '{ee_key}' in the dataset "
                    "(run tools/convert_joints_to_eepose.py for offline datasets) or "
                    f"'{OBS_STATE}' for real-time inference (EpisodeEEPreprocessorStep converts "
                    "joint angles to EE pose at runtime)."
                )
            # else: OBS_STATE present but EE column absent → inference mode.

    def apply_action_mode(self) -> None:
        """Route the action according to ``action_mode`` (mirrors :meth:`apply_state_mode`).

        The dataset carries the joint ``action`` plus the EE variants ``action_episode_ee`` /
        ``action_absolute_ee``/``action_absolute_quat``. This selects one as the canonical ``action``
        output and drops the unselected variants. The rotation format (rot6d/quat) is determined by
        ``action_mode``; the episode/absolute variant is determined by ``state_mode``.
        """
        if self.output_features is None:
            return

        # All EE action column keys (drop whatever is not selected).
        _all_action_ee = (
            ACTION_EPISODE_EE, ACTION_ABSOLUTE_EE,
            ACTION_EPISODE_QUAT, ACTION_ABSOLUTE_QUAT,
        )
        if self.action_representation == "joint":
            for k in _all_action_ee:
                self.output_features.pop(k, None)
        elif self.action_representation in ("rot6d", "quat"):
            # EE actions are always loaded in the robot base frame. Relative encoding, when
            # requested, is applied later against the hidden current-observation anchor.
            if self.action_representation == "rot6d":
                ee_key = ACTION_ABSOLUTE_EE
            else:
                ee_key = ACTION_ABSOLUTE_QUAT
            for k in _all_action_ee:
                if k != ee_key:
                    self.output_features.pop(k, None)
            ee_ft = self.output_features.pop(ee_key, None)
            if ee_ft is not None:
                self.output_features.pop(ACTION, None)
                self.output_features[ACTION] = ee_ft
            elif ACTION not in self.output_features:
                raise ValueError(
                    f"action_mode='{self.action_mode}' (state_mode='{self.state_mode}') requires "
                    f"'{ee_key}' in the dataset. Run tools/convert_joints_to_eepose.py first."
                )

    def add_empty_cameras(self, num: int, image_resolution: tuple[int, int]) -> list[str]:
        """Add ``num`` zero-padded placeholder cameras; return their keys."""
        keys = []
        for i in range(num):
            key = OBS_IMAGES + f".empty_camera_{i}"
            self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_resolution))
            keys.append(key)
        return keys
