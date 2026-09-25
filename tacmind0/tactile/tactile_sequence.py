"""Sequence-form tactile encoder for LeWM.

The historical tactile images are kept as independent frames rather than
being laid out as a raster mosaic.  Each frame is resized to ``42 x 42`` and
embedded into a ``3 x 3`` patch grid with a ViT patch size of 14.  Tokens are
ordered as ``L0, R0, L1, R1, ...``; positions carry time and cell-local
coordinates, while finger identity is a learned categorical embedding.

The causal path uses the existing LeWM Q/K RoPE implementation and SDPA.  A
stream state retains the most recent eight slots (144 patch keys/values per
layer) and computes only the current 18 patches on each update.  Like any
sliding KV cache over a deep transformer, this is a bounded approximate
stream: exact equivalence to recomputing a shifted window would require
re-encoding the retained frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .mosaic_posemb import (
    SPIRAL_DIRECTIONS,
    MosaicCausalRoPESelfAttention,
    MosaicRoPESelfAttention,
    _encoder_config,
    _resolve_vit_embeddings,
    _vit_hidden_size,
)

FINGERS = 2
DEFAULT_HISTORY_SIZE = 8
DEFAULT_HISTORY_STRIDE = 5
DEFAULT_FRAME_SIZE = 42
DEFAULT_PATCH_SIZE = 14
PATCHES_PER_FRAME = 9


def parse_tactile_layout(value: str | None) -> str:
    """Normalize the tactile input representation name."""
    key = "sequence_3x3" if value is None else str(value).strip().lower()
    aliases = {
        "sequence": "sequence_3x3",
        "sequence_3x3": "sequence_3x3",
        "history": "sequence_3x3",
        "mosaic": "mosaic",
        "4x4_mosaic": "mosaic",
    }
    if key not in aliases:
        raise ValueError(
            "tactile_layout must be mosaic or sequence_3x3, "
            f"got {value!r}"
        )
    return aliases[key]


def parse_tactile_rope(value: str | None) -> str:
    """Normalize the sequence position mode."""
    key = "canonical_spiral" if value is None else str(value).strip().lower()
    aliases = {
        "spiral": "spiral",
        "spatiotemporal_spiral": "spiral",
        "canonical": "canonical_spiral",
        "canonical_spiral": "canonical_spiral",
        "canonical_spiral_rope": "canonical_spiral",
    }
    if key not in aliases:
        raise ValueError(
            "tactile_rope must be spiral or canonical_spiral, "
            f"got {value!r}"
        )
    return aliases[key]


def sequence_patch_coordinates(
    *,
    frame_size: int = DEFAULT_FRAME_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    history_size: int = DEFAULT_HISTORY_SIZE,
    time_stride: int = DEFAULT_HISTORY_STRIDE,
    canonicalize_right: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return coordinates, finger ids, and slot ids for sequence tokens.

    The first coordinate is a zero coordinate for CLS.  Patch tokens are
    slot-major and finger-major, with nine row-major local patches per finger.
    The time coordinate uses the raw-frame stride, so adjacent sampled frames
    are five time units apart in the default configuration.
    """
    frame = int(frame_size)
    patch = int(patch_size)
    history = int(history_size)
    stride = int(time_stride)
    if frame <= 0 or patch <= 0 or frame % patch:
        raise ValueError(
            f"frame_size must be divisible by patch_size, got {frame} and {patch}"
        )
    grid = frame // patch
    if grid != 3:
        raise ValueError(
            f"sequence_3x3 requires frame_size/patch_size=3, got {grid}"
        )
    if history <= 0 or stride <= 0:
        raise ValueError("history_size and time_stride must be positive")

    coordinates: list[list[int]] = [[0, 0, 0]]
    finger_ids: list[int] = []
    slot_ids: list[int] = []
    for slot in range(history):
        for finger in range(FINGERS):
            for local_y in range(grid):
                for local_x in range(grid):
                    canonical_x = grid - 1 - local_x if finger and canonicalize_right else local_x
                    coordinates.append([slot * stride, local_y, canonical_x])
                    finger_ids.append(finger)
                    slot_ids.append(slot)
    return (
        torch.tensor(coordinates, dtype=torch.long),
        torch.tensor(finger_ids, dtype=torch.long),
        torch.tensor(slot_ids, dtype=torch.long),
    )


class MosaicToTactileSequence:
    """Split a BGR 4x4 tactile mosaic into slot-major 3x3-frame inputs.

    Input can be one HWC/CHW image or a batch of HWC/CHW images.  The output
    for one image is ``(2, 8, 3, 42, 42)`` and is deliberately BGR in
    ``[0, 1]`` to match the existing marker-mosaic preprocessing contract.
    The source mosaic already contains the eight stride-five history frames;
    this transform does not sample the raw stream a second time.

    A prepared dataset may store the de-interleaved cells directly as
    ``(2, 8, 3, 42, 42)`` (or with a leading observation-window dimension).
    That fast path only casts and scales the stored uint8 values; it avoids
    repeating the expensive CPU resize in every DataLoader worker.
    """

    def __init__(
        self,
        frame_size: int = DEFAULT_FRAME_SIZE,
        history_size: int = DEFAULT_HISTORY_SIZE,
        history_stride: int = DEFAULT_HISTORY_STRIDE,
    ) -> None:
        self.frame_size = int(frame_size)
        self.history_size = int(history_size)
        self.history_stride = int(history_stride)
        if self.history_size != DEFAULT_HISTORY_SIZE:
            raise ValueError("the 4x4 TacBench mosaic currently contains exactly 8 slots")
        if self.frame_size != DEFAULT_FRAME_SIZE:
            raise ValueError("sequence_3x3 currently requires 42x42 tactile frames")
        if self.history_stride <= 0:
            raise ValueError("history_stride must be positive")

    @staticmethod
    def _as_batch(image: Any) -> tuple[torch.Tensor, bool]:
        tensor = torch.as_tensor(image)
        if tensor.ndim == 3:
            if tensor.shape[0] == 3:
                return tensor.unsqueeze(0), True
            if tensor.shape[-1] == 3:
                return tensor.permute(2, 0, 1).unsqueeze(0), True
        elif tensor.ndim == 4:
            if tensor.shape[1] == 3:
                return tensor, False
            if tensor.shape[-1] == 3:
                return tensor.permute(0, 3, 1, 2), False
        raise ValueError(f"expected HWC/CHW image, got {tuple(tensor.shape)}")

    def _prepared(self, image: Any) -> torch.Tensor | None:
        """Return a prepared sequence tensor, or ``None`` for raw mosaics."""
        tensor = torch.as_tensor(image)
        expected = (FINGERS, self.history_size, 3, self.frame_size, self.frame_size)
        if tensor.ndim not in (5, 6) or tuple(tensor.shape[-5:]) != expected:
            return None
        values = tensor.float()
        if tensor.dtype == torch.uint8 or float(values.max()) > 1.0:
            values = values / 255.0
        return values

    def __call__(self, image: Any) -> torch.Tensor:
        prepared = self._prepared(image)
        if prepared is not None:
            return prepared
        tensor, squeeze = self._as_batch(image)
        if tensor.shape[-2] != tensor.shape[-1]:
            raise ValueError("tactile mosaic must be square")
        source_size = int(tensor.shape[-1])
        if source_size % 4:
            raise ValueError(f"mosaic size must be divisible by 4, got {source_size}")
        source_cell = source_size // 4
        values = tensor.float()
        if tensor.dtype == torch.uint8 or float(values.max()) > 1.0:
            values = values / 255.0
        cells = values.reshape(
            values.shape[0], 3, 4, source_cell, 4, source_cell
        ).permute(0, 2, 4, 1, 3, 5)
        left = cells[:, :, :2].reshape(values.shape[0], 8, 3, source_cell, source_cell)
        right = cells[:, :, 2:].reshape(values.shape[0], 8, 3, source_cell, source_cell)
        slot_major = torch.stack((left, right), dim=2)
        flat = slot_major.reshape(-1, 3, source_cell, source_cell)
        resized = F.interpolate(
            flat,
            size=(self.frame_size, self.frame_size),
            mode="bilinear",
            align_corners=False,
        )
        output = resized.reshape(
            values.shape[0], self.history_size, FINGERS, 3, self.frame_size, self.frame_size
        ).permute(0, 2, 1, 3, 4, 5)
        return output.squeeze(0) if squeeze else output


@dataclass
class TactileSequenceStreamState:
    """Bounded approximate streaming state for the sequence encoder."""

    patch_keys: list[torch.Tensor]
    patch_values: list[torch.Tensor]
    cls_state: torch.Tensor
    next_time: int
    history_size: int
    patches_per_slot: int


class TactileSequenceEncoder(nn.Module):
    """ViT encoder for independent tactile history frames."""

    def __init__(
        self,
        vit: nn.Module,
        *,
        frame_size: int = DEFAULT_FRAME_SIZE,
        patch_size: int = DEFAULT_PATCH_SIZE,
        history_size: int = DEFAULT_HISTORY_SIZE,
        time_stride: int = DEFAULT_HISTORY_STRIDE,
        canonicalize_right: bool = True,
        causal_attention: bool = True,
        rope_kind: str = "spatiotemporal_spiral",
        spiral_directions: int = SPIRAL_DIRECTIONS,
    ) -> None:
        super().__init__()
        if rope_kind != "spatiotemporal_spiral":
            raise ValueError(f"unsupported sequence rope_kind: {rope_kind}")
        embedding_owner, embeddings = _resolve_vit_embeddings(vit)
        # Prefer the actual embedding tensor over a config hint.  Some vendor
        # ViT configs carry ``num_register_tokens`` for compatibility even
        # when the instantiated HF embeddings contain only a CLS token.
        register_tokens = getattr(embeddings, "register_tokens", None)
        registers = (
            int(register_tokens.size(1))
            if torch.is_tensor(register_tokens) and register_tokens.ndim == 3
            else 0
        )
        hidden = _vit_hidden_size(embedding_owner)
        coordinates, finger_ids, slot_ids = sequence_patch_coordinates(
            frame_size=frame_size,
            patch_size=patch_size,
            history_size=history_size,
            time_stride=time_stride,
            canonicalize_right=canonicalize_right,
        )
        patch_count = int(finger_ids.numel())
        prefix_length = 1 + registers
        coordinates = torch.cat(
            (
                torch.zeros((prefix_length, 3), dtype=torch.long),
                coordinates[1:],
            ),
            dim=0,
        )
        causal_mask = torch.ones(
            coordinates.size(0), coordinates.size(0), dtype=torch.bool
        )
        if causal_attention:
            causal_mask.zero_()
            causal_mask[:prefix_length, :] = True
            causal_mask[prefix_length:, prefix_length:] = (
                slot_ids.unsqueeze(0) <= slot_ids.unsqueeze(1)
            )
        self._embedding_owner_is_inner = embedding_owner is not vit
        self.vit = vit
        self.frame_size = int(frame_size)
        self.patch_size = int(patch_size)
        self.history_size = int(history_size)
        self.time_stride = int(time_stride)
        self.prefix_length = prefix_length
        self.patches_per_slot = FINGERS * (int(frame_size) // int(patch_size)) ** 2
        self.patch_count = patch_count
        self.canonicalize_right = bool(canonicalize_right)
        self.rope_kind = rope_kind
        self.spiral_directions = int(spiral_directions)
        self.register_buffer("coordinates", coordinates, persistent=False)
        self.register_buffer("finger_ids", finger_ids, persistent=False)
        self.register_buffer("slot_ids", slot_ids, persistent=False)
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.finger_emb = nn.Embedding(FINGERS, hidden)
        self.finger_gate = nn.Parameter(torch.tensor([0.1]))
        nn.init.zeros_(self.finger_emb.weight)

        layers = getattr(getattr(embedding_owner, "encoder", None), "layer", None)
        if layers is None or not len(layers):
            raise TypeError("sequence tactile attention requires HF ViT encoder.layer")
        for layer in layers:
            outer_attention = getattr(layer, "attention", None)
            attention = getattr(outer_attention, "attention", None)
            if attention is None:
                raise TypeError("sequence tactile attention requires layer attention.attention")
            if causal_attention:
                outer_attention.attention = MosaicCausalRoPESelfAttention(
                    attention,
                    coordinates,
                    causal_mask,
                    prefix_length=prefix_length,
                    rope_kind=rope_kind,
                    spiral_directions=spiral_directions,
                )
            else:
                outer_attention.attention = MosaicRoPESelfAttention(
                    attention,
                    coordinates,
                    rope_kind=rope_kind,
                    spiral_directions=spiral_directions,
                )

    def freeze_unused_embedding_parameters(self) -> None:
        """Freeze native embedding parameters bypassed by the sequence path.

        The sequence path constructs its own CLS/register-plus-patch sequence
        and intentionally does not use DINO's absolute position table or mask
        token.  They must not be left trainable, otherwise DDP reports them as
        unused parameters and an optimizer would allocate pointless state.
        """
        embeddings = self._owner.embeddings
        for name in ("position_embeddings", "mask_token"):
            value = getattr(embeddings, name, None)
            if isinstance(value, nn.Parameter):
                value.requires_grad_(False)

    @property
    def config(self) -> Any:
        return _encoder_config(self._owner)

    @property
    def _owner(self) -> nn.Module:
        return getattr(self.vit, "model") if self._embedding_owner_is_inner else self.vit

    def _patch_tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 6 or pixels.size(1) != FINGERS or pixels.size(2) != self.history_size:
            raise ValueError(
                "sequence pixels must be (B, 2, 8, 3, 42, 42), "
                f"got {tuple(pixels.shape)}"
            )
        batch, _, _, channels, height, width = pixels.shape
        if channels != 3 or height != self.frame_size or width != self.frame_size:
            raise ValueError(
                f"sequence frames must be 3x{self.frame_size}x{self.frame_size}, "
                f"got {channels}x{height}x{width}"
            )
        flat = pixels.permute(0, 2, 1, 3, 4, 5).reshape(
            batch * self.history_size * FINGERS, channels, height, width
        )
        # The LeWM DINOv2 patch-embedding module is a plain Conv2d wrapper
        # and does not accept HuggingFace's ``interpolate_pos_encoding``
        # keyword.  Sequence frames are exactly 42x42 with a 14px patch, so
        # there is no interpolation step to request here.
        patch_tokens = self._owner.embeddings.patch_embeddings(flat)
        patch_tokens = patch_tokens.reshape(
            batch, self.history_size, FINGERS, patch_tokens.size(1), patch_tokens.size(2)
        ).reshape(batch, self.patch_count, patch_tokens.size(2))
        finger_ids = self.finger_ids.to(device=patch_tokens.device)
        return patch_tokens + self.finger_gate * self.finger_emb(finger_ids).unsqueeze(0).to(
            dtype=patch_tokens.dtype
        )

    def _embeddings(self, pixels: torch.Tensor) -> torch.Tensor:
        patch_tokens = self._patch_tokens(pixels)
        embeddings = self._owner.embeddings
        cls = embeddings.cls_token.expand(patch_tokens.size(0), -1, -1)
        prefix = [cls]
        register_tokens = getattr(embeddings, "register_tokens", None)
        if torch.is_tensor(register_tokens):
            prefix.append(register_tokens.expand(patch_tokens.size(0), -1, -1))
        sequence = torch.cat((*prefix, patch_tokens), dim=1)
        dropout = getattr(embeddings, "dropout", None)
        return dropout(sequence) if dropout is not None else sequence

    def forward(self, pixel_values: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if pixel_values.ndim == 5 and pixel_values.size(1) == self.patch_count:
            raise ValueError(
                "sequence encoder expects structured (B,2,8,3,42,42) pixels; "
                "the training transform must not flatten tactile history"
            )
        embedding_output = self._embeddings(pixel_values)
        encoder_output = self._owner.encoder(embedding_output)
        sequence = getattr(encoder_output, "last_hidden_state", None)
        if sequence is None:
            sequence = encoder_output[0]
        sequence = self._owner.layernorm(sequence)
        return SimpleNamespace(last_hidden_state=sequence)

    def _attention_layers(self) -> list[MosaicCausalRoPESelfAttention]:
        return [layer.attention.attention for layer in self._owner.encoder.layer]

    @torch.no_grad()
    def stream_init(
        self, pixels: torch.Tensor
    ) -> tuple[torch.Tensor, TactileSequenceStreamState]:
        """Bootstrap the cache from one complete eight-slot sequence."""
        embedding_output = self._embeddings(pixels)
        for attention in self._attention_layers():
            attention.last_patch_keys = None
            attention.last_patch_values = None
            attention.capture_stream_state = True
        try:
            encoder_output = self._owner.encoder(embedding_output)
        finally:
            for attention in self._attention_layers():
                attention.capture_stream_state = False
        sequence = getattr(encoder_output, "last_hidden_state", None)
        if sequence is None:
            sequence = encoder_output[0]
        output_hidden = self._owner.layernorm(sequence)
        keys = [attention.last_patch_keys for attention in self._attention_layers()]
        values = [attention.last_patch_values for attention in self._attention_layers()]
        if any(key is None for key in keys) or any(value is None for value in values):
            raise RuntimeError("sequence stream initialization did not capture KV")
        state = TactileSequenceStreamState(
            patch_keys=[key for key in keys if key is not None],
            patch_values=[value for value in values if value is not None],
            cls_state=embedding_output[:, : self.prefix_length].detach(),
            next_time=self.history_size * self.time_stride,
            history_size=self.history_size,
            patches_per_slot=self.patches_per_slot,
        )
        return output_hidden[:, 0], state

    def _current_patch_tokens(self, cells: torch.Tensor, *, legacy_interleaved: bool = False) -> torch.Tensor:
        if cells.ndim != 5 or cells.size(1) != FINGERS:
            raise ValueError(
                "stream_step cells must be (B,2,3,42,42), "
                f"got {tuple(cells.shape)}"
            )
        batch, _, channels, height, width = cells.shape
        if channels != 3 or height != self.frame_size or width != self.frame_size:
            raise ValueError("stream_step cells must contain 42x42 three-channel frames")
        flat = cells.reshape(batch * FINGERS, channels, height, width)
        patch_tokens = self._owner.embeddings.patch_embeddings(flat)
        patches = patch_tokens.size(1)
        patch_tokens = patch_tokens.reshape(batch, FINGERS, patches, -1)
        if legacy_interleaved:
            patch_tokens = patch_tokens.permute(0, 2, 1, 3)
        # Coordinates and finger_ids use all left patches followed by all right.
        patch_tokens = patch_tokens.reshape(batch, -1, patch_tokens.size(-1))
        finger_ids = torch.arange(FINGERS, device=patch_tokens.device).repeat_interleave(patches)
        return patch_tokens + self.finger_gate * self.finger_emb(finger_ids).unsqueeze(0).to(
            dtype=patch_tokens.dtype
        )

    @torch.no_grad()
    def stream_step(
        self,
        cells: torch.Tensor,
        state: TactileSequenceStreamState,
        *,
        corrected_window: bool = True,
    ) -> tuple[torch.Tensor, TactileSequenceStreamState]:
        """Encode only current L/R frames and evict the oldest cached slot."""
        if state.history_size != self.history_size:
            raise ValueError("stream state history_size does not match encoder")
        current = self._current_patch_tokens(cells, legacy_interleaved=not corrected_window)
        batch = current.size(0)
        rows = torch.arange(self.frame_size // self.patch_size, device=current.device)
        yy, xx = torch.meshgrid(rows, rows, indexing="ij")
        local = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
        coords_one = []
        for finger in range(FINGERS):
            x = local[:, 1]
            if finger and self.canonicalize_right:
                x = (self.frame_size // self.patch_size - 1) - x
            coords_one.append(
                torch.stack(
                    (
                        torch.full_like(local[:, 0], state.next_time),
                        local[:, 0],
                        x,
                    ),
                    dim=1,
                )
            )
        current_coords = torch.cat(coords_one, dim=0).to(dtype=torch.long)
        prefix = state.cls_state
        new_keys: list[torch.Tensor] = []
        new_values: list[torch.Tensor] = []
        keep = (self.history_size - 1) * self.patches_per_slot
        for index, layer in enumerate(self._owner.encoder.layer):
            attention = layer.attention.attention
            if not isinstance(attention, MosaicCausalRoPESelfAttention):
                raise TypeError("sequence stream found an unexpected attention wrapper")
            past_keys = state.patch_keys[index][:, :, -keep:] if keep else state.patch_keys[index][:, :, :0]
            past_values = state.patch_values[index][:, :, -keep:] if keep else state.patch_values[index][:, :, :0]
            query_input = torch.cat((prefix, current), dim=1)
            is_dino = hasattr(layer, "norm1")
            query_norm = layer.norm1(query_input) if is_dino else layer.layernorm_before(query_input)
            prefix_coords = torch.zeros((self.prefix_length, 3), device=current.device, dtype=torch.long)
            if corrected_window:
                # Cache patch keys keep absolute times. Move CLS/register queries
                # to the start of the current window to preserve relative RoPE.
                prefix_coords[:, 0] = state.next_time - (self.history_size - 1) * self.time_stride
            query_coords = torch.cat(
                (
                    prefix_coords,
                    current_coords,
                ),
                dim=0,
            )
            context, current_key, current_value = attention.forward_incremental(
                query_norm,
                past_keys,
                past_values,
                query_norm[:, self.prefix_length :],
                query_coords,
                current_coords,
            )
            # Incremental attention returns the inner attention context, before
            # the output projection. DINO additionally requires LayerScale.
            context = layer.attention.output(context, query_norm)
            if is_dino:
                hidden = query_input + layer.drop_path(layer.layer_scale1(context))
                layer_output = hidden + layer.drop_path(layer.layer_scale2(layer.mlp(layer.norm2(hidden))))
            else:
                hidden = query_input + context
                layer_output = layer.output(
                    layer.intermediate(layer.layernorm_after(hidden)), hidden
                )
            prefix = layer_output[:, : self.prefix_length]
            current = layer_output[:, self.prefix_length :]
            new_keys.append(torch.cat((past_keys, current_key.detach()), dim=2))
            new_values.append(torch.cat((past_values, current_value.detach()), dim=2))
        cls = self._owner.layernorm(prefix)[:, 0]
        next_state = TactileSequenceStreamState(
            patch_keys=new_keys,
            patch_values=new_values,
            cls_state=state.cls_state,
            next_time=state.next_time + self.time_stride,
            history_size=state.history_size,
            patches_per_slot=state.patches_per_slot,
        )
        if any(key.size(2) != self.patch_count for key in new_keys):
            raise RuntimeError("sequence stream cache lost its fixed eight-slot bound")
        if cls.size(0) != batch:
            raise RuntimeError("sequence stream batch size changed unexpectedly")
        return cls, next_state


def wrap_vit_tactile_sequence(
    vit: nn.Module,
    *,
    frame_size: int = DEFAULT_FRAME_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    history_size: int = DEFAULT_HISTORY_SIZE,
    time_stride: int = DEFAULT_HISTORY_STRIDE,
    canonicalize_right: bool = True,
    causal_attention: bool = True,
    rope_kind: str = "spatiotemporal_spiral",
) -> TactileSequenceEncoder:
    """Wrap a vendor/HuggingFace ViT with sequence RoPE attention.

    ``causal_attention=False`` retains the same sequence representation and
    Spiral RoPE but removes the slot-causal mask.  Streaming remains available
    only for the causal form because the incremental KV contract depends on
    historical slots never reading future slots.
    """
    return TactileSequenceEncoder(
        vit,
        frame_size=frame_size,
        patch_size=patch_size,
        history_size=history_size,
        time_stride=time_stride,
        canonicalize_right=canonicalize_right,
        causal_attention=causal_attention,
        rope_kind=rope_kind,
    )
