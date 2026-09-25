"""Per-source action Linear maps into a shared LeWM Embedder."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


SOURCE_NAMES = (
    "tacbench",  # 0
    "touchhd",  # 1 (deprecated: no actions; excluded from ftp1 cotrain mix)
    "daimon",  # 2
    "ftp1_rdp",  # 3
    "ftp1_vla_touch",  # 4
    "ftp1_qingloong",  # 5
)
SOURCE_NAME_TO_ID = {name: idx for idx, name in enumerate(SOURCE_NAMES)}


def parse_source_dims(value: Mapping[str, Any] | None) -> dict[str, int]:
    """Normalize Hydra ``source_dims`` to ``{name: int}`` for known sources.

    ``touchhd`` is allowed to be absent from explicit maps (legacy no-action
    source, excluded from the ftp1 cotrain mix); every other source must be
    present.
    """
    if value is None:
        return {name: 8 for name in SOURCE_NAMES}
    dims: dict[str, int] = {}
    for name in SOURCE_NAMES:
        if name == "touchhd" and name not in value:
            dims[name] = 8
            continue
        if name not in value:
            raise KeyError(f"source_dims missing {name!r}; need {SOURCE_NAMES}")
        dims[name] = int(value[name])
        if dims[name] <= 0:
            raise ValueError(f"source_dims[{name!r}] must be > 0, got {dims[name]}")
    return dims


class MultiSourceActionEmbedder(nn.Module):
    """Map heterogeneous actions into one shared Embedder space.

    Each source keeps a trainable ``Linear(src_dim → unified_dim)``.  The
    shared ``Embedder`` (Conv1d + MLP) is identical to the single-source
    LeWM path, so the predictor still consumes one action embedding family.
    ``set_source_id`` is called by ``DualEncoderJEPA.encode`` before JEPA
    runs ``action_encoder(action)``.
    """

    def __init__(
        self,
        *,
        source_dims: Mapping[str, Any] | None = None,
        unified_dim: int = 8,
        emb_dim: int = 384,
        smoothed_dim: int | None = None,
        mlp_scale: int = 4,
        embedder: nn.Module | None = None,
        # Train entry still writes ``action_encoder.input_dim`` from the
        # dataset (frameskip * action_dim).  Treat it as unified_dim so the
        # shared Embedder stays compatible with that Hydra hook.
        input_dim: int | None = None,
        # Fine-tuning on a single-source dataset leaves the other source
        # Linears dead (always mask=0).  ``freeze_dead`` freezes them and
        # makes them identity (weight = I, bias = 0) so their stored state
        # does not poison the shared Embedder through dead-masked inputs.
        freeze_dead: bool = False,
    ) -> None:
        super().__init__()
        self.source_dims = parse_source_dims(source_dims)
        self.unified_dim = int(input_dim) if input_dim is not None else int(unified_dim)
        if self.unified_dim <= 0:
            raise ValueError(f"unified_dim must be > 0, got {self.unified_dim}")
        self.emb_dim = int(emb_dim)
        self.projs = nn.ModuleDict(
            {
                name: nn.Linear(dim, self.unified_dim)
                for name, dim in self.source_dims.items()
            }
        )
        if embedder is None:
            from .module import Embedder

            smooth = int(smoothed_dim) if smoothed_dim is not None else self.unified_dim
            embedder = Embedder(
                input_dim=self.unified_dim,
                smoothed_dim=smooth,
                emb_dim=self.emb_dim,
                mlp_scale=int(mlp_scale),
            )
        self.embedder = embedder
        # JEPA calls ``action_encoder(action)`` with no source kwarg; stash
        # the batch source ids on the module for the duration of encode.
        self._source_id: torch.Tensor | None = None
        self._freeze_dead = bool(freeze_dead)
        self._active_sources: set[int] = set()
        if self._freeze_dead:
            self.set_active_sources({0})

    @property
    def patch_embed(self) -> nn.Module:
        """SWM / DualEncoderJEPA probe the Embedder via ``patch_embed``."""
        return self.embedder.patch_embed

    def set_source_id(self, source_id: torch.Tensor | None) -> None:
        if source_id is None:
            self._source_id = None
            return
        tensor = torch.as_tensor(source_id)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        tensor = tensor.long()
        self._source_id = tensor.reshape(-1)
        if self._freeze_dead and self._source_id is not None:
            # Keep only the sources present in this batch trainable.
            active = {int(s) for s in self._source_id.unique().tolist()}
            if active != self._active_sources:
                self.set_active_sources(active)

    def set_active_sources(self, active: set[int] | frozenset[int] | None) -> None:
        """Freeze every source Linear whose id is not in ``active``.

        Frozen sources keep their current weights (so re-activating them
        later restores the same mapping), but ``_project`` bypasses the
        Linear and uses the identity slice of the action instead.  This
        keeps the shared Embedder fed with a stable single-source
        distribution while leaving the frozen branch's parameters intact.
        """
        active_set: set[int] = set() if active is None else {int(s) for s in active}
        self._active_sources = active_set
        with torch.no_grad():
            for name, sid in SOURCE_NAME_TO_ID.items():
                proj = self.projs[name]
                if not self._freeze_dead:
                    proj.weight.requires_grad_(True)
                    proj.bias.requires_grad_(True)
                    continue
                trainable = sid in active_set
                proj.weight.requires_grad_(trainable)
                proj.bias.requires_grad_(trainable)

    def clear_source_id(self) -> None:
        self._source_id = None

    def _project(self, action: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        if action.ndim != 3:
            raise ValueError(f"expected action (B, T, D), got {tuple(action.shape)}")
        batch = int(action.size(0))
        if int(source_id.numel()) != batch:
            raise ValueError(
                f"source_id length {int(source_id.numel())} != batch {batch}"
            )
        out = action.new_zeros(batch, int(action.size(1)), self.unified_dim)
        for name, sid in SOURCE_NAME_TO_ID.items():
            dim = self.source_dims[name]
            mask = (source_id == int(sid)).to(dtype=out.dtype).view(batch, 1, 1)
            if not mask.any():
                # No sample in this batch uses this source.  In DDP every
                # non-frozen param must receive a grad each step; if this
                # Linear is frozen (freeze_dead), there is nothing to sync,
                # so skip it entirely.  Otherwise keep the legacy behaviour
                # of running every Linear so DDP finds grads for all params.
                if not self.projs[name].weight.requires_grad:
                    continue
            if self._freeze_dead and int(sid) not in self._active_sources:
                # Frozen source: identity projection (weight = I, bias = 0),
                # so output == raw action slice without a dead Linear call.
                out = out + mask * action[..., :dim].to(dtype=out.dtype)
                continue
            projected = self.projs[name](action[..., :dim]).to(dtype=out.dtype)
            out = out + mask * projected
        return out

    def forward(self, action: torch.Tensor, source_id: torch.Tensor | None = None) -> torch.Tensor:
        """Embed ``(B, T, D)`` actions; ``source_id`` defaults to the stash."""
        sid = source_id if source_id is not None else self._source_id
        if sid is None:
            sid = action.new_zeros(int(action.size(0)), dtype=torch.long)
        else:
            sid = torch.as_tensor(sid, device=action.device).long().reshape(-1)
        projected = self._project(action, sid)
        return self.embedder(projected)
