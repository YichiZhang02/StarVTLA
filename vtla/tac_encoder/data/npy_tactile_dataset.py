"""Map-style datasets backed exclusively by versioned ``.npy`` caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from vtla.datasets.mixture_registry import load_mixture_definitions, resolve_member_root

from .cache_schema import SPLIT_NAME_TO_ID, cache_signature, stable_hash, validate_cache


IN_PLACE_CACHE_DIR = "tactile_backbone_cache"


@dataclass(frozen=True)
class ResolvedTactileMember:
    dataset_id: str
    root: str
    cache_root: str
    weight: float
    normalized_weight: float
    episodes: list[int] | None
    revision: str | None
    cache_signature: str


@dataclass(frozen=True)
class ResolvedTactileDataset:
    dataset_id: str
    kind: str
    mixture_config: str
    mixture_config_hash: str | None
    members: tuple[ResolvedTactileMember, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "kind": self.kind,
            "mixture_config": self.mixture_config,
            "mixture_config_hash": self.mixture_config_hash,
            "members": [asdict(member) for member in self.members],
        }

    @property
    def signature(self) -> str:
        return stable_hash(self.to_dict())


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_tactile_dataset(
    dataset_id: str,
    *,
    cache_root: str | Path | None = None,
    dataset_catalog_root: str | Path = "playground/data",
    mixture_config: str | Path = "configs/data_mixtures.yaml",
    require_caches: bool = True,
) -> ResolvedTactileDataset:
    catalog = Path(dataset_catalog_root)
    registry_path = Path(mixture_config)
    definitions = load_mixture_definitions(registry_path)
    definition = definitions.get(dataset_id)
    physical = catalog / dataset_id
    if definition is not None and physical.is_dir():
        raise ValueError(f"Dataset ID {dataset_id!r} is both a mixture and a directory: {physical}")

    if definition is None:
        raw_members = [(dataset_id, physical, 1.0, 1.0, None, None)]
        kind = "dataset"
    else:
        raw_members = []
        for member, normalized_weight in zip(
            definition.members, definition.normalized_weights, strict=True
        ):
            root = resolve_member_root(definition, member, catalog)
            if root is None:
                raise ValueError(f"Mixture member {member.dataset_id!r} has no local root")
            raw_members.append(
                (member.dataset_id, root, member.weight, normalized_weight, member.episodes, member.revision)
            )
        kind = "mixture"

    resolved_members = []
    reference = None
    for member_id, root, weight, normalized_weight, episodes, revision in raw_members:
        root = Path(root)
        if not require_caches and not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Processed dataset metadata not found: {root / 'meta' / 'info.json'}")
        member_cache = (
            root / IN_PLACE_CACHE_DIR
            if cache_root is None
            else Path(cache_root) / member_id
        )
        signature = cache_signature(member_cache) if require_caches else ""
        if require_caches:
            cache = validate_cache(member_cache)
            metadata = (
                int(cache.scalar("fps")),
                tuple(str(x) for x in cache.arrays["sensor_names.npy"].tolist()),
                int(cache.scalar("image_size")),
                int(cache.scalar("num_frames")),
                int(cache.scalar("frame_stride")),
            )
            cache.close()
            if reference is None:
                reference = metadata
            elif metadata != reference:
                raise ValueError(
                    f"Mixture member cache {member_id!r} is incompatible with the first member: "
                    f"expected {reference}, got {metadata}"
                )
        resolved_members.append(
            ResolvedTactileMember(
                dataset_id=member_id,
                root=str(root),
                cache_root=str(member_cache),
                weight=float(weight),
                normalized_weight=float(normalized_weight),
                episodes=None if episodes is None else list(episodes),
                revision=revision,
                cache_signature=signature,
            )
        )
    return ResolvedTactileDataset(
        dataset_id=dataset_id,
        kind=kind,
        mixture_config=str(registry_path),
        mixture_config_hash=_file_sha256(registry_path) if kind == "mixture" else None,
        members=tuple(resolved_members),
    )


class TactileNpyDataset(Dataset):
    """A concrete member cache. No raw dataset path is accessed after init."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        split: str = "train",
        episodes: list[int] | None = None,
        member_index: int = 0,
    ) -> None:
        if split not in SPLIT_NAME_TO_ID:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_NAME_TO_ID)}")
        self.cache = validate_cache(cache_root)
        split_mask = self.cache.arrays["split.npy"] == SPLIT_NAME_TO_ID[split]
        if episodes is not None:
            anchors = self.cache.arrays["window_anchor.npy"]
            anchor_episodes = self.cache.arrays["episode_index.npy"][anchors]
            split_mask &= np.isin(anchor_episodes, np.asarray(episodes, dtype=np.int64))
        self.indices = np.flatnonzero(split_mask).astype(np.int64)
        if len(self.indices) == 0:
            raise ValueError(f"No {split} windows remain in tactile cache {cache_root}")
        self.member_index = int(member_index)

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.cache.arrays["sensor_names.npy"].tolist())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window_row = int(self.indices[index])
        frame_rows = self.cache.arrays["windows.npy"][window_row]
        # Advanced indexing returns a writable in-memory copy while frames.npy remains mmap-only.
        frames = np.asarray(self.cache.arrays["frames.npy"][frame_rows]).copy()
        valid = np.asarray(self.cache.arrays["valid.npy"][frame_rows]).copy()
        # Cache is [T,S,H,W,C]; public model contract is [S,T,C,H,W].
        images = torch.from_numpy(frames).permute(1, 0, 4, 2, 3).contiguous().float().div_(255.0)
        valid_tensor = torch.from_numpy(valid.T.copy())
        anchor = int(self.cache.arrays["window_anchor.npy"][window_row])
        return {
            "images": images,
            "valid": valid_tensor,
            "contact_scores": torch.from_numpy(
                np.asarray(self.cache.arrays["contact_scores.npy"][frame_rows]).T.copy()
            ),
            "contact_mask": torch.from_numpy(
                np.asarray(self.cache.arrays["contact_mask.npy"][frame_rows]).T.copy()
            ),
            "episode_index": int(self.cache.arrays["episode_index.npy"][anchor]),
            "frame_index": int(self.cache.arrays["frame_index.npy"][anchor]),
            "member_index": self.member_index,
            "window_index": window_row,
        }


class TactileMixtureDataset(Dataset):
    def __init__(self, datasets: list[TactileNpyDataset], resolved: ResolvedTactileDataset) -> None:
        if len(datasets) != len(resolved.members):
            raise ValueError("Resolved members and concrete datasets do not match")
        self.datasets = datasets
        self.resolved = resolved
        self.offsets = []
        running = 0
        for dataset in datasets:
            self.offsets.append(running)
            running += len(dataset)
        self.length = running

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        member = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return self.datasets[member][index - self.offsets[member]]


class WeightedMixtureSampler(Sampler[int]):
    """Choose a member by configured weight, then a window uniformly."""

    def __init__(
        self,
        dataset: TactileMixtureDataset,
        *,
        num_samples: int | None = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.num_samples = int(num_samples or len(dataset))
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.weights = torch.tensor(
            [member.normalized_weight for member in dataset.resolved.members], dtype=torch.double
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (self.num_samples + self.world_size - 1) // self.world_size

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total = len(self) * self.world_size
        choices = torch.multinomial(self.weights, total, replacement=True, generator=generator)
        result = []
        for member_index in choices.tolist():
            child = self.dataset.datasets[member_index]
            local = int(torch.randint(len(child), (1,), generator=generator))
            result.append(self.dataset.offsets[member_index] + local)
        yield from result[self.rank:total:self.world_size]


def build_training_dataset(
    resolved: ResolvedTactileDataset,
    *,
    split: str,
) -> TactileMixtureDataset:
    children = [
        TactileNpyDataset(
            member.cache_root,
            split=split,
            episodes=member.episodes,
            member_index=index,
        )
        for index, member in enumerate(resolved.members)
    ]
    return TactileMixtureDataset(children, resolved)


def save_resolved_definition(resolved: ResolvedTactileDataset, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(resolved.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
