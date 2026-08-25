"""Named, storage-free dataset mixture definitions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MIXTURE_CONFIG = "configs/data_mixtures.yaml"


@dataclass(frozen=True)
class MixtureMember:
    dataset_id: str
    weight: float = 1.0
    episodes: list[int] | None = None
    revision: str | None = None
    root: str | None = None


@dataclass(frozen=True)
class MixtureDefinition:
    dataset_id: str
    members: tuple[MixtureMember, ...]
    root: str | None = None

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        total = sum(member.weight for member in self.members)
        return tuple(member.weight / total for member in self.members)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["members"] = [asdict(member) for member in self.members]
        data["normalized_weights"] = list(self.normalized_weights)
        return data


def _parse_member(raw: Any, mixture_id: str) -> MixtureMember:
    if isinstance(raw, str):
        raw = {"dataset_id": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"Mixture {mixture_id!r} members must be strings or mappings, got {raw!r}.")

    unknown = set(raw) - {"dataset_id", "weight", "episodes", "revision", "root"}
    if unknown:
        raise ValueError(f"Mixture {mixture_id!r} member has unknown fields: {sorted(unknown)}")
    dataset_id = raw.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError(f"Mixture {mixture_id!r} member requires a non-empty dataset_id.")
    weight = float(raw.get("weight", 1.0))
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(
            f"Mixture {mixture_id!r} member {dataset_id!r} weight must be finite and > 0, got {weight}."
        )
    episodes = raw.get("episodes")
    if episodes is not None:
        if (
            not isinstance(episodes, list)
            or not episodes
            or any(not isinstance(ep, int) or ep < 0 for ep in episodes)
        ):
            raise ValueError(
                f"Mixture {mixture_id!r} member {dataset_id!r} episodes must be a non-empty list "
                "of non-negative integers."
            )
        if len(episodes) != len(set(episodes)):
            raise ValueError(f"Mixture {mixture_id!r} member {dataset_id!r} contains duplicate episodes.")
    return MixtureMember(
        dataset_id=dataset_id,
        weight=weight,
        episodes=episodes,
        revision=raw.get("revision"),
        root=raw.get("root"),
    )


def load_mixture_definitions(path: str | Path = DEFAULT_MIXTURE_CONFIG) -> dict[str, MixtureDefinition]:
    """Load named mixtures. A missing registry is equivalent to an empty one."""
    config_path = Path(path)
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Mixture registry {config_path} must contain a mapping.")
    if raw_config.get("version", 1) != 1:
        raise ValueError(f"Unsupported mixture registry version in {config_path}: {raw_config.get('version')!r}")
    raw_mixtures = raw_config.get("mixtures", {})
    if not isinstance(raw_mixtures, dict):
        raise ValueError(f"The 'mixtures' field in {config_path} must be a mapping.")

    definitions = {}
    for mixture_id, raw_definition in raw_mixtures.items():
        if not isinstance(mixture_id, str) or not mixture_id.strip():
            raise ValueError(f"Mixture names in {config_path} must be non-empty strings.")
        if not isinstance(raw_definition, dict):
            raise ValueError(f"Mixture {mixture_id!r} must be a mapping.")
        unknown = set(raw_definition) - {"datasets", "root"}
        if unknown:
            raise ValueError(f"Mixture {mixture_id!r} has unknown fields: {sorted(unknown)}")
        raw_members = raw_definition.get("datasets")
        if not isinstance(raw_members, list) or not raw_members:
            raise ValueError(f"Mixture {mixture_id!r} must contain a non-empty datasets list.")
        members = tuple(_parse_member(member, mixture_id) for member in raw_members)
        member_ids = [member.dataset_id for member in members]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError(f"Mixture {mixture_id!r} contains duplicate dataset IDs.")
        if mixture_id in member_ids:
            raise ValueError(f"Mixture {mixture_id!r} cannot contain itself.")
        definitions[mixture_id] = MixtureDefinition(
            dataset_id=mixture_id,
            members=members,
            root=raw_definition.get("root"),
        )
    for definition in definitions.values():
        nested = sorted(member.dataset_id for member in definition.members if member.dataset_id in definitions)
        if nested:
            raise ValueError(
                f"Mixture {definition.dataset_id!r} contains nested mixtures {nested}; "
                "only concrete dataset members are supported."
            )
    return definitions


def mixture_from_dict(data: dict[str, Any]) -> MixtureDefinition:
    """Restore a resolved definition embedded in a saved training config."""
    mixture_id = str(data["dataset_id"])
    members = tuple(_parse_member(member, mixture_id) for member in data["members"])
    return MixtureDefinition(dataset_id=mixture_id, members=members, root=data.get("root"))


def resolve_mixture(
    dataset_id: str,
    registry_path: str | Path = DEFAULT_MIXTURE_CONFIG,
    resolved: dict[str, Any] | None = None,
) -> MixtureDefinition | None:
    if resolved is not None:
        definition = mixture_from_dict(resolved)
        if definition.dataset_id != dataset_id:
            raise ValueError(
                f"Saved mixture ID {definition.dataset_id!r} does not match dataset.repo_id {dataset_id!r}."
            )
        return definition
    return load_mixture_definitions(registry_path).get(dataset_id)


def resolve_member_root(
    definition: MixtureDefinition,
    member: MixtureMember,
    catalog_root: str | Path | None = None,
) -> Path | None:
    if member.root is not None:
        return Path(member.root)
    base_root = definition.root if definition.root is not None else catalog_root
    return Path(base_root) / member.dataset_id if base_root is not None else None
