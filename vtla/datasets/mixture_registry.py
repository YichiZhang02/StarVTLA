"""Named, storage-free dataset mixture definitions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MIXTURE_CONFIG = "playground/data/data_mixtures.yaml"


def _validate_local_id(value: str, label: str) -> None:
    if not isinstance(value, str) or value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be one directory name, got {value!r}.")


def _validate_namespace(namespace: str) -> None:
    parts = namespace.split("/") if isinstance(namespace, str) else []
    if len(parts) != 2:
        raise ValueError(f"Expected source/division namespace, got {namespace!r}.")
    _validate_local_id(parts[0], "source")
    _validate_local_id(parts[1], "division")


def parse_dataset_selection(
    selection: str, registry_path: str | Path = DEFAULT_MIXTURE_CONFIG
) -> tuple[str, str, str]:
    """Return source, division and ID for a registered name or a three-part path."""
    parts = selection.split("/") if isinstance(selection, str) else []
    if len(parts) == 1:
        _validate_local_id(parts[0], "mixture name")
        if parts[0] not in load_mixture_definitions(registry_path):
            raise ValueError(f"Unknown dataset mixture {selection!r} in {registry_path}.")
        return parts[0], "all", "all"
    if len(parts) != 3:
        raise ValueError(
            f"Dataset selection must be a registered mixture name or source/group/id, got {selection!r}."
        )
    for label, value in zip(("source", "group", "dataset_id"), parts, strict=True):
        _validate_local_id(value, label)
    return parts[0], parts[1], parts[2]


def resolve_dataset_root(
    dataset_id: str, catalog_root: str | Path, namespace: str | None = None
) -> Path:
    """Resolve a source/group/dataset ID, retaining legacy lookup without a namespace."""
    catalog = Path(catalog_root)
    if namespace is not None:
        _validate_namespace(namespace)
        _validate_local_id(dataset_id, "dataset_id")
        return catalog / namespace / dataset_id
    direct = catalog / dataset_id
    if (direct / "meta" / "info.json").is_file():
        return direct
    matches = (
        sorted(
            child / dataset_id
            for child in catalog.iterdir()
            if child.is_dir() and (child / dataset_id / "meta" / "info.json").is_file()
        )
        if catalog.is_dir()
        else []
    )
    if len(matches) > 1:
        raise ValueError(
            f"Dataset ID {dataset_id!r} matches multiple local datasets: {matches}. "
            "Specify the concrete dataset root explicitly."
        )
    return matches[0] if matches else direct


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

    unknown = set(raw) - {
        "source", "group", "dataset_mixture", "dataset_id", "weight", "episodes", "revision", "root"
    }
    if unknown:
        raise ValueError(f"Mixture {mixture_id!r} member has unknown fields: {sorted(unknown)}")
    dataset_id = raw.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError(f"Mixture {mixture_id!r} member requires a non-empty dataset_id.")
    source, group = raw.get("source"), raw.get("group")
    if (source is None) != (group is None):
        raise ValueError(f"Mixture {mixture_id!r} member requires both source and group.")
    if source is not None:
        if raw.get("dataset_mixture") is not None:
            raise ValueError("Use source/group or dataset_mixture, not both.")
        for label, value in (("source", source), ("group", group)):
            if not isinstance(value, str) or value in {"", ".", ".."} or "/" in value:
                raise ValueError(f"Mixture {mixture_id!r} has invalid {label} {value!r}.")
    dataset_mixture = f"{source}/{group}" if source is not None else raw.get("dataset_mixture")
    if dataset_mixture is not None:
        _validate_namespace(dataset_mixture)
        _validate_local_id(dataset_id, "dataset_id")
        dataset_id = f"{dataset_mixture}/{dataset_id}"
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


def discover_group_mixture(
    dataset_id: str, catalog_root: str | Path | None, namespace: str | None = None
) -> MixtureDefinition | None:
    """Expand a source/group namespace into concrete LeRobot datasets."""
    if catalog_root is None or namespace is None:
        return None
    _validate_namespace(namespace)
    if dataset_id != "all":
        return None
    group_name = namespace
    group = Path(catalog_root) / group_name
    source_wide = namespace.endswith("/all")
    scan_root = group.parent if source_wide else group
    if not scan_root.is_dir() or (scan_root / "meta" / "info.json").is_file():
        return None
    if source_wide:
        divisions = sorted(
            child for child in scan_root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        )
        children = sorted(
            dataset for division in divisions for dataset in division.iterdir()
            if dataset.is_dir() and not dataset.name.startswith(".")
        )
    else:
        children = sorted(
            child for child in scan_root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        )
    if not children:
        raise ValueError(f"Dataset group {scan_root} contains no dataset directories.")
    invalid = [child for child in children if not (child / "meta" / "info.json").is_file()]
    if invalid:
        raise ValueError(f"Dataset group {scan_root} contains directories without meta/info.json: {invalid}")
    return MixtureDefinition(
        dataset_id=dataset_id,
        members=tuple(
            MixtureMember(
                dataset_id=str(child.relative_to(Path(catalog_root)))
            )
            for child in children
        ),
        root=str(scan_root),
    )


def _expand_group_members(
    definition: MixtureDefinition, catalog_root: str | Path | None
) -> MixtureDefinition:
    """Expand `source/all` entries into concrete datasets, preserving per-frame weights."""
    expanded = []
    for member in definition.members:
        if not member.dataset_id.endswith("/all"):
            expanded.append(member)
            continue
        if catalog_root is None or member.root is not None:
            raise ValueError(f"Cannot expand {member.dataset_id!r} without a catalog root.")
        source = member.dataset_id.removesuffix("/all")
        group = discover_group_mixture("all", catalog_root, source)
        if group is None:
            raise ValueError(f"Dataset group {source!r} does not exist.")
        expanded.extend(
            MixtureMember(
                dataset_id=child.dataset_id, weight=member.weight,
                episodes=member.episodes, revision=member.revision,
            )
            for child in group.members
        )
    ids = [member.dataset_id for member in expanded]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Mixture {definition.dataset_id!r} contains duplicate datasets after expansion.")
    return MixtureDefinition(definition.dataset_id, tuple(expanded), definition.root)


def resolve_mixture(
    dataset_id: str,
    registry_path: str | Path = DEFAULT_MIXTURE_CONFIG,
    resolved: dict[str, Any] | None = None,
    catalog_root: str | Path | None = None,
    namespace: str | None = None,
) -> MixtureDefinition | None:
    if resolved is not None:
        definition = mixture_from_dict(resolved)
        if definition.dataset_id != dataset_id:
            raise ValueError(
                f"Saved mixture ID {definition.dataset_id!r} does not match dataset.repo_id {dataset_id!r}."
            )
        saved_namespace = resolved.get("namespace")
        if saved_namespace is not None and saved_namespace != namespace:
            raise ValueError(
                f"Saved mixture namespace {saved_namespace!r} does not match {namespace!r}."
            )
        return definition
    registry_key = dataset_id if namespace is None else None
    if namespace is not None and dataset_id == "all":
        source, separator, group = namespace.partition("/")
        registry_key = source if separator and group == "all" else namespace
    registered = load_mixture_definitions(registry_path).get(registry_key) if registry_key else None
    if registered is not None:
        definition = MixtureDefinition(dataset_id, registered.members, registered.root)
        return _expand_group_members(definition, catalog_root)
    return discover_group_mixture(dataset_id, catalog_root, namespace)


def resolve_member_root(
    definition: MixtureDefinition,
    member: MixtureMember,
    catalog_root: str | Path | None = None,
) -> Path | None:
    if member.root is not None:
        return Path(member.root)
    if "/" in member.dataset_id:
        if catalog_root is None:
            return None
        return Path(catalog_root) / member.dataset_id
    base_root = definition.root if definition.root is not None else catalog_root
    return resolve_dataset_root(member.dataset_id, base_root) if base_root is not None else None
