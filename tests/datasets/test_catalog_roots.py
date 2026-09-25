import json
import sys

import pytest

from tools.resolve_training_dataset import main as resolve_training_main
from vtla.datasets.mixture_registry import (
    parse_dataset_selection, resolve_dataset_root, resolve_member_root, resolve_mixture,
)
from vtla.engine.configs.default import DatasetConfig
from vtla.tac_encoder.data.npy_tactile_dataset import resolve_tactile_dataset


def _dataset(root):
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"features": {}, "fps": 30}))


def test_three_level_single_dataset(tmp_path, monkeypatch, capsys):
    assert DatasetConfig(
        repo_id="episode_a", dataset_source="Daimon", dataset_group="realman_single"
    ).dataset_group == "realman_single"
    root = tmp_path / "Daimon" / "realman_single" / "episode_a"
    _dataset(root)
    assert resolve_dataset_root("episode_a", tmp_path, "Daimon/realman_single") == root
    tactile = resolve_tactile_dataset(
        "episode_a", dataset_source="Daimon", dataset_group="realman_single",
        dataset_catalog_root=tmp_path, require_caches=False,
    )
    assert tactile.kind == "dataset"
    assert tactile.members[0].root == str(root)

    monkeypatch.setattr(sys, "argv", [
        "resolve_training_dataset.py", "Daimon/realman_single/episode_a",
        "--catalog-root", str(tmp_path),
    ])
    resolve_training_main()
    assert capsys.readouterr().out.splitlines()[:5] == [
        "Daimon", "realman_single", "episode_a", "dataset", str(root),
    ]


def test_division_and_source_all_expand_to_concrete_members(tmp_path):
    for group, names in (("realman_single", ("b", "a")), ("umi", ("a",))):
        for name in names:
            _dataset(tmp_path / "Daimon" / group / name)
    division = resolve_mixture(
        "all", catalog_root=tmp_path, namespace="Daimon/realman_single",
        registry_path=tmp_path / "missing.yaml",
    )
    assert [member.dataset_id for member in division.members] == [
        "Daimon/realman_single/a", "Daimon/realman_single/b",
    ]
    source = resolve_mixture(
        "all", catalog_root=tmp_path, namespace="Daimon/all",
        registry_path=tmp_path / "missing.yaml",
    )
    assert [member.dataset_id for member in source.members] == [
        "Daimon/realman_single/a", "Daimon/realman_single/b", "Daimon/umi/a",
    ]
    assert resolve_mixture("all", resolved=source.to_dict()).to_dict() == source.to_dict()
    with pytest.raises(ValueError, match="namespace"):
        resolve_mixture(
            "all", resolved={**source.to_dict(), "namespace": "Daimon/all"},
            namespace="N0/all",
        )


def test_registry_combines_divisions_and_individual_datasets(tmp_path, monkeypatch, capsys):
    for source, group, names in (
        ("Daimon", "realman_single", ("a", "b")),
        ("N0", "umi", ("a", "c")),
    ):
        for name in names:
            _dataset(tmp_path / source / group / name)
    registry = tmp_path / "mixtures.yaml"
    registry.write_text(
        "version: 1\nmixtures:\n  Mix:\n    datasets:\n"
        "      - {source: Daimon, group: realman_single, dataset_id: all}\n"
        "      - {source: N0, group: umi, dataset_id: a, weight: 3}\n"
    )
    mixture = resolve_mixture(
        "all", registry_path=registry, catalog_root=tmp_path, namespace="Mix/all"
    )
    assert [member.dataset_id for member in mixture.members] == [
        "Daimon/realman_single/a", "Daimon/realman_single/b", "N0/umi/a",
    ]
    assert [member.weight for member in mixture.members] == [1, 1, 3]
    assert [resolve_member_root(mixture, member, tmp_path) for member in mixture.members] == [
        tmp_path / "Daimon/realman_single/a",
        tmp_path / "Daimon/realman_single/b",
        tmp_path / "N0/umi/a",
    ]

    monkeypatch.setattr(sys, "argv", [
        "resolve_training_dataset.py", "Mix",
        "--catalog-root", str(tmp_path), "--mixture-config", str(registry),
    ])
    resolve_training_main()
    assert capsys.readouterr().out.splitlines()[:5] == [
        "Mix", "all", "all", "mixture",
        "|".join(str(resolve_member_root(mixture, member, tmp_path)) for member in mixture.members),
    ]

    assert parse_dataset_selection("Mix", registry) == ("Mix", "all", "all")
    assert parse_dataset_selection("Daimon/realman_single/a", registry) == (
        "Daimon", "realman_single", "a",
    )
    with pytest.raises(ValueError, match="Unknown dataset mixture"):
        parse_dataset_selection("missing", registry)


def test_division_rejects_unconverted_leaf(tmp_path):
    (tmp_path / "UniVTAC" / "isaac45" / "raw_task").mkdir(parents=True)
    with pytest.raises(ValueError, match="without meta/info.json"):
        resolve_mixture(
            "all", registry_path=tmp_path / "missing.yaml", catalog_root=tmp_path,
            namespace="UniVTAC/isaac45",
        )
