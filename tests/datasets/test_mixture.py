from types import SimpleNamespace

import numpy as np
import pytest

from vtla.datasets.feature_schema import mixture_feature_schema_diff
from vtla.datasets.mixture_registry import load_mixture_definitions, mixture_from_dict
from vtla.datasets.multi_dataset import (
    MixtureLeRobotDataset,
    aggregate_weighted_stats,
    validate_mixture_metadata,
)
from vtla.datasets.sampler import MixtureSampler
from vtla.datasets.visual_preprocess import make_visual_preprocess


def _visual_feature(**overrides):
    feature = {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channels"],
    }
    feature.update(overrides)
    return feature


def test_mixture_schema_ignores_camera_calibration_and_storage_metadata():
    reference = {
        "observation.images.cam_top": _visual_feature(
            intrinsics={"224x224": {"fx": 80.0}},
            imu_to_rgb_camera=[[1, 0], [0, 1]],
            info={"video.codec": "h264", "video.pix_fmt": "yuv420p"},
            video_path="videos/reference/{video_key}.mp4",
            external_video=False,
        )
    }
    candidate = {
        "observation.images.cam_top": _visual_feature(
            intrinsics={"224x224": {"fx": 75.0}},
            imu_to_rgb_camera=[[0, 1], [1, 0]],
            extrinsics={"camera": "different"},
            info={"video.codec": "hevc", "video.pix_fmt": "gbrp"},
            video_path="other/layout/{video_key}.mkv",
            external_video=True,
        )
    }

    assert mixture_feature_schema_diff(reference, candidate) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtype", "image"),
        ("shape", [256, 256, 3]),
        ("names", ["channels", "height", "width"]),
        ("tactile_encoding", "tactile_u8_linear_v1"),
        ("storage_dtype", "uint8"),
    ],
)
def test_mixture_schema_rejects_training_contract_differences(field, value):
    reference = {"observation.images.cam_top": _visual_feature()}
    candidate = {"observation.images.cam_top": _visual_feature(**{field: value})}

    differences = mixture_feature_schema_diff(reference, candidate)

    assert len(differences) == 1
    assert f"field {field!r}" in differences[0]


def test_mixture_schema_rejects_missing_or_extra_feature_keys():
    reference = {"camera": _visual_feature()}

    assert mixture_feature_schema_diff(reference, {}) == ["missing feature 'camera'"]
    assert mixture_feature_schema_diff({}, reference) == ["extra feature 'camera'"]


def test_runtime_mixture_validation_uses_training_contract_schema():
    reference = _visual_feature(intrinsics={"224x224": {"fx": 80.0}})
    candidate = _visual_feature(intrinsics={"224x224": {"fx": 75.0}})
    preprocess = make_visual_preprocess(size=224, wrist_undistort=True, tactile_encoding=None)
    datasets = [
        SimpleNamespace(
            repo_id="first",
            meta=SimpleNamespace(
                fps=30,
                robot_type="umi",
                features={"camera": reference},
                visual_preprocess=preprocess,
            ),
        ),
        SimpleNamespace(
            repo_id="second",
            meta=SimpleNamespace(
                fps=30,
                robot_type="umi",
                features={"camera": candidate},
                visual_preprocess=preprocess,
            ),
        ),
    ]

    validate_mixture_metadata(datasets)

    datasets[1].meta.features["camera"]["shape"] = [256, 256, 3]
    with pytest.raises(ValueError, match="field 'shape'"):
        validate_mixture_metadata(datasets)


def test_runtime_mixture_validation_rejects_visual_preprocess_mismatch():
    preprocess = make_visual_preprocess(
        size=224, wrist_undistort=True, tactile_encoding="tactile_u8_linear_v1"
    )
    datasets = [
        SimpleNamespace(
            repo_id="first",
            meta=SimpleNamespace(
                fps=30,
                robot_type="umi",
                features={"camera": _visual_feature()},
                visual_preprocess=preprocess,
            ),
        ),
        SimpleNamespace(
            repo_id="second",
            meta=SimpleNamespace(
                fps=30,
                robot_type="umi",
                features={"camera": _visual_feature()},
                visual_preprocess={**preprocess, "wrist_undistort": False},
            ),
        ),
    ]

    with pytest.raises(ValueError, match="visual_preprocess expected"):
        validate_mixture_metadata(datasets)


def test_runtime_mixture_requires_robot_and_visual_contract():
    dataset = SimpleNamespace(
        repo_id="first",
        meta=SimpleNamespace(fps=30, robot_type=None, features={}, visual_preprocess=None),
    )
    with pytest.raises(ValueError, match="has no robot_type"):
        validate_mixture_metadata([dataset])

    dataset.meta.robot_type = "umi"
    with pytest.raises(ValueError, match="has no visual_preprocess contract"):
        validate_mixture_metadata([dataset])


def test_mixture_metadata_carries_visual_preprocess_contract():
    class Child(SimpleNamespace):
        def __len__(self):
            return 10

    preprocess = make_visual_preprocess(size=224, wrist_undistort=True, tactile_encoding=None)
    children = []
    for dataset_id in ("first", "second"):
        child = Child(
            repo_id=dataset_id,
            root=f"/tmp/{dataset_id}",
            num_episodes=1,
            meta=SimpleNamespace(
                fps=30,
                robot_type="umi",
                features={"camera": _visual_feature()},
                visual_preprocess=preprocess,
                stats={},
                tasks=SimpleNamespace(index=[f"task-{dataset_id}"]),
            ),
        )
        children.append(child)
    definition = mixture_from_dict(
        {
            "dataset_id": "combined",
            "root": "/tmp",
            "members": [{"dataset_id": "first"}, {"dataset_id": "second"}],
        }
    )

    mixture = MixtureLeRobotDataset(children, definition)

    assert mixture.meta.visual_preprocess == preprocess


def test_registry_defaults_to_equal_weights_and_roundtrips(tmp_path):
    registry = tmp_path / "mixtures.yaml"
    registry.write_text(
        """
version: 1
mixtures:
  combined:
    root: data
    datasets:
      - dataset_id: first
      - dataset_id: second
""",
        encoding="utf-8",
    )

    definition = load_mixture_definitions(registry)["combined"]

    assert definition.normalized_weights == (0.5, 0.5)
    restored = mixture_from_dict(definition.to_dict())
    assert restored == definition


def test_weighted_stats_follow_dataset_sampling_weights():
    first = {
        "action": {
            "mean": np.array([0.0]),
            "std": np.array([1.0]),
            "min": np.array([-2.0]),
            "max": np.array([2.0]),
            "count": np.array([10]),
        }
    }
    second = {
        "action": {
            "mean": np.array([10.0]),
            "std": np.array([1.0]),
            "min": np.array([8.0]),
            "max": np.array([12.0]),
            "count": np.array([90]),
        }
    }

    stats = aggregate_weighted_stats([first, second], (0.5, 0.5))["action"]

    np.testing.assert_allclose(stats["mean"], [5.0])
    np.testing.assert_allclose(stats["std"], [np.sqrt(26.0)])
    np.testing.assert_array_equal(stats["count"], [100])


def test_mixture_sampler_uses_dataset_weights_not_dataset_lengths():
    class Child:
        def __init__(self, length):
            self.length = length

        def __len__(self):
            return self.length

    children = [Child(100), Child(10_000)]
    mixture = SimpleNamespace(
        _datasets=children,
        repo_ids=["small", "large"],
        weights=(0.5, 0.5),
        _offsets=[0, 100],
    )
    sampler = MixtureSampler(mixture, num_samples=20_000, seed=123)

    indices = list(sampler)
    small_fraction = sum(index < 100 for index in indices) / len(indices)

    assert 0.48 < small_fraction < 0.52


def test_mixture_rejects_tcp_contract_or_relative_stats_mismatch():
    from vtla.datasets.tcp_contract import build_tcp_contract
    contract = build_tcp_contract('umi', 3, 0)
    preprocess = make_visual_preprocess(size=224, wrist_undistort=True, tactile_encoding=None)
    def member(name):
        return SimpleNamespace(repo_id=name, meta=SimpleNamespace(
            fps=30, robot_type='umi', features={}, visual_preprocess=preprocess,
            tcp_contract=contract, stats={'action_relative_ee': {}}))
    first, second = member('first'), member('second')
    validate_mixture_metadata([first, second])
    second.meta.tcp_contract = build_tcp_contract('umi', 3, 1)
    with pytest.raises(ValueError, match='TCP contract'):
        validate_mixture_metadata([first, second])
    second.meta.tcp_contract = contract
    second.meta.stats = {}
    with pytest.raises(ValueError, match='missing action_relative_ee'):
        validate_mixture_metadata([first, second])


def _sampling_child(name, lengths, episodes=None, mean=0.0):
    class Child(SimpleNamespace):
        def __len__(self):
            selected = range(len(lengths)) if self.episodes is None else self.episodes
            return sum(lengths[i] for i in selected)

    ends = np.cumsum(lengths)
    starts = np.concatenate(([0], ends[:-1]))
    return Child(
        repo_id=name,
        root=f"/tmp/{name}",
        episodes=episodes,
        num_episodes=len(lengths) if episodes is None else len(episodes),
        meta=SimpleNamespace(
            fps=30,
            robot_type="umi",
            visual_preprocess=make_visual_preprocess(size=224, wrist_undistort=True, tactile_encoding=None),
            features={},
            tasks=SimpleNamespace(index=[name]),
            total_episodes=len(lengths),
            episodes=[dict(dataset_from_index=int(a), dataset_to_index=int(b)) for a, b in zip(starts, ends)],
            stats={"action": {"mean": np.array([mean]), "std": np.array([1.0])}},
        ),
    )


def _sampling_definition(weights=(1.0, 1.0)):
    return mixture_from_dict({
        "dataset_id": "combined",
        "members": [dict(dataset_id=name, weight=w) for name, w in zip(("a", "b"), weights)],
    })


@pytest.mark.parametrize("weights,expected", [((1.0, 1.0), (0.1, 0.9)), ((9.0, 1.0), (0.5, 0.5))])
def test_vla_frame_weighted_sampling_and_statistics(weights, expected):
    mixture = MixtureLeRobotDataset(
        [_sampling_child("a", [10]), _sampling_child("b", [90], mean=10)],
        _sampling_definition(weights),
    )
    np.testing.assert_allclose(mixture.weights, expected)
    np.testing.assert_allclose(mixture.meta.stats["action"]["mean"], [10 * expected[1]])
    np.testing.assert_allclose(mixture.meta.stats["action"]["std"], [np.sqrt(1 + 100 * expected[0] * expected[1])])
    sampler = MixtureSampler(mixture, num_samples=30_000, seed=12)
    assert sampler.valid_indices is mixture.valid_indices
    np.testing.assert_allclose(sampler.weights.numpy(), expected)
    indices = list(sampler)
    assert all(0 <= index < len(mixture) for index in indices)
    assert abs(sum(index < 10 for index in indices) / len(indices) - expected[0]) < 0.015


def test_vla_effective_counts_use_selected_episodes_and_trimming():
    mixture = MixtureLeRobotDataset(
        [_sampling_child("a", [10, 4, 20, 100], episodes=[2, 0, 1]), _sampling_child("b", [12, 12])],
        _sampling_definition(),
        drop_n_last_frames=6,
    )
    # The four-frame episode is discarded; unselected episode 3 contributes nothing.
    assert mixture.effective_num_frames == (18, 12)
    np.testing.assert_allclose(mixture.weights, (0.6, 0.4))
    sampler = MixtureSampler(mixture, drop_n_last_frames=6, num_samples=3000)
    valid = set(range(4)) | set(range(14, 28)) | set(range(34, 40)) | set(range(46, 52))
    assert set(sampler) <= valid
    with pytest.raises(ValueError, match="must match"):
        MixtureSampler(mixture)


def test_vla_rejects_empty_member_after_trimming():
    with pytest.raises(ValueError, match="No valid frames.*a"):
        MixtureLeRobotDataset(
            [_sampling_child("a", [4]), _sampling_child("b", [12])],
            _sampling_definition(), drop_n_last_frames=6,
        )


def test_vla_saved_sampling_strategy_compatibility():
    from vtla.datasets.multi_dataset import resolve_sampling_strategy

    assert resolve_sampling_strategy(None) == "weighted_frames"
    # Old resolved snapshots have normalized_weights but no strategy marker.
    old_snapshot = _sampling_definition().to_dict()
    assert resolve_sampling_strategy(old_snapshot) == "dataset_weights"
    legacy = MixtureLeRobotDataset(
        [_sampling_child("a", [10]), _sampling_child("b", [90], mean=10)],
        mixture_from_dict(old_snapshot), sampling_strategy=resolve_sampling_strategy(old_snapshot),
    )
    np.testing.assert_allclose(legacy.weights, (0.5, 0.5))
    np.testing.assert_allclose(legacy.meta.stats["action"]["mean"], [5.0])
    new_snapshot = {**old_snapshot, "sampling_strategy": "weighted_frames", "normalized_weights": [0.1, 0.9]}
    restored = MixtureLeRobotDataset(
        legacy._datasets, mixture_from_dict(new_snapshot),
        sampling_strategy=resolve_sampling_strategy(new_snapshot),
    )
    np.testing.assert_allclose(restored.weights, (0.1, 0.9))
    # Shared definition normalization (used by backbone) is unchanged.
    assert restored.definition.normalized_weights == (0.5, 0.5)
    with pytest.raises(ValueError, match="Unknown VLA"):
        resolve_sampling_strategy({"sampling_strategy": "typo"})
