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
