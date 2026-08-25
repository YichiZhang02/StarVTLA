from types import SimpleNamespace

import numpy as np

from vtla.datasets.mixture_registry import load_mixture_definitions, mixture_from_dict
from vtla.datasets.multi_dataset import aggregate_weighted_stats
from vtla.datasets.sampler import MixtureSampler


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
