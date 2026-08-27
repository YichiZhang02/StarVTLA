from types import SimpleNamespace

import torch

from vtla.datasets.dataset_reader import apply_image_transforms
from vtla.datasets.factory import _resolve_image_transform_keys


def test_policy_transform_keys_exclude_configured_tactile_inputs():
    config = SimpleNamespace(
        tactile_keys=["observation.images.finger0", "observation.images.finger1"],
        selected_camera_keys=lambda: [
            "observation.images.wrist",
            "observation.images.finger0",
        ],
    )

    assert _resolve_image_transform_keys(config) == ["observation.images.wrist"]


def test_image_transforms_only_touch_allowlisted_rgb_keys():
    rgb_key = "observation.images.wrist"
    tactile_key = "observation.images.finger"
    images = {
        rgb_key: torch.zeros(3, 4, 4),
        tactile_key: torch.zeros(4, 3, 4, 4),
    }

    apply_image_transforms(
        images,
        [rgb_key, tactile_key],
        lambda image: image + 1,
        {rgb_key},
    )

    torch.testing.assert_close(images[rgb_key], torch.ones_like(images[rgb_key]))
    torch.testing.assert_close(images[tactile_key], torch.zeros_like(images[tactile_key]))


def test_image_transforms_keep_legacy_all_camera_default():
    images = {
        "observation.images.wrist": torch.zeros(3, 2, 2),
        "observation.images.finger": torch.zeros(3, 2, 2),
    }

    apply_image_transforms(images, images, lambda image: image + 1)

    assert all(torch.count_nonzero(image == 1) == image.numel() for image in images.values())
