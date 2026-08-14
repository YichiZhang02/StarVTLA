from pathlib import Path

import pytest
import torch

from vtla.engine.configs.default import DatasetConfig
from vtla.engine.configs.train import TrainPipelineConfig
from vtla.engine.transforms import ImageTransformsConfig
from vtla.frameworks.factory import get_policy_class, make_policy_config
from vtla.frameworks.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig
from vtla.frameworks.starvla_groot_dinoalign.configuration_starvla_groot_dinoalign import (
    StarvlaGrootDinoAlignConfig,
)
from vtla.frameworks.starvla_groot_dinoalign.dinov3_alignment import (
    DinoAlignmentHead,
    IlluminationAugment,
)


def test_dinoalign_policy_is_independently_registered():
    config = make_policy_config("starvla_groot_dinoalign")

    assert isinstance(config, StarvlaGrootDinoAlignConfig)
    assert config.type == "starvla_groot_dinoalign"
    assert get_policy_class(config.type).__name__ == "StarvlaGrootDinoAlignPolicy"
    assert config.dinov3_model_name == "vit_base_patch16_dinov3"
    assert config.dinov3_input_size == 256
    assert config.dinov3_checkpoint.endswith("vit_base_patch16_dinov3.lvd1689m")


def test_dinoalign_does_not_import_starvla_groot_implementation():
    package = Path("vtla/frameworks/starvla_groot_dinoalign")
    python_sources = "\n".join(path.read_text() for path in package.rglob("*.py"))

    assert "..starvla_groot." not in python_sources


def _train_config(tmp_path, policy, preset="none", color_temp=(0.0, 0.0)):
    policy.push_to_hub = False
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="test/dataset",
            image_transforms=ImageTransformsConfig(preset=preset, color_temp=color_temp),
        ),
        policy=policy,
        output_dir=tmp_path / "output",
    )


def test_dinoalign_accepts_only_unaugmented_dataset_config(tmp_path, monkeypatch):
    monkeypatch.setattr("vtla.engine.configs.train.parser.get_path_arg", lambda _: None)
    config = _train_config(
        tmp_path,
        StarvlaGrootDinoAlignConfig(device="cpu"),
        preset="none",
        color_temp=(0.0, 0.0),
    )

    config.validate()


@pytest.mark.parametrize(
    ("preset", "color_temp"),
    [
        ("mild", (0.0, 0.0)),
        ("strong", (0.0, 0.0)),
        ("none", None),
        ("none", (-0.2, 0.2)),
    ],
)
def test_dinoalign_rejects_dataset_augmentation(
    tmp_path, monkeypatch, preset, color_temp
):
    monkeypatch.setattr("vtla.engine.configs.train.parser.get_path_arg", lambda _: None)
    config = _train_config(
        tmp_path,
        StarvlaGrootDinoAlignConfig(device="cpu"),
        preset=preset,
        color_temp=color_temp,
    )

    with pytest.raises(
        ValueError,
        match="requires augmentation_mode=none and COLOR_TEMP_RANGE",
    ):
        config.validate()


def test_original_starvla_allows_dataset_augmentation(tmp_path, monkeypatch):
    monkeypatch.setattr("vtla.engine.configs.train.parser.get_path_arg", lambda _: None)
    config = _train_config(
        tmp_path,
        StarvlaGrootConfig(device="cpu"),
        preset="strong",
        color_temp=(-0.2, 0.2),
    )

    config.validate()


def test_illumination_augment_preserves_shape_and_range():
    augment = IlluminationAugment(
        probability=1.0,
        brightness_range=(0.6, 1.4),
        contrast_range=(0.7, 1.3),
        gamma_range=(0.7, 1.5),
        shadow_probability=1.0,
        shadow_strength_range=(0.2, 0.6),
    ).train()
    images = torch.rand(2, 3, 3, 32, 32)

    augmented = augment(images)

    assert augmented.shape == images.shape
    assert augmented.dtype == images.dtype
    assert augmented.min() >= 0
    assert augmented.max() <= 1
    assert not torch.equal(augmented, images)


def test_alignment_head_returns_per_sample_losses_and_student_gradients():
    head = DinoAlignmentHead(qwen_hidden_size=8, dino_hidden_size=6)
    qwen_tokens = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    dino_tokens = torch.randn(2, 3, 4, 6)

    total, patch, global_loss = head(qwen_tokens, dino_tokens, global_loss_weight=0.2)
    total.mean().backward()

    assert total.shape == patch.shape == global_loss.shape == (2,)
    assert qwen_tokens.grad is not None
    assert dino_tokens.grad is None
