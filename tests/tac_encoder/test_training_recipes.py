from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vtla.tac_encoder.frameworks.sparsh_vjepa.training import (
    SparshVJEPATrainingModel,
    SparshVJEPATrainingRecipe,
    _load_full_pretrained,
)
from vtla.tac_encoder.common.training import WarmupCosineScheduler


def _tiny_vjepa() -> SparshVJEPATrainingModel:
    return SparshVJEPATrainingModel(
        num_frames=4,
        image_size=32,
        embed_dim=32,
        depth=1,
        num_heads=4,
        predictor_dim=32,
        predictor_depth=1,
        predictor_heads=4,
        checkpoint_source_grid=(2, 2, 2),
    )


def test_vjepa_rejects_incomplete_encoder_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "encoder_only.pth"
    torch.save({"blocks.0.norm1.weight": torch.ones(32)}, checkpoint)
    with pytest.raises(ValueError, match="encoder checkpoint is incomplete"):
        _load_full_pretrained(_tiny_vjepa(), str(checkpoint))


def test_vjepa_initializes_both_encoders_and_leaves_predictor_from_scratch(
    tmp_path: Path,
) -> None:
    source = _tiny_vjepa()
    encoder = {
        key: value.clone() for key, value in source.context_encoder.backbone.state_dict().items()
    }
    checkpoint = tmp_path / "encoder_only.pth"
    torch.save(encoder, checkpoint)

    model = _tiny_vjepa()
    predictor_before = {
        key: value.clone() for key, value in model.predictor.state_dict().items()
    }
    report = _load_full_pretrained(model, str(checkpoint))

    assert report["initialization"] == "encoder_only"
    assert report["predictor_init"] == "scratch"
    assert report["loaded_tensors"] == 2 * len(encoder)
    for key, expected in encoder.items():
        torch.testing.assert_close(model.context_encoder.backbone.state_dict()[key], expected)
        torch.testing.assert_close(model.target_encoder.backbone.state_dict()[key], expected)
    for key, expected in predictor_before.items():
        torch.testing.assert_close(model.predictor.state_dict()[key], expected)


def test_vjepa_full_checkpoint_gradients_and_ema(tmp_path: Path) -> None:
    source = _tiny_vjepa()
    checkpoint = tmp_path / "full_vjepa.pth"
    torch.save(
        {"state_dict": {f"model.{key}": value.clone() for key, value in source.state_dict().items()}},
        checkpoint,
    )
    model = _tiny_vjepa()
    report = _load_full_pretrained(model, str(checkpoint))
    assert report["loaded_tensors"] == len(model.state_dict())
    assert report["predictor_init"] == "checkpoint"

    output = model(torch.rand(2, 1, 4, 3, 32, 32))
    output.loss.backward()
    assert model.context_encoder.backbone.patch_embed.proj.weight.grad is not None
    assert model.predictor.backbone.input_projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    optimizer.step()
    before = model.target_encoder.backbone.patch_embed.proj.weight.detach().clone()
    SparshVJEPATrainingRecipe().after_optimizer_step(model, step=1, total_steps=10)
    after = model.target_encoder.backbone.patch_embed.proj.weight.detach()
    assert not torch.equal(before, after)


def test_vjepa_saves_latent_prediction_target_and_error_map(tmp_path: Path) -> None:
    model = _tiny_vjepa()
    dataset = [{"images": torch.rand(1, 4, 3, 32, 32)}]
    destination = tmp_path / "latent.png"
    SparshVJEPATrainingRecipe().save_visualization(
        model,
        dataset,
        [0],
        destination,
        torch.device("cpu"),
        args=None,
        autocast_dtype=None,
    )
    assert destination.is_file()
    assert destination.stat().st_size > 1000


def test_per_update_scheduler_restores_lr_and_cosine_weight_decay() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "weight_decay": 0.04, "WD_exclude": False}], lr=1e-3
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=4,
        warmup_steps=1,
        start_lr=0.0,
        final_lr=1e-6,
        final_weight_decay=0.4,
    )
    assert optimizer.param_groups[0]["lr"] == 0.0
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["weight_decay"] > 0.04

    state = scheduler.state_dict()
    restored = WarmupCosineScheduler(
        optimizer, total_steps=1, warmup_steps=0, start_lr=0.0, final_lr=0.0
    )
    restored.load_state_dict(state)
    assert restored.get_last_lr() == pytest.approx(scheduler.get_last_lr())
