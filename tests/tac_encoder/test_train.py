from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from tests.tac_encoder.test_data import _write_cache
from vtla.tac_encoder.inference import TactileBackboneFeatureExtractor
from vtla.tac_encoder.models.registry import build_backbone
from vtla.tac_encoder.train import main
from vtla.frameworks.tactile_encode import TactileEncoder


def test_unified_train_entrypoint_and_checkpoint_inference(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "data"
    _write_cache(catalog / "tiny" / "tactile_backbone_cache", windows=2, image_size=32)
    output = tmp_path / "output"
    model_config = dict(
        image_size=32,
        embed_dim=32,
        projection_dim=32,
        depth=1,
        num_heads=4,
        decoder_dim=32,
        decoder_depth=1,
        decoder_heads=4,
    )
    source = build_backbone("anytouch2", **model_config)
    pretrained = tmp_path / "anytouch2_full.pth"
    torch.save(
        {f"touch_mae_model.{key}": value.clone() for key, value in source.state_dict().items()},
        pretrained,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--dataset_id",
            "tiny",
            "--model_id",
            "anytouch2",
            "--pretrained_path",
            str(pretrained),
            "--dataset_catalog_root",
            str(catalog),
            "--output_dir",
            str(output),
            "--image_size",
            "32",
            "--batch_size",
            "2",
            "--epochs",
            "1",
            "--num_workers",
            "0",
            "--device",
            "cpu",
            "--amp_dtype",
            "none",
            "--encoder_dim",
            "32",
            "--encoder_depth",
            "1",
            "--encoder_heads",
            "4",
            "--projection_dim",
            "32",
            "--decoder_dim",
            "32",
            "--decoder_depth",
            "1",
            "--decoder_heads",
            "4",
            "--eval_freq",
            "0",
        ],
    )
    main()
    assert (output / "last.pth").is_file()
    assert (output / "resolved_data_mixture.json").is_file()

    checkpoint = torch.load(output / "last.pth", map_location="cpu", weights_only=False)
    assert checkpoint["format_version"] == 2
    assert checkpoint["objective"] == "masked_pixel_and_temporal_residual_reconstruction"
    assert checkpoint["encoder"]
    assert checkpoint["trainer"]
    assert not any("decoder" in key or "mask_token" in key for key in checkpoint["encoder"])

    extractor = TactileBackboneFeatureExtractor.from_pretrained(output / "last.pth")
    assert not any(
        "decoder" in key or "mask_token" in key for key in extractor.backbone.state_dict()
    )
    result = extractor(torch.zeros(1, 1, 4, 3, 32, 32))
    assert result.tokens.shape == (1, 19, 32)

    class PolicyConfig:
        tactile_encoder_path = str(output / "last.pth")
        freeze_tactile_encoder = True
        tactile_num_frames = 4

        @staticmethod
        def tactile_encoder_keys():
            return ["finger0"]

    policy_encoder = TactileEncoder(PolicyConfig(), output_dim=16)
    # Policy datasets decode uint8 at their native resolution. The shared tactile
    # boundary converts to [0, 1] and resizes to the checkpoint's image size.
    projected = policy_encoder.forward_flat(
        {"finger0": torch.zeros(1, 4, 3, 40, 48, dtype=torch.uint8)}
    )
    assert projected.shape == (1, 19, 16)

    class Pool2PolicyConfig(PolicyConfig):
        tactile_pool_size = 2

    pool2_encoder = TactileEncoder(Pool2PolicyConfig(), output_dim=16)
    pool2_projected = pool2_encoder.forward_flat({"finger0": torch.zeros(1, 4, 3, 32, 32)})
    assert pool2_projected.shape == (1, 9, 16)


def test_policy_rejects_legacy_query_checkpoint(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.pth"
    torch.save({"model": {}}, legacy)

    class PolicyConfig:
        tactile_encoder_path = str(legacy)
        freeze_tactile_encoder = True
        tactile_num_frames = 4

        @staticmethod
        def tactile_encoder_keys():
            return ["finger0"]

    with pytest.raises(ValueError, match="Not a unified tactile backbone checkpoint"):
        TactileEncoder(PolicyConfig(), output_dim=16)
