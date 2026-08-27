from __future__ import annotations

import torch

from vtla.tac_encoder.models.base import EncodedFeatures
from vtla.tac_encoder.models.checkpoint import interpolate_video_position_embedding
from vtla.tac_encoder.models.pooling import pool_encoded_features
from vtla.tac_encoder.models.registry import build_backbone


def test_anytouch1_pooling_contract_is_40_tokens_per_sensor() -> None:
    features = EncodedFeatures(
        global_tokens=torch.randn(2, 2, 4, 32),
        spatial_grid=torch.randn(2, 2, 4, 16, 16, 32),
        global_time_ids=torch.arange(4),
        interleave_global=True,
    )
    result = pool_encoded_features(features, pool_size=3)
    assert result.tokens.shape == (2, 80, 32)
    assert result.sensor_ids[0].tolist() == [0] * 40 + [1] * 40
    assert result.time_ids[0].tolist() == sum(([time] * 10 for time in range(4)), []) * 2


def test_anytouch2_joint_reconstruction_and_temporal_pooling() -> None:
    model = build_backbone(
        "anytouch2",
        image_size=32,
        embed_dim=48,
        projection_dim=48,
        depth=1,
        num_heads=4,
        decoder_dim=32,
        decoder_depth=1,
        decoder_heads=4,
    )
    images = torch.rand(1, 2, 4, 3, 32, 32)
    output = model(images, 0.75)
    assert output.reconstruction.shape == images.shape
    assert output.residual_prediction.shape == (1, 2, 3, 3, 32, 32)
    assert output.residual_target.shape == output.residual_prediction.shape
    assert output.mask.shape == (1, 2, 2, 2, 2)
    assert torch.equal(output.mask[:, :, 0], output.mask[:, :, 1])
    output.loss.backward()
    assert model.decoder_pred_video.weight.grad is not None
    assert model.diff_decoder_pred_video.weight.grad is not None
    pooled = model.extract_pooled_features(images, pool_size=3)
    assert pooled.tokens.shape == (1, 38, 48)
    assert pooled.sensor_ids[0].tolist() == [0] * 19 + [1] * 19


def test_sparsh_downstream_backbone_is_encoder_only() -> None:
    model = build_backbone(
        "sparsh_vjepa", image_size=32, embed_dim=48, depth=1, num_heads=4
    )
    assert not any("decoder" in key or "predictor" in key for key in model.state_dict())
    pooled = model.extract_pooled_features(torch.rand(1, 2, 4, 3, 32, 32), pool_size=3)
    assert pooled.tokens.shape == (1, 38, 48)
    assert pooled.sensor_ids[0].tolist() == [0] * 19 + [1] * 19


def test_spatiotemporal_position_interpolation_preserves_shape_contract() -> None:
    source = torch.randn(1, 2 * 20 * 15, 24)
    target = interpolate_video_position_embedding(
        source,
        source_grid=(2, 20, 15),
        target_grid=(2, 14, 14),
        has_cls=False,
    )
    assert target.shape == (1, 2 * 14 * 14, 24)


def test_anytouch2_loads_released_encoder_and_decoder_names(tmp_path) -> None:
    config = dict(
        image_size=32,
        embed_dim=48,
        projection_dim=32,
        depth=1,
        num_heads=4,
        decoder_dim=32,
        decoder_depth=2,
        decoder_heads=4,
    )
    source = build_backbone("anytouch2", **config)
    released = {
        f"touch_mae_model.{key}": value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("normalization_")
    }
    checkpoint = tmp_path / "anytouch2_released.pth"
    torch.save(released, checkpoint)

    torch.manual_seed(1234)
    target = build_backbone("anytouch2", **config)
    report = target.load_pretrained(checkpoint)

    required_prefixes = (
        "touch_model.",
        "touch_projection.",
        "sensor_token",
        "decoder_embed.",
        "decoder_pos_embed",
        "touch_decoder_blocks.",
        "decoder_norm.",
        "decoder_pred_video.",
        "mask_token",
        "decoder_embed_diff.",
        "diff_touch_decoder_blocks.",
        "diff_decoder_norm.",
        "diff_decoder_pred_video.",
        "mask_token_diff",
    )
    assert not [key for key in report["missing_keys"] if key.startswith(required_prefixes)]
    assert report["shape_mismatch"] == {}
    assert torch.equal(target.decoder_embed.weight, source.decoder_embed.weight)
    assert torch.equal(
        target.touch_decoder_blocks[1].self_attn.q_proj.weight,
        source.touch_decoder_blocks[1].self_attn.q_proj.weight,
    )
    assert torch.equal(target.decoder_pred_video.bias, source.decoder_pred_video.bias)
    assert torch.equal(target.decoder_embed_diff.weight, source.decoder_embed_diff.weight)
    assert torch.equal(
        target.diff_touch_decoder_blocks[1].self_attn.q_proj.weight,
        source.diff_touch_decoder_blocks[1].self_attn.q_proj.weight,
    )
    assert torch.equal(target.diff_decoder_pred_video.bias, source.diff_decoder_pred_video.bias)


def test_sparsh_loads_released_bare_encoder_state_dict(tmp_path) -> None:
    source = build_backbone(
        "sparsh_vjepa",
        image_size=32,
        embed_dim=48,
        depth=1,
        num_heads=4,
    )
    released = {
        key.removeprefix("encoder."): value
        for key, value in source.state_dict().items()
        if key.startswith("encoder.")
    }
    checkpoint = tmp_path / "vjepa_released.pth"
    torch.save(released, checkpoint)

    target = build_backbone(
        "sparsh_vjepa",
        image_size=32,
        embed_dim=48,
        depth=1,
        num_heads=4,
    )
    report = target.load_pretrained(checkpoint)

    assert not [key for key in report["missing_keys"] if key.startswith("encoder.")]
    assert report["shape_mismatch"] == {}
    assert report["unexpected_keys"] == []
