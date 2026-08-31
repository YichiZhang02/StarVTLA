import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vtla.engine.configs import FeatureType, PolicyFeature, PreTrainedConfig
from vtla.engine.types import TransitionKey
from vtla.engine.utils.constants import ACTION, OBS_STATE
from vtla.frameworks.dream_tac.configuration_dream_tac import DreamTacConfig
from vtla.frameworks.dream_tac.modeling_dream_tac import DreamTacPolicy
from vtla.frameworks.dream_tac.processor_dream_tac import DreamTacPrepareBatchStep
from vtla.frameworks.dream_tac.runtime import (
    build_cosmos_experiment_opts,
    resolve_cosmos_text_assets,
)
from vtla.frameworks.dream_tac.slot_layout import compile_slot_layout
from vtla.frameworks.factory import _is_dream_tac_policy_checkpoint


def _write_cache(path, task="pick", value=1.0):
    path.mkdir(parents=True)
    save_file(
        {
            "context.0": torch.full((2, 3), value, dtype=torch.bfloat16),
            "mask.0": torch.ones(2, dtype=torch.bool),
        },
        str(path / "embeddings.safetensors"),
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "world_model": "dream_tac",
                "context_length": 2,
                "embedding_dim": 3,
                "prompt_template": "{task}",
                "text_encoder_model_hash": "same",
                "tasks": [{"task": task, "slot": 0, "task_index": 0}],
            }
        ),
        encoding="utf-8",
    )


def _prepare_step(cache, *, wrist_only=False, wrist_keys=None, top_keys=None, tactile_keys=None):
    wrist_keys = wrist_keys or ["observation.images.wrist"]
    top_keys = top_keys or ["observation.images.top"]
    tactile_keys = tactile_keys or ["observation.images.left", "observation.images.right"]
    layout = compile_slot_layout(
        wrist_only=wrist_only,
        wrist_camera_keys=wrist_keys,
        top_camera_keys=top_keys,
        tactile_mode="as_image",
        tactile_keys=tactile_keys,
        state_mode="absolute_joint",
    )
    return DreamTacPrepareBatchStep(
        rgb_keys=list(layout.rgb_keys),
        tactile_keys=list(layout.tactile_keys),
        layout_version=layout.version,
        layout_records=layout.records(),
        layout_fingerprint=layout.fingerprint,
        temporal_compression_factor=4,
        state_dim=5,
        action_dim=4,
        image_size=8,
        context_len=2,
        text_dim=3,
        prompt_template="{task}",
        text_embedding_cache_dirs=[str(cache)],
    )


def test_config_allows_starvtla_behavior_modes():
    kwargs = dict(
        tactile_mode="as_image",
        wrist_only=True,
        state_mode="absolute_joint",
        action_mode="relative_joint",
        action_gap=3,
        ee_num_arms=2,
    )
    config = DreamTacConfig(**kwargs)
    assert config.future_prediction_offset == 23
    assert config.slot_layout().state_t == 10
    assert config.slot_layout().pixel_frames == 37
    with pytest.raises(ValueError, match="chunk_size=20"):
        DreamTacConfig(**kwargs, chunk_size=16)
    no_tactile = DreamTacConfig(**{**kwargs, "tactile_mode": "none"})
    assert no_tactile.slot_layout().state_t == 6
    with pytest.raises(ValueError, match="does not support tactile_mode='encode'"):
        DreamTacConfig(**{**kwargs, "tactile_mode": "encode", "tactile_encoder_config": {}})


def test_layout_compiles_dual_wrist_and_four_tactile_slots():
    layout = compile_slot_layout(
        wrist_only=True,
        wrist_camera_keys=["left_wrist", "right_wrist"],
        top_camera_keys=["top"],
        tactile_mode="as_image",
        tactile_keys=["left_0", "left_1", "right_0", "right_1"],
        state_mode="absolute_rot6d",
    )
    assert layout.state_t == 16
    assert layout.pixel_frames == 61
    assert layout.action_index == 8
    assert layout.num_conditional_frames == 8
    assert layout.tactile_indices == (4, 5, 6, 7, 12, 13, 14, 15)


def test_checkpoint_layout_fingerprint_rejects_sensor_changes():
    config = DreamTacConfig(
        wrist_only=True,
        tactile_mode="none",
        wrist_camera_keys=["left_wrist"],
    )
    with pytest.raises(ValueError, match="differs from the checkpoint contract"):
        DreamTacConfig(
            wrist_only=True,
            tactile_mode="none",
            wrist_camera_keys=["left_wrist", "right_wrist"],
            slot_layout_fingerprint=config.slot_layout_fingerprint,
        )


def test_pretrained_checkpoint_layout_rejects_cli_sensor_changes():
    checkpoint_config = DreamTacConfig(
        wrist_only=True,
        tactile_mode="none",
        wrist_camera_keys=["left_wrist"],
    )
    current_config = DreamTacConfig(
        wrist_only=True,
        tactile_mode="none",
        wrist_camera_keys=["left_wrist", "right_wrist"],
    )
    with pytest.raises(ValueError, match="differs from the pretrained checkpoint"):
        current_config.validate_checkpoint_layout(checkpoint_config)


def test_checkpoint_config_round_trip_preserves_layout(tmp_path):
    config = DreamTacConfig(
        wrist_only=True,
        tactile_mode="as_image",
        state_mode="absolute_rot6d",
        wrist_camera_keys=["left_wrist", "right_wrist"],
        tactile_keys=["left_0", "left_1", "right_0", "right_1"],
    )
    config.save_pretrained(tmp_path, push_to_hub=False)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(restored, DreamTacConfig)
    assert restored.slot_layout().state_t == 16
    assert restored.slot_layout_fingerprint == config.slot_layout_fingerprint


def test_pretrained_path_distinguishes_cosmos_from_starvtla_checkpoint(tmp_path):
    cosmos = tmp_path / "cosmos"
    cosmos.mkdir()
    (cosmos / "model-480p-16fps.pt").touch()
    assert not _is_dream_tac_policy_checkpoint(cosmos)

    policy_dir = tmp_path / "policy"
    config = DreamTacConfig(pretrained_path=cosmos)
    config.save_pretrained(policy_dir, push_to_hub=False)
    save_file({"core.scale": torch.ones(1)}, str(policy_dir / "model.safetensors"))
    assert _is_dream_tac_policy_checkpoint(policy_dir)

    restored = PreTrainedConfig.from_pretrained(policy_dir)
    assert restored.pretrained_path == cosmos


def test_cosmos_text_assets_are_derived_from_pretrained_path(tmp_path):
    cosmos = tmp_path / "cosmos"
    (cosmos / "text_encoder").mkdir(parents=True)
    (cosmos / "tokenizer").mkdir()
    (cosmos / "model-480p-16fps.pt").touch()
    (cosmos / "text_encoder/config.json").write_text("{}", encoding="utf-8")
    (cosmos / "text_encoder/model.safetensors").touch()
    (cosmos / "tokenizer/tokenizer_config.json").write_text("{}", encoding="utf-8")
    (cosmos / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")

    encoder, tokenizer = resolve_cosmos_text_assets(str(cosmos))

    assert encoder == (cosmos / "text_encoder").resolve()
    assert tokenizer == (cosmos / "tokenizer").resolve()


def test_cosmos_text_assets_report_missing_components(tmp_path):
    cosmos = tmp_path / "cosmos"
    cosmos.mkdir()
    (cosmos / "model-480p-16fps.pt").touch()

    with pytest.raises(FileNotFoundError, match="missing bundled T5 assets"):
        resolve_cosmos_text_assets(str(cosmos))


def test_processor_packs_original_12_slot_sequence_and_dynamic_actions(tmp_path):
    cache = tmp_path / "cache"
    _write_cache(cache)
    step = _prepare_step(cache)
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.images.top": torch.rand(1, 2, 3, 8, 8),
            "observation.images.wrist": torch.rand(1, 2, 3, 8, 8),
            "observation.images.left": torch.rand(1, 3, 3, 8, 8),
            "observation.images.right": torch.rand(1, 3, 3, 8, 8),
            OBS_STATE: torch.rand(1, 2, 5),
        },
        TransitionKey.ACTION: torch.rand(1, 20, 4),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick"]},
    }
    output = step(transition)[TransitionKey.COMPLEMENTARY_DATA]
    assert output["video"].shape == (1, 3, 45, 8, 8)
    assert output["video"].dtype == torch.uint8
    assert output["actions"].shape == (1, 20, 4)
    assert output["action_latent_idx"].item() == 6
    assert output["future_tactile_right_latent_idx"].item() == 11
    assert output["t5_text_mask"].all()
    assert 0.15 <= output["tactile_self_attn_gate"].item() <= 1.0

    restored_config = step.get_config()
    restored_config["text_embedding_cache_dirs"] = [str(tmp_path / "missing")]
    restored = DreamTacPrepareBatchStep(**restored_config)
    restored.load_state_dict(step.state_dict())
    assert restored.task_to_slot == {"pick": 0}


def test_processor_wrist_only_omits_top_slots(tmp_path):
    cache = tmp_path / "cache"
    _write_cache(cache)
    step = _prepare_step(cache, wrist_only=True)
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.images.wrist": torch.rand(1, 2, 3, 8, 8),
            "observation.images.left": torch.rand(1, 3, 3, 8, 8),
            "observation.images.right": torch.rand(1, 3, 3, 8, 8),
            OBS_STATE: torch.rand(1, 2, 5),
        },
        TransitionKey.ACTION: torch.rand(1, 20, 4),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick"]},
    }
    output = step(transition)[TransitionKey.COMPLEMENTARY_DATA]
    assert output["video"].shape[2] == 37
    assert output["action_latent_idx"].item() == 5
    assert output["future_rgb_latent_indices"].tolist() == [[7]]


def test_processor_omits_state_and_tactile_slots(tmp_path):
    cache = tmp_path / "cache"
    _write_cache(cache)
    layout = compile_slot_layout(
        wrist_only=True,
        wrist_camera_keys=["wrist"],
        top_camera_keys=[],
        tactile_mode="none",
        tactile_keys=["unused"],
        state_mode="none",
    )
    step = DreamTacPrepareBatchStep(
        rgb_keys=["wrist"],
        tactile_keys=[],
        layout_version=layout.version,
        layout_records=layout.records(),
        layout_fingerprint=layout.fingerprint,
        temporal_compression_factor=4,
        state_dim=None,
        action_dim=4,
        image_size=8,
        context_len=2,
        text_dim=3,
        prompt_template="{task}",
        text_embedding_cache_dirs=[str(cache)],
    )
    transition = {
        TransitionKey.OBSERVATION: {"wrist": torch.rand(1, 2, 3, 8, 8)},
        TransitionKey.ACTION: torch.rand(1, 20, 4),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick"]},
    }
    output = step(transition)[TransitionKey.COMPLEMENTARY_DATA]
    assert output["video"].shape == (1, 3, 13, 8, 8)
    assert output["current_proprio_latent_idx"].item() == -1
    assert output["future_proprio_latent_idx"].item() == -1
    assert output["future_tactile_latent_indices"].shape == (1, 0)


def _policy_config(
    state_dim=10,
    action_dim=10,
    state_mode="absolute_rot6d",
    action_mode="absolute_rot6d",
    ee_num_arms=1,
):
    visual = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8))
    state = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    action = PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
    config = DreamTacConfig(
        tactile_mode="as_image",
        wrist_only=False,
        state_mode=state_mode,
        action_mode=action_mode,
        action_gap=0,
        ee_num_arms=ee_num_arms,
        top_camera_keys=["top"],
        wrist_camera_keys=["wrist"],
        tactile_keys=["left", "right"],
    )
    config.input_features = {"top": visual, "wrist": visual, "left": visual, "right": visual, OBS_STATE: state}
    config.output_features = {ACTION: action}
    return config


class _FakeCore(nn.Module):
    def __init__(self, state_t=12, pixel_frames=45):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.state_t = state_t
        self.pixel_frames = pixel_frames

    def training_step(self, batch, iteration):
        del iteration
        values = torch.arange(self.state_t, device=self.scale.device, dtype=torch.float32)
        per_frame = values.unsqueeze(0).expand(batch["video"].shape[0], -1) * self.scale
        return {"edm_loss_per_frame": per_frame}, per_frame.mean()

    def generate_samples_from_batch(self, batch, **kwargs):
        del kwargs
        return torch.zeros(batch["video"].shape[0], 16, self.state_t, 4, 4, device=self.scale.device)

    def decode(self, latent):
        return torch.zeros(latent.shape[0], 3, self.pixel_frames, 8, 8, device=latent.device)


@pytest.mark.parametrize(
    ("state_mode", "action_mode", "state_dim", "action_dim", "ee_num_arms"),
    [
        ("absolute_joint", "absolute_joint", 16, 16, 2),
        ("absolute_rot6d", "relative_rot6d", 10, 10, 1),
        ("absolute_quat", "absolute_quat", 16, 16, 2),
    ],
)
def test_policy_accepts_dynamic_starvtla_state_action_dimensions(
    state_mode, action_mode, state_dim, action_dim, ee_num_arms
):
    policy = DreamTacPolicy(
        _policy_config(state_dim, action_dim, state_mode, action_mode, ee_num_arms),
        core_model=_FakeCore(),
    )
    assert policy.action_dim == action_dim


def test_policy_weights_per_slot_edm_losses():
    policy = DreamTacPolicy(_policy_config(), core_model=_FakeCore())
    batch = {"video": torch.zeros(2, 3, 45, 8, 8)}
    for name, slot in {
        "action_latent_idx": 6,
        "future_proprio_latent_idx": 7,
        "future_wrist_image_latent_idx": 8,
        "future_image_latent_idx": 9,
        "future_tactile_left_latent_idx": 10,
        "future_tactile_right_latent_idx": 11,
    }.items():
        batch[name] = torch.full((2,), slot)
    batch["future_rgb_latent_indices"] = torch.tensor([[8, 9], [8, 9]])
    batch["future_tactile_latent_indices"] = torch.tensor([[10, 11], [10, 11]])
    loss, metrics = policy(batch)
    torch.testing.assert_close(loss, torch.tensor(32.75))
    assert metrics["action_loss"].item() == 6
    assert metrics["video_loss"].item() == 8.5


def test_action_latent_decode_returns_starvtla_rot6d():
    action_slot = _policy_config().slot_layout().action_index
    latent = torch.zeros(1, 16, 12, 4, 4)
    normalized = torch.linspace(-0.5, 0.5, 200).reshape(20, 10)
    flat = normalized.flatten().repeat((16 * 4 * 4) // 200 + 1)[: 16 * 4 * 4]
    latent[0, :, action_slot] = flat.reshape(16, 4, 4)
    extracted = DreamTacPolicy.extract_action_chunk(
        latent, torch.tensor([action_slot]), action_dim=10
    )
    torch.testing.assert_close(extracted[0], normalized, atol=1e-6, rtol=1e-6)
    assert extracted.shape == (1, 20, 10)


def test_training_visualization_writes_four_modalities_and_metrics(tmp_path):
    policy = DreamTacPolicy(_policy_config(), core_model=_FakeCore())
    sample = {
        "video": torch.zeros(1, 3, 45, 8, 8, dtype=torch.uint8),
        "actions": torch.zeros(1, 20, 10),
        "action_latent_idx": torch.tensor([_policy_config().slot_layout().action_index]),
    }
    summary = policy.generate_training_visualizations([sample], tmp_path, step=10)
    assert set(summary["samples"][0]["image_paths"]) == {
        "future_rgb:wrist",
        "future_rgb:top",
        "future_tactile:left",
        "future_tactile:right",
    }
    assert (tmp_path / "visualizations" / "step_000010" / "metrics.json").is_file()
    assert summary["aggregate"]["mean_action_mae_normalized"] == 0.0


def test_runtime_options_follow_compiled_layout(tmp_path):
    layout = compile_slot_layout(
        wrist_only=True,
        wrist_camera_keys=["left_wrist", "right_wrist"],
        top_camera_keys=[],
        tactile_mode="as_image",
        tactile_keys=["left_0", "left_1", "right_0", "right_1"],
        state_mode="absolute_rot6d",
    )
    opts = build_cosmos_experiment_opts(layout, tmp_path)
    assert "model.config.state_t=16" in opts
    assert "model.config.min_num_conditional_frames=8" in opts
    assert "model.config.tokenizer.chunk_duration=61" in opts
    assert "++model.config.net.tactile_latent_t_indices=[4,5,6,7,12,13,14,15]" in opts
