from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vtla.engine.configs import FeatureType, PolicyFeature
from vtla.frameworks.factory import get_policy_class, make_policy_config, make_pre_post_processors
from vtla.frameworks.tacmind0.modeling_tacmind0 import TacMind0Policy
from vtla.train import _configure_tacmind0_fsdp, _validate_tacmind0_fsdp


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def set_trainable(self, value):
        self.requires_grad_(value)


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.tactile_encoder = _Encoder()

    def forward(self, *, action, action_mask, tactile_pixel_values, **kwargs):
        loss = (action * action_mask).sum() * self.tactile_encoder.weight
        loss = loss + tactile_pixel_values.mean() * self.tactile_encoder.weight
        return SimpleNamespace(loss=loss)

    def inference_action(self, *, tactile_pixel_values, **kwargs):
        batch = tactile_pixel_values.shape[0]
        return torch.ones(batch, 32, 32) * self.tactile_encoder.weight


class _Processor:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def apply_chat_template(self, messages, **kwargs):
        assert sum(part["type"] == "image" for part in messages[0]["content"]) == 2
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "token_type_ids": torch.zeros(1, 3, dtype=torch.long),
            "pixel_values": torch.zeros(2, 3, 42, 42),
        }


def _config():
    keys = ["observation.images.cam_finger0", "observation.images.cam_finger1"]
    visual = PolicyFeature(FeatureType.VISUAL, (3, 42, 42))
    return make_policy_config(
        "tacmind0", device="cpu", tactile_keys=keys,
        wrist_camera_keys=["observation.images.wrist"], top_camera_keys=[],
        state_mode="none", action_mode="absolute_joint", action_gap=6,
        input_features={**{key: visual for key in keys}, "observation.images.wrist": visual},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
    )


def test_tacmind0_infra_contract_and_trainable_encoder():
    config = _config()
    assert get_policy_class("tacmind0") is TacMind0Policy
    assert config.tactile_delta_indices() == [-35, -30, -25, -20, -15, -10, -5, 0]
    assert config.action_delta_indices == list(range(6, 38))
    policy = TacMind0Policy(config, core_model=_Core(), processor=_Processor())
    batch = {key: torch.rand(1, 8, 3, 42, 42) for key in config.tactile_keys}
    batch["observation.images.wrist"] = torch.rand(1, 3, 42, 42)
    batch["task"] = ["pick up object"]
    batch["action"] = torch.ones(1, 32, 8)
    loss, _ = policy(batch)
    loss.backward()
    assert policy.model.tactile_encoder.weight.grad is not None
    assert policy.model.tactile_encoder.weight.grad.abs() > 0
    assert policy._model_inputs(batch)["tactile_pixel_values"].shape == (1, 2, 8, 3, 42, 42)
    assert policy.predict_action_chunk(batch).shape == (1, 32, 8)
    assert policy.select_action(batch).shape == (1, 8)


def test_tacmind0_processors_do_not_load_pi05_tokenizer(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("TacMind0 must use its own local processor")

    monkeypatch.setattr(
        "vtla.engine.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        fail_if_called,
    )
    preprocessor, _ = make_pre_post_processors(_config())
    assert any(type(step).__name__ == "TactileTemporalWindowStep" for step in preprocessor.steps)
    assert not any(type(step).__name__ == "TokenizerProcessorStep" for step in preprocessor.steps)


def test_tacmind0_native_training_defaults():
    config = _config()
    assert config.freeze_vlm_embedding
    assert config.vlm_gradient_checkpointing
    assert config.ae_gradient_checkpointing
    assert config.get_optimizer_preset().weight_decay == 1e-10


def test_tacmind0_fsdp_uses_native_wrap_targets():
    class _NativeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.wrap_target = nn.Linear(1, 1)
            self.other = nn.Linear(1, 1)

        def fsdp_wrap_modules(self):
            return [self.wrap_target]

    plugin = SimpleNamespace(
        fsdp_version=1,
        auto_wrap_policy=None,
        transformer_cls_names_to_wrap=["old"],
        min_num_params=100,
    )
    accelerator = SimpleNamespace(
        num_processes=4,
        mixed_precision="bf16",
        state=SimpleNamespace(fsdp_plugin=plugin),
    )
    policy = SimpleNamespace(model=_NativeModel())

    _validate_tacmind0_fsdp(accelerator)
    _configure_tacmind0_fsdp(accelerator, policy)

    assert plugin.auto_wrap_policy(policy.model.wrap_target, False, 1)
    assert not plugin.auto_wrap_policy(policy.model.other, False, 1)
    assert plugin.auto_wrap_policy(policy.model.other, True, 1)
    assert plugin.transformer_cls_names_to_wrap is None
    assert plugin.min_num_params == 0


def test_tacmind0_rejects_ddp_and_single_process_training():
    ddp = SimpleNamespace(
        num_processes=4,
        mixed_precision="bf16",
        state=SimpleNamespace(fsdp_plugin=None),
    )
    single = SimpleNamespace(
        num_processes=1,
        state=SimpleNamespace(fsdp_plugin=None),
    )
    with pytest.raises(RuntimeError, match="FSDP-1"):
        _validate_tacmind0_fsdp(ddp)
    with pytest.raises(RuntimeError, match="at least two"):
        _validate_tacmind0_fsdp(single)
