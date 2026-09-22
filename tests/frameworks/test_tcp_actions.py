from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from vtla.engine.utils.ee_transforms import (
    matrix_to_rot6d, encode_relative_tcp, decode_relative_tcp,
    ee_to_relative, ee_to_absolute,
)
from vtla.engine.processor.relative_action_processor import (
    ACTION_ANCHOR, route_ee_batch, RelativeActionsProcessorStep, AbsoluteActionsProcessorStep,
)
from vtla.engine.types import TransitionKey
from vtla.engine.utils.constants import ACTION, OBS_STATE
from vtla.frameworks.ee_processor_utils import remap_ee_dataset_stats
from vtla.datasets.tcp_contract import build_tcp_contract, validate_tcp_contract


def poses(shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.tensor(Rotation.random(int(np.prod(shape)), random_state=seed).as_matrix()).reshape(*shape, 3, 3)
    return torch.cat((torch.randn(*shape, 3, generator=generator, dtype=torch.float64),
                      matrix_to_rot6d(matrix), torch.rand(*shape, 1, generator=generator, dtype=torch.float64)), -1).flatten(-2)


@pytest.mark.parametrize('arms', [1, 2])
def test_local_tcp_roundtrip_and_identity(arms):
    anchor = poses((4, arms))
    target = poses((4, 7, arms), 12)
    delta = encode_relative_tcp(anchor, target, arms)
    torch.testing.assert_close(decode_relative_tcp(anchor, delta, arms), target)
    stationary = encode_relative_tcp(anchor, anchor, arms).reshape(4, arms, 10)
    torch.testing.assert_close(stationary[..., :9], torch.zeros_like(stationary[..., :9]), atol=1e-12, rtol=0)
    # Geometric composition remains independent of the zero-centered action format.
    torch.testing.assert_close(ee_to_absolute(anchor, ee_to_relative(anchor, target, arms), arms), target)


def test_translation_is_in_current_tcp_axes():
    rotation = torch.tensor(Rotation.from_euler('z', 90, degrees=True).as_matrix())
    anchor = torch.cat((torch.zeros(3), matrix_to_rot6d(rotation), torch.tensor([0.2]))).unsqueeze(0)
    target = anchor.clone()
    target[:, 0] += 0.01
    target[:, 9] = 0.8
    delta = encode_relative_tcp(anchor, target, 1)
    torch.testing.assert_close(delta[0, :3], delta.new_tensor([0, -0.01, 0]), atol=1e-12, rtol=0)
    torch.testing.assert_close(delta[0, 3:9], torch.zeros(6, dtype=delta.dtype), atol=1e-12, rtol=0)
    assert delta[0, 9] == 0.8


@pytest.mark.parametrize('state_mode', ['none', 'absolute_rot6d', 'episode_rot6d', 'absolute_joint'])
def test_hidden_anchor_and_locked_chunk(state_mode):
    anchor, target = poses((1, 1)), poses((1, 3, 1), 19)
    raw = {OBS_STATE: torch.zeros(1, 8), OBS_STATE + '_absolute_ee': anchor,
           OBS_STATE + '_episode_ee': anchor, ACTION + '_absolute_ee': target}
    routed = route_ee_batch(raw, state_mode, 'relative_rot6d')
    step = RelativeActionsProcessorStep(enabled=True, mode='pose', n_arms=1)
    encoded = step({TransitionKey.OBSERVATION: routed, TransitionKey.ACTION: routed[ACTION]})
    step.lock_action_anchor()
    step({TransitionKey.OBSERVATION: {ACTION_ANCHOR: poses((1, 1), 40)}})
    restored = AbsoluteActionsProcessorStep(enabled=True, relative_step=step)(encoded)[TransitionKey.ACTION]
    torch.testing.assert_close(restored, target)
    # Serialized settings contain no coordinate-selection switch.
    assert 'rot_mode' not in step.get_config()
    assert 'relative_encoding' not in step.get_config()


def test_zero_prediction_and_degenerate_rotation_hold_anchor():
    anchor = poses((1, 1))
    delta = torch.zeros_like(anchor)
    delta[:, 9] = anchor[:, 9]
    torch.testing.assert_close(decode_relative_tcp(anchor, delta, 1), anchor)
    delta[:, 3:9] = delta.new_tensor([-1, 0, 0, 0, -1, 0])
    torch.testing.assert_close(decode_relative_tcp(anchor, delta, 1), anchor)
    delta[:, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        decode_relative_tcp(anchor, delta, 1)


def test_stats_and_contract_reject_old_semantics():
    config = SimpleNamespace(state_mode='none', action_mode='relative_rot6d')
    with pytest.raises(KeyError):
        remap_ee_dataset_stats({'action_relative_ee_se3': {}}, config)
    contract = build_tcp_contract('umi', 3, 2)
    validate_tcp_contract(contract, offsets=[2, 3, 4], robot_type='umi')
    with pytest.raises(ValueError, match='offsets'):
        validate_tcp_contract(contract, offsets=[0, 1, 2])
    with pytest.raises(ValueError, match='obsolete'):
        validate_tcp_contract(None)


def test_rot6d_only_modes():
    from vtla.frameworks.act.configuration_act import ACTConfig
    for action in ('absolute_rot6d', 'relative_rot6d', 'absolute_joint', 'relative_joint'):
        cfg = ACTConfig(action_mode=action, device='cpu')
        assert not hasattr(cfg, 'ee_frame')
    for action in ('absolute_quat', 'relative_quat', 'relative_rot6d_se3', 'relative_rot6d_delta'):
        with pytest.raises(ValueError, match='action_mode'):
            ACTConfig(action_mode=action, device='cpu')


def test_processor_serialization_and_normalization_roundtrip(tmp_path):
    from vtla.frameworks.act.configuration_act import ACTConfig
    from vtla.frameworks.act.processor_act import make_act_pre_post_processors
    from vtla.frameworks.factory import _reconnect_relative_absolute_steps
    from vtla.engine.configs import FeatureType, PolicyFeature
    from vtla.engine.processor import PolicyProcessorPipeline, policy_action_to_transition, transition_to_policy_action
    config = ACTConfig(device='cpu', state_mode='none', action_mode='relative_rot6d', ee_num_arms=1,
                       input_features={}, output_features={ACTION: PolicyFeature(FeatureType.ACTION, (10,))})
    stats = {'action_relative_ee': {'mean': torch.linspace(-.3, .6, 10), 'std': torch.ones(10) * .5}}
    pre, post = make_act_pre_post_processors(config, stats)
    anchor = poses((1, 1)).float()
    target = poses((1, 3, 1), 5).float()
    prepared = pre({ACTION_ANCHOR: anchor, ACTION: target})
    torch.testing.assert_close(post(prepared[ACTION]), target, atol=1e-6, rtol=1e-5)
    pre.save_pretrained(tmp_path, config_filename='pre.json')
    post.save_pretrained(tmp_path, config_filename='post.json')
    pre2 = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename='pre.json')
    post2 = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename='post.json',
               to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    _reconnect_relative_absolute_steps(pre2, post2)
    prepared2 = pre2({ACTION_ANCHOR: anchor, ACTION: target})
    torch.testing.assert_close(prepared2[ACTION], prepared[ACTION])
    torch.testing.assert_close(post2(prepared2[ACTION]), target, atol=1e-6, rtol=1e-5)


def test_temporal_ensemble_decodes_each_tcp_anchor(monkeypatch):
    from vtla.engine.common.control_utils import predict_action
    from vtla.frameworks.act.modeling_act import ACTTemporalEnsembler
    monkeypatch.setattr('vtla.engine.common.control_utils.prepare_observation_for_inference',
                        lambda observation, *args: observation)
    step = RelativeActionsProcessorStep(enabled=True, mode='pose', n_arms=1)
    class Pre:
        steps = [step]
        def __call__(self, observation):
            step({TransitionKey.OBSERVATION: observation})
            return observation
    class Policy:
        config = SimpleNamespace(action_mode='relative_rot6d', temporal_ensemble_coeff=0.0)
        temporal_ensembler = ACTTemporalEnsembler(0.0, 2)
        def is_action_queue_empty(self):
            return True
        def predict_action_chunk(self, observation):
            return torch.zeros(1, 2, 10, dtype=torch.float64)
    policy = Policy()
    post = lambda value: decode_relative_tcp(step.get_cached_state(), value, 1)
    anchor = poses((1, 1))
    first = predict_action({ACTION_ANCHOR: anchor}, policy, torch.device('cpu'), Pre(), post, False)
    second_anchor = anchor.clone(); second_anchor[:, 0] += 0.2
    second = predict_action({ACTION_ANCHOR: second_anchor}, policy, torch.device('cpu'), Pre(), post, False)
    torch.testing.assert_close(first[:, :9], anchor[:, :9])
    torch.testing.assert_close(second[:, :3], (anchor[:, :3] + second_anchor[:, :3]) / 2)


def test_checkpoint_and_calibration_contracts(tmp_path):
    from vtla.frameworks.act.configuration_act import ACTConfig
    from vtla.engine.configs import PreTrainedConfig
    config = ACTConfig(device='cpu', state_mode='none', action_mode='relative_rot6d')
    config.save_pretrained(tmp_path, push_to_hub=False)
    with pytest.raises(ValueError, match='obsolete'):
        PreTrainedConfig.from_pretrained(tmp_path)
    config.tcp_contract = build_tcp_contract('rm_isf_umi_left', 32, 0)
    config.save_pretrained(tmp_path, push_to_hub=False)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert loaded.tcp_contract == config.tcp_contract
    assert not hasattr(loaded, 'ee_frame')
    loaded.tcp_contract['flange_tcp_calibration']['left'][0][0] += .01
    with pytest.raises(ValueError, match='calibration'):
        validate_tcp_contract(loaded.tcp_contract, robot_type='rm_isf_umi_left')


def test_episode_state_cannot_replace_missing_absolute_anchor():
    step = RelativeActionsProcessorStep(enabled=True, mode='pose', n_arms=1)
    with pytest.raises(KeyError, match='absolute TCP'):
        step({TransitionKey.OBSERVATION: {OBS_STATE: poses((1, 1))}})
