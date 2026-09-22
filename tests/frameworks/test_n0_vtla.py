from vtla.datasets.tcp_contract import build_tcp_contract
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vtla.engine.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from vtla.engine.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from vtla.frameworks.n0_vtla.configuration_n0_vtla import N0VTLAConfig
from vtla.frameworks.n0_vtla.modeling_n0_vtla import N0VTLAPolicy
from vtla.frameworks.n0_vtla.runtime import load_native_weights, native_key
from vtla.frameworks.factory import get_policy_class, make_policy_config


def config(**overrides):
    args = dict(device='cpu', dtype='float32', chunk_size=3, n_action_steps=2,
                tactile_mode='as_image', wrist_only=True, wrist_camera_keys=['observation.images.wrist'],
                tactile_keys=['observation.images.touch'], state_mode='absolute_joint',
                action_mode='absolute_joint', image_resolution=(28, 28), tactile_image_size=28,
                max_action_dim=8, max_state_dim=8, predictor_n_heads=4, predictor_n_layers=1,
                n_latent=2, num_inference_steps=2, allow_random_init=True,
                dinov2_config=dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                                   intermediate_size=64, image_size=28, patch_size=14),
                input_features={
                    'observation.images.wrist': PolicyFeature(FeatureType.VISUAL, (3, 28, 28)),
                    'observation.images.touch': PolicyFeature(FeatureType.VISUAL, (3, 28, 28)),
                    OBS_STATE: PolicyFeature(FeatureType.STATE, (4,)),
                }, output_features={ACTION: PolicyFeature(FeatureType.ACTION, (4,))})
    args.update(overrides)
    return N0VTLAConfig(**args)


@pytest.fixture
def tiny_backbone(monkeypatch):
    # Real Gemma, SigLIP, DINOv2 and predictor operations, reduced widths/depths.
    import vtla.frameworks.pi05.modeling_pi05 as pi
    original_config = pi.CONFIG_MAPPING['paligemma']
    def small_paligemma():
        cfg = original_config()
        cfg.projection_dim = 32
        cfg.vision_config.hidden_size = 32
        cfg.vision_config.num_attention_heads = 4
        cfg.vision_config.num_hidden_layers = 1
        cfg.vision_config.patch_size = 14
        return cfg
    original_model = pi.PaliGemmaWithExpertModel
    class SmallBackbone(original_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.paligemma.model.multi_modal_projector.linear = nn.Linear(32, 32)
    monkeypatch.setattr(pi, 'PaliGemmaWithExpertModel', SmallBackbone)
    monkeypatch.setitem(pi.CONFIG_MAPPING._extra_content, 'paligemma', small_paligemma)
    monkeypatch.setattr(pi, 'get_gemma_config', lambda _: pi.GemmaConfig(32, 2, 64, 8, 1, 4))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old_threads)


def batch(c, b=2):
    return {
        'observation.images.wrist': torch.rand(b, 3, 28, 28),
        'observation.images.touch': torch.rand(b, 2, 3, 28, 28),
        OBS_STATE: torch.zeros(b, 4), ACTION: torch.rand(b, c.chunk_size, 4),
        OBS_LANGUAGE_TOKENS: torch.randint(2, 40, (b, 6)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(b, 6, dtype=torch.bool),
        'action_is_pad': torch.zeros(b, c.chunk_size, dtype=torch.bool),
    }


def test_registration_and_serialization(tmp_path):
    c = config()
    assert config(action_gap=6).drop_n_last_frames == 6
    assert get_policy_class('n0_vtla') is N0VTLAPolicy
    assert isinstance(make_policy_config('n0_vtla'), N0VTLAConfig)
    c.save_pretrained(tmp_path, push_to_hub=False)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(restored, N0VTLAConfig)
    assert restored.dinov2_config == c.dinov2_config
    with pytest.raises(ValueError, match='contract mismatch'):
        replace(c, action_mode='relative_joint').validate_checkpoint_layout(restored)
    with pytest.raises(ValueError, match='only supports'):
        config(tactile_mode='none')
    with pytest.raises(ValueError, match='owns'):
        config(tactile_num_frames=2)
    with pytest.raises(ValueError, match='exceeds'):
        config(max_action_dim=2).validate_features()




def test_native_dinov2_position_grid():
    from vtla.frameworks.n0_vtla.tactile_encoder import FrozenDINOv2TactileEncoder

    with torch.device('meta'):
        encoder = FrozenDINOv2TactileEncoder(llm_dim=2048)
    assert encoder.backbone.embeddings.position_embeddings.shape == (1, 1370, 768)
    # The pretrained position grid must also work with smaller runtime images.
    encoder = FrozenDINOv2TactileEncoder(llm_dim=32, backbone_config={
        'hidden_size': 32, 'num_hidden_layers': 1, 'num_attention_heads': 4,
    })
    assert encoder(torch.zeros(1, 3, 28, 28)).shape == (1, 10, 32)


def test_exact_episode_baseline_and_filtered_indices():
    from vtla.datasets.dataset_reader import DatasetReader
    reader = object.__new__(DatasetReader)
    reader._meta = SimpleNamespace(episodes={7: {'dataset_from_index': 100, 'dataset_to_index': 150}})
    reader.delta_indices = {'touch': [0, 0], ACTION: [0, 1, 2]}
    reader.episode_start_image_keys = ['touch']
    for index in (100, 127, 149):
        query, padding = reader._get_query_indices(index, 7)
        assert query['touch'] == [100, index]
        assert not padding['touch_is_pad'].any()
    assert padding['action_is_pad'].tolist() == [False, True, True]
    reader._absolute_to_relative_idx = {100: 0, 127: 27}
    class FilteredRows:
        def __getitem__(self, indices):
            return {'timestamp': [torch.tensor(i / 10) for i in indices]}
    reader.hf_dataset = FilteredRows()
    reader._meta.video_keys = ['touch']
    reader._use_video_keys = ['touch']
    assert reader._get_query_timestamps(2.7, {'touch': [100, 127]}) == {'touch': [0.0, pytest.approx(2.7)]}


def test_real_core_backward_cached_sampling_and_save_reload(tiny_backbone, tmp_path):
    torch.manual_seed(11)
    c = config()
    policy = N0VTLAPolicy(c)
    # Open gate so tactile and predictor gradients are exercised, not just gate gradients.
    policy.model.z_gate.data.fill_(0.8)
    data = batch(c)
    noise = torch.randn(2, c.chunk_size, c.max_action_dim)
    time = torch.full((2,), 0.4)
    policy.train()
    loss, metrics = policy(data, noise=noise, time=time)
    assert loss.isfinite()
    loss.backward()
    assert policy.model.tactile_predictor.latent_queries.grad.abs().sum() > 0
    assert policy.model.tactile_encoder.tactile_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.model.tactile_encoder.backbone.parameters())
    assert not policy.model.tactile_encoder.backbone.training
    policy.eval()
    output = policy.predict_action_chunk(data, noise=noise)
    assert output.shape == (2, 3, 4) and output.isfinite().all()
    changed = dict(data)
    changed['observation.images.touch'] = torch.zeros_like(data['observation.images.touch'])
    assert not torch.allclose(output, policy.predict_action_chunk(changed, noise=noise))
    policy.save_pretrained(tmp_path, push_to_hub=False)
    restored = N0VTLAPolicy.from_pretrained(tmp_path)
    torch.testing.assert_close(output, restored.predict_action_chunk(data, noise=noise))


def test_training_sampling_single_step_agree(tiny_backbone):
    c = config(num_inference_steps=1)
    policy = N0VTLAPolicy(c).eval()
    policy.model.z_gate.data.fill_(1)
    data = batch(c, b=1)
    inputs = policy._inputs(data, training=True)
    noise = torch.randn(1, c.chunk_size, c.max_action_dim)
    actions = torch.zeros_like(noise)
    losses = policy.model(*inputs, actions, noise=noise, time=torch.ones(1))
    sample = policy.model.sample_actions(*inputs, noise=noise)
    # t=1: loss=(velocity-noise)^2 and one Euler step gives sample=noise-velocity.
    torch.testing.assert_close(losses, sample.square(), rtol=1e-4, atol=1e-5)


class LossCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))
        self.calls = 0
    def forward(self, *args, **kwargs):
        actions = args[-1]
        loss = torch.ones_like(actions) * self.value
        loss[:, -1] = 999
        loss[..., 4:] = 999
        return loss
    def sample_actions(self, *args, **kwargs):
        self.calls += 1
        return self.value.expand(args[2].shape[0], 3, 8)


def test_loss_masks_baseline_reset_and_action_queue():
    c = config()
    core = LossCore()
    policy = N0VTLAPolicy(c, core_model=core)
    data = batch(c, b=1)
    data['action_is_pad'][:, -1] = True
    loss, _ = policy(data)
    assert loss.item() == 1
    data['observation.images.touch'] = data['observation.images.touch'][:, 1]
    with pytest.raises(ValueError, match='exact episode'):
        policy(data)
    policy.select_action(data)
    assert core.calls == 1
    baseline = policy._baseline['observation.images.touch'].clone()
    later = dict(data)
    later['observation.images.touch'] = torch.zeros_like(baseline)
    policy.select_action(later)
    assert core.calls == 1
    policy.select_action(later)
    assert core.calls == 2
    torch.testing.assert_close(policy._baseline['observation.images.touch'], baseline)
    policy.reset()
    policy.select_action(later)
    assert policy._baseline['observation.images.touch'].count_nonzero() == 0


def test_native_key_mapping_and_strict_import(tmp_path):
    assert native_key('paligemma_with_expert.paligemma.language_model.x') == 'paligemma_with_expert.paligemma.model.language_model.x'
    assert native_key('tactile_prior.null_g') == 'tactile_predictor.null_g'
    model = nn.Linear(3, 2)
    save_file({'weight': model.weight.detach().clone()}, str(tmp_path / 'model.safetensors'))
    with pytest.raises(ValueError, match='Missing'):
        load_native_weights(model, tmp_path)
    save_file({k: v.clone() for k, v in model.state_dict().items()}, str(tmp_path / 'model.safetensors'))
    load_native_weights(model, tmp_path)


def test_source_has_no_reference_repo_dependency():
    root = Path(__file__).parents[2] / 'vtla/frameworks/n0_vtla'
    for path in root.glob('*.py'):
        text = path.read_text()
        assert 'ref_repo' not in text
        assert 'from n0vtla.' not in text
        assert 'sys.path' not in text


@pytest.mark.parametrize('representation', ['joint', 'rot6d'])
@pytest.mark.parametrize('reference', ['absolute', 'relative'])
@pytest.mark.parametrize('arms', [1, 2])
def test_action_processor_roundtrip_and_no_state(tmp_path, representation, reference, arms):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from vtla.engine.processor.relative_action_processor import route_ee_batch
    from vtla.frameworks.n0_vtla.processor_n0_vtla import make_n0_vtla_pre_post_processors
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1}, unk_token='[UNK]')),
                                       unk_token='[UNK]', pad_token='[PAD]')
    tokenizer.save_pretrained(tmp_path / 'tokenizer')
    width = {'joint': 4, 'rot6d': 10}[representation]
    dim = width * arms
    c = config(state_mode='none', action_mode=f'{reference}_{representation}', ee_num_arms=arms,
               max_action_dim=32, paligemma_tokenizer_path=str(tmp_path / 'tokenizer'),
               action_feature_names=[f'arm{a}_{j}' for a in range(arms) for j in ['j0','j1','j2','gripper']],
               normalization_mapping={k: NormalizationMode.IDENTITY for k in ['VISUAL', 'STATE', 'ACTION']})
    state = torch.zeros(1, dim)
    if representation == 'rot6d':
        for a in range(arms):
            state[0, a * width + 3:a * width + 9] = torch.tensor([1, 0, 0, 0, 1, 0.])
    state[:, 0] = 0.4
    actions = state[:, None, :].expand(-1, 3, -1).clone()
    actions[..., 0] += 0.1
    actions[..., width - 1] = 0.8
    source = {'joint': ACTION, 'rot6d': ACTION + '_absolute_ee'}[representation]
    c.output_features = {source: PolicyFeature(FeatureType.ACTION, (dim,))}
    c.validate_features()
    pre, post = make_n0_vtla_pre_post_processors(c)
    raw = batch(c, b=1)
    raw[ACTION] = actions
    raw[OBS_STATE] = state
    raw['task'] = ['pick']
    if representation != 'joint':
        raw[source] = actions
        raw[OBS_STATE + '_absolute_ee'] = state
    routed = route_ee_batch(raw, c.state_mode, c.action_mode)
    prepared = pre(routed)
    assert OBS_STATE not in prepared
    assert prepared['observation.images.touch'].shape == (1, 2, 3, 28, 28)
    torch.testing.assert_close(post(prepared[ACTION]), actions)
    if reference == 'relative':
        torch.testing.assert_close(prepared[ACTION][..., width - 1], actions[..., width - 1])


@pytest.mark.parametrize('policy_type', ['pi05', 'n0_vtla'])
@pytest.mark.parametrize('explicit_override', [False, True])
def test_relocated_checkpoint_tokenizer(tmp_path, monkeypatch, policy_type, explicit_override):
    from vtla.frameworks.factory import make_pre_post_processors, PolicyProcessorPipeline

    asset = tmp_path / 'paligemma-3b-pt-224-tokenizer'
    asset.mkdir()
    (asset / 'tokenizer_config.json').write_text('{}')
    cfg = make_policy_config(policy_type, paligemma_tokenizer_path=f'/old/training/machine/{asset.name}')
    cfg.state_mode = 'absolute_joint'
    cfg.action_mode = 'absolute_joint'
    captured = []

    def load_pipeline(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(steps=[])

    monkeypatch.setattr(PolicyProcessorPipeline, 'from_pretrained', load_pipeline)
    overrides = {'tokenizer_processor': {'max_length': 123}}
    if explicit_override:
        overrides['tokenizer_processor']['tokenizer_name'] = 'explicit/tokenizer'
    make_pre_post_processors(cfg, pretrained_path=tmp_path, preprocessor_overrides=overrides)
    actual = captured[0]['overrides']['tokenizer_processor']
    assert actual['tokenizer_name'] == ('explicit/tokenizer' if explicit_override else str(asset))
    assert actual['max_length'] == 123
    assert overrides['tokenizer_processor'] == (
        {'max_length': 123, 'tokenizer_name': 'explicit/tokenizer'} if explicit_override else {'max_length': 123}
    )


def test_processor_saved_step_reload(tmp_path):
    from vtla.engine.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch
    from vtla.frameworks.n0_vtla.processor_n0_vtla import N0VTLAPrepareStateStep
    pipeline = PolicyProcessorPipeline(steps=[N0VTLAPrepareStateStep(max_state_dim=8)], name='test')
    pipeline.save_pretrained(tmp_path, config_filename='pre.json', push_to_hub=False)
    restored = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename='pre.json',
                                                       to_transition=batch_to_transition, to_output=transition_to_batch)
    output = restored({OBS_STATE: torch.zeros(1, 4), 'task': ['pick']})
    assert output[OBS_STATE].shape == (1, 8)
    assert len(output['task'][0].split('State: ')[1].split(';')[0].split()) == 8


@pytest.mark.parametrize('arch', ['joint_kv', 'tactile_kv'])
def test_predictor_masks_missing_sensor_and_has_finite_gradients(arch):
    from vtla.frameworks.n0_vtla.tactile_predictor import TactileActionPredictor
    torch.manual_seed(3)
    predictor = TactileActionPredictor(16, 2, 1, 4, arch)
    context = torch.randn(2, 3, 16)
    tactile = torch.randn(2, 4, 16, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    output = predictor(context, tactile, mask, torch.ones(2, 3, dtype=torch.bool))
    changed = tactile.detach().clone()
    changed[~mask] = 100
    torch.testing.assert_close(output, predictor(context, changed, mask, torch.ones(2, 3, dtype=torch.bool)))
    output.sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(tactile.grad).all()
    assert tactile.grad[~mask].count_nonzero() == 0


def test_native_import_real_core(tiny_backbone, tmp_path):
    policy = N0VTLAPolicy(config())
    native = {}
    for key, value in policy.model.state_dict().items():
        old_key = key.replace('tactile_predictor.', 'tactile_prior.')
        for component in ('language_model', 'vision_tower', 'multi_modal_projector'):
            old_key = old_key.replace(f'paligemma.model.{component}.', f'paligemma.{component}.')
        native[old_key] = value.clone()
    # safetensors native export may omit this tied alias.
    native.pop('paligemma_with_expert.paligemma.lm_head.weight')
    save_file(native, str(tmp_path / 'model.safetensors'))
    restored = N0VTLAPolicy(config(base_model_path=tmp_path, allow_random_init=False))
    for key, value in policy.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key])


def test_uint8_and_float_tactile_preprocessing_agree():
    c = config()
    policy = N0VTLAPolicy(c, core_model=LossCore())
    data = batch(c, b=1)
    for key in c.image_keys():
        data[key] = torch.randint(0, 256, data[key].shape, dtype=torch.uint8)
    float_data = dict(data)
    for key in c.image_keys():
        float_data[key] = data[key].float() / 255
    byte_inputs = policy._inputs(data, training=True)
    float_inputs = policy._inputs(float_data, training=True)
    torch.testing.assert_close(byte_inputs[0], float_inputs[0])
    torch.testing.assert_close(byte_inputs[4], float_inputs[4])
    expected = 2 * (float_data[c.tactile_keys[0]][:, 1] - float_data[c.tactile_keys[0]][:, 0])
    torch.testing.assert_close(byte_inputs[4][0], expected)


def test_checkpoint_width_validation_routes_eef_before_comparing(tmp_path):
    from vtla.frameworks.sensor_routing import ACTION_ABSOLUTE_EE
    c = config(state_mode='none', action_mode='absolute_rot6d', ee_num_arms=1,
               tcp_contract=build_tcp_contract('umi', 32, 0),
               max_action_dim=32, output_features={ACTION: PolicyFeature(FeatureType.ACTION, (10,))})
    policy = N0VTLAPolicy(c, core_model=LossCore())
    policy.save_pretrained(tmp_path, push_to_hub=False)
    incoming = config(state_mode='none', action_mode='absolute_rot6d', ee_num_arms=1,
                      max_action_dim=32, output_features={
                          ACTION: PolicyFeature(FeatureType.ACTION, (8,)),
                          ACTION_ABSOLUTE_EE: PolicyFeature(FeatureType.ACTION, (10,)),
                      })
    restored = N0VTLAPolicy.from_pretrained(tmp_path, config=incoming, core_model=LossCore())
    assert restored.action_dim == 10


def test_action_gap_sampler_uses_filtered_local_rows():
    from vtla.datasets.sampler import MixtureSampler
    dataset = SimpleNamespace(episodes=[2], repo_id='test', meta=SimpleNamespace(episodes={
        2: {'dataset_from_index': 100, 'dataset_to_index': 110}
    }))
    c = config(action_gap=3)
    assert MixtureSampler._valid_child_indices(dataset, 0, c.drop_n_last_frames) == list(range(7))
