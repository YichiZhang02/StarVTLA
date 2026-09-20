"""Offline numerical smoke test of real N0-VTLA weights, not a task evaluation.

Run from the repository root: python -m tools.test_n0_vtla_weights --help
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vtla.engine.configs import FeatureType, PolicyFeature
from vtla.engine.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from vtla.frameworks.n0_vtla.configuration_n0_vtla import N0VTLAConfig
from vtla.frameworks.n0_vtla.modeling_n0_vtla import N0VTLAPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path('playground/pretrained_models/n0-vtla-base'))
    parser.add_argument('--tokenizer', default='playground/pretrained_models/pi05_base/paligemma-3b-pt-224-tokenizer')
    parser.add_argument('--output', type=Path, default=Path('playground/results/n0_vtla_weights_smoke.json'))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This full-size smoke test requires a CUDA GPU.')
    torch.set_num_threads(4)
    torch.manual_seed(7)
    rgb = ['observation.images.top', 'observation.images.left_wrist', 'observation.images.right_wrist']
    tactile = [f'observation.images.{name}' for name in (
        'left_wrist_left_tactile', 'left_wrist_right_tactile',
        'right_wrist_left_tactile', 'right_wrist_right_tactile')]
    config = N0VTLAConfig(
        device='cuda', dtype='bfloat16', base_model_path=args.weights,
        top_camera_keys=rgb[:1], wrist_camera_keys=rgb[1:], tactile_keys=tactile,
        state_mode='absolute_joint', action_mode='absolute_joint',
        gradient_checkpointing=True, num_inference_steps=10,
        input_features={**{k: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for k in rgb + tactile},
                        OBS_STATE: PolicyFeature(FeatureType.STATE, (20,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (20,))},
    )
    started = time.perf_counter()
    with torch.device('cuda'):
        policy = N0VTLAPolicy(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    print(f'Strict weight load passed ({load_seconds:.2f}s)', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    # Synthetic normalized state, already formatted as a padded PI0.5 prompt.
    prompt = 'Task: pick up the object, State: ' + ' '.join(['128'] * 32) + ';\nAction: '
    tokens = tokenizer(prompt, return_tensors='pt', max_length=200, padding='max_length', truncation=True)
    batch = {k: torch.rand(1, 3, 224, 224, device='cuda') for k in rgb}
    batch.update({k: torch.rand(1, 2, 3, 224, 224, device='cuda') for k in tactile})
    batch.update({OBS_LANGUAGE_TOKENS: tokens['input_ids'].cuda(),
                  OBS_LANGUAGE_ATTENTION_MASK: tokens['attention_mask'].bool().cuda(),
                  ACTION: torch.randn(1, 50, 20, device='cuda')})
    policy.train()
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss), 'Nonfinite training loss'
    gradients = {}
    for name, parameter in policy.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), f'Nonfinite gradient: {name}'
        if name in ('model.tactile_predictor.latent_queries', 'model.tactile_encoder.tactile_proj.weight',
                    'model.z_gate', 'model.action_out_proj.weight'):
            assert parameter.grad is not None and parameter.grad.abs().max() > 0, name
            gradients[name] = parameter.grad.float().norm().item()
    assert all(p.grad is None for p in policy.model.tactile_encoder.backbone.parameters())
    training_peak = torch.cuda.max_memory_allocated() / 1e9
    policy.zero_grad(set_to_none=True)
    print(f'Forward/backward passed; loss={loss.item():.6f}', flush=True)
    noise = torch.randn(1, 50, 32, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    # Warm up before timing. Fix noise to isolate the tactile intervention.
    original = policy.predict_action_chunk(batch, noise=noise.clone())
    torch.cuda.synchronize()
    started = time.perf_counter()
    repeated = policy.predict_action_chunk(batch, noise=noise.clone())
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    torch.testing.assert_close(original, repeated, rtol=0, atol=0)
    changed_batch = dict(batch)
    for key in tactile:
        changed_batch[key] = batch[key].clone()
        changed_batch[key][:, 1] = changed_batch[key][:, 0]
    changed = policy.predict_action_chunk(changed_batch, noise=noise.clone())
    for output in (original, repeated, changed):
        assert output.shape == (1, 50, 20) and torch.isfinite(output).all()
    difference = (original - changed).abs()
    assert difference.max() > 0, 'Tactile intervention had no effect on actions'
    report = dict(
        status='passed', scope='synthetic numerical smoke; no task quality or native-policy parity evaluation',
        weights=str(args.weights.resolve()), gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        rgb_views=3, tactile_views=4, action_dim=20, chunk_size=50, denoising_steps=10,
        parameter_count=sum(p.numel() for p in policy.parameters()), load_seconds=load_seconds,
        loss=loss.item(), gradient_norms=gradients, z_gate=policy.model.z_gate.item(),
        deterministic_fixed_noise=True, tactile_action_max_difference=difference.max().item(),
        tactile_action_mean_difference=difference.mean().item(), inference_seconds=inference_seconds,
        training_peak_allocated_gb=training_peak,
        inference_peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
