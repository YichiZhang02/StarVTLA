# StarvlaGroot (QwenGR00T) policy

Qwen-VL backbone (prefix encoder) + GR00T flow-matching DiT action head, ported
from starVLA into the vtla (LeRobot-style) framework. Registered policy type:
`starvla_groot`.

## Layout
- `configuration_starvla_groot.py` — `StarvlaGrootConfig(PreTrainedConfig)`
- `modeling_starvla_groot.py` — `StarvlaGrootPolicy(PreTrainedPolicy)`
- `processor_starvla_groot.py` — `make_starvla_groot_pre_post_processors`
- `qwen_vl_interface.py` — generic `AutoModelForImageTextToText` wrapper
- `action_head/` — vendored GR00T flow-matching head (`flow_matching_head.py`,
  `cross_attention_dit.py` [needs `diffusers`], `action_encoder.py`)

No YAML is required: the config is a draccus-registered dataclass, exactly like
`act` / `pi05`. Override any field on the command line with `--policy.<field>=...`.

## Minimal training command
```bash
python -m vtla.train \
  --dataset.repo_id=<your/dataset> \
  --policy.type=starvla_groot \
  --policy.base_vlm=./playground/pretrained_models/Qwen3.5-0.8B \
  --policy.chunk_size=8 \
  --policy.n_action_steps=8 \
  --policy.state_mode=absolute_joint \
  --policy.top_camera_key=observation.images.cam_top \
  --policy.wrist_camera_key=observation.images.cam_right_wrist \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/train/starvla_groot
```

Useful overrides:
- `--policy.action_model_type=DiT-B|DiT-L`
- `--policy.repeated_diffusion_steps=8` (noise samples per batch element)
- `--policy.num_inference_timesteps=4` (denoising Euler steps at inference)
- `--policy.wrist_only=true` (single-view) / `--policy.state_mode=none`
- `--policy.train_expert_only=true` (freeze the VLM, train only the action head)
- `--policy.freeze_vision_encoder=true`

## VLM / transformers compatibility
The action head is VLM-agnostic; the backbone is loaded by
`AutoModelForImageTextToText.from_pretrained(base_vlm)`, so `base_vlm` must be an
architecture your installed `transformers` recognizes.

- `Qwen/Qwen3-VL-*` (model_type `qwen3_vl`) works on transformers >= 4.57.0.
- `Qwen3.5-0.8B` (model_type `qwen3_5`) is **not** in any stable transformers
  release as of 4.57.x — its `config.json` reports `transformers_version
  4.57.0.dev0`. To use it, install a transformers build that ships `qwen3_5`
  (e.g. from source), or point `base_vlm` at a `qwen3_vl` checkpoint instead.
