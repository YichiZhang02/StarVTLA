# TacMind0 in StarVTLA

`tacmind0` is a registered StarVTLA policy. The native Gemma3 action model and
Tac-LeWM implementation live in the repository's `tacmind0/` package. Runtime
imports and asset loading use this checkout only. The internal native tensor
names remain compatible with the original weights; the public policy type,
training entrypoint, and checkpoint type are `tacmind0`.

## Assets

The local assets are under `playground/pretrained_models/tacmind0/`. The base
policy, tokenizer, encoder checkpoint, and backbone config are all required for
initial training. These files are ignored by Git because the base weights are
about 23 GB. `python3 tools/tacmind0/verify_weights.py` checks their SHA-256
hashes against the bundled manifest. A saved StarVTLA checkpoint includes the
full policy and its processor, so loading it does not require the base assets.

## Training

```bash
bash train.sh <registered_name|source/group/dataset_id> tacmind0 4 1 20000 \
  true as_image none relative_rot6d 6 strong
```

`train.sh` forces `tactile_mode=as_image`. TacMind0 routes these image streams
through its Tac-LeWM encoder and FiLM fusion, not through the RGB vision tower.
The encoder is trainable and included in the optimizer. The data reader takes
eight tactile frames at offsets `[-35,-30,-25,-20,-15,-10,-5,0]`; the native
preprocessing converts RGB marker frames to BGR 42×42 inputs. Exactly two
`tactile_keys`, ordered left then right, are required. Set `TACTILE_KEYS` if
the dataset catalog has more than two tactile cameras.

The native model consumes two RGB views. When `wrist_only=true` selects one
wrist camera, the adapter supplies that view twice. With `wrist_only=false`,
the selected top and wrist cameras must total at most two. The model action
container has 32 dimensions; the actual state and action semantics, action
gap, normalization, mixture sampling, and RGB augmentation follow StarVTLA's
existing processors and dataset statistics. Action dimensions above 32 are
rejected.

Training follows the native TacMind runtime defaults: FSDP-1 with SHARD_GRAD_OP on at least
two GPU processes, gradient checkpointing for the VLM and action expert, a frozen VLM token
embedding, and trainable tactile encoder/FiLM. The StarVTLA optimizer keeps the native
2.5e-5 learning rate and 1e-10 weight decay. The train.sh TacMind-0 route configures
FSDP automatically; the top-level 1.sh verifies the local asset manifest before starting
its sequential comparison runs.

For new training, `TACMIND0_BASE_PATH`, `TACMIND0_TACTILE_WEIGHTS_PATH`, and
`TACMIND0_BACKBONE_CONFIG_PATH` can override the local asset paths.
`PRETRAINED_PATH` points to a saved StarVTLA `tacmind0` checkpoint for fine
tuning or resuming. The checkpoint contains the adapted tactile encoder.

## Inference and validation

Saved checkpoints use the existing `inference.sh` and offline evaluation
entrypoints. The processor buffers the same eight-frame tactile history during
online inference and clears it on episode reset.

```bash
python3 -m pytest tests/frameworks/test_tacmind0.py -q
python3 -m tools.tacmind0.smoke_policy --device cuda:0
```

The smoke command loads the real base weights and runs one action chunk on
synthetic RGB and tactile inputs. It checks model loading and tensor flow;
robot task performance requires evaluation on actual data.
