# Tactile MAE

A clean, self-contained re-implementation of **AnyTouch (stage 1) MAE** for
pretraining a tactile-image backbone directly on **LeRobot** datasets.

Because we only have tactile *images* (no text / cross-sensor labels / multi-stage
contrastive learning), AnyTouch degenerates to a masked auto-encoder. This repo
keeps the AnyTouch model structure and stage-1 training recipe **identical**, but:

- trains directly on LeRobot v3.0 datasets (no data-format conversion);
- supports both **ViT-L/14** and **ViT-B/16**;
- supports three init modes through a single `--pretrained_path`;
- ships eval + reconstruction visualization + t-SNE.

## Model (identical to AnyTouch stage1, image path)

| component | spec |
|---|---|
| encoder | CLIP ViT (`touch_model.*`) — ViT-L/14 (1024-d, 24L) or ViT-B/16 (768-d, 12L) |
| projection | `touch_projection`: hidden → 768 (L) / 512 (B) |
| decoder | 8× ViT layer, 512-d / 16 heads / 2048 mlp (`touch_decoder_blocks.*`) |
| tokens | cls + **5 sensor tokens** (`sensor_token` ∈ ℝ^{10×5×d}) |
| masking | random 75% |
| loss | masked-patch MSE (`norm_pix_loss=False`) |
| patch-embed | stage1 `use_same_patchemb`: image → 3×-repeat → `video_patch_embedding` (Conv3d) |

The encoder is assembled from `transformers` CLIP building blocks and the decoder is
vendored ([models/vit_decoder.py](models/vit_decoder.py)) so the parameter names match
the released AnyTouch checkpoint exactly and it **strict-loads** (missing=0, unexpected=0),
independent of the installed `transformers` version.

## Sensor id

AnyTouch `sensor_token` has 10 slots (each 5 tokens). Pretrain used:
`0` GelSight(early) · `1` DIGIT · `2` GelSight(OF-Real) · `3` GelSight-Mini · `4` DuraGel · `-1` agnostic (slot 9).
Our HD tac16 finger defaults to `--sensor_id -1` (agnostic). Switch to `3` (gelsight-like)
or a free slot (`6`) via `--sensor_id`.

## Three init modes — one `--pretrained_path`

| mode | `--pretrained_path` | behavior |
|---|---|---|
| from scratch | *(empty)* | random init |
| from CLIP | `playground/pretrained_models/CLIP-ViT-L-14-DataComp.XL-s13B-b90K` | loads encoder+projection (decoder/sensor tokens init) |
| from AnyTouch | `playground/pretrained_models/checkpoint.pth` (or converted dir) | strict full MAE load |

The loader auto-detects the source namespace (`vision_model.*` = CLIP, `touch_mae_model.*` /
`touch_model.*` = AnyTouch). Optionally normalize the AnyTouch `.pth` into an HF-style dir:

```bash
python -m vtla.tac_encoder.tactile_mae.tools.convert_anytouch_to_hf \
  --src playground/pretrained_models/checkpoint.pth \
  --out playground/pretrained_models/anytouch_mae_vitl --arch vit_l
```

## Train

```bash
# scripts/train.sh <scratch|clip|anytouch> [arch] [num_gpus] [dataset_ids...]
bash vtla/tac_encoder/tactile_mae/scripts/train.sh anytouch vit_l 4 \
     rm_nist_260320_strawberry rm_nist_260520_usb
```

Or directly:

```bash
torchrun --nproc_per_node=4 -m vtla.tac_encoder.tactile_mae.train \
  --arch vit_l --pretrained_path playground/pretrained_models/checkpoint.pth \
  --dataset_root playground/data --dataset_ids rm_nist_260320_strawberry \
  --camera_keys observation.images.cam_finger0 observation.images.cam_finger1 \
  --sensor_id -1 --use_sensor_token --use_same_patchemb \
  --sensor_token_for_all --batch_size 64 --epochs 20 --warmup_epochs 1 \
  --weight_decay 0.1 --blr 1e-3 --output_dir playground/results/tac_mae
```

Defaults mirror `train_stage1.sh`: AdamW(β=0.9,0.99), wd 0.1, `lr=blr·eff_bs/256`,
half-cycle cosine + warmup, AMP, ImageNet-norm + H/V-flip + ColorJitter aug.

## Eval & visualization

```bash
# scripts/eval.sh <checkpoint> [arch] [dataset_ids...]
bash vtla/tac_encoder/tactile_mae/scripts/eval.sh \
     playground/results/tac_mae/checkpoint-19.pth vit_l rm_nist_260320_strawberry
```

Produces, under `--output_dir`:
- `metrics.txt` — masked-patch MSE on the held-out split;
- `reconstruction.png` — `[original | masked | reconstruction | pasted]`;
- `tsne.png` — t-SNE of CLS features (colored by dataset, or by camera for a single dataset).

## Layout

```
tactile_mae/
├── models/        mae_model.py · vit_decoder.py · pos_embed.py · build.py
├── data/          lerobot_tactile_dataset.py
├── engine/        train_engine.py · lr_sched.py · misc.py
├── tools/         convert_anytouch_to_hf.py
├── scripts/       train.sh · eval.sh
├── config.py · train.py · eval.py
```

## Pretrain on a flat image stream (no LeRobot)

To pretrain on a raw directory of tactile PNGs (e.g. AnyTouch `data_tac2_s`,
a single continuous stream with no episodes / state / action), skip LeRobot
entirely: convert the PNGs straight into the decode-once **frame cache** and
train with `--raw_frame_cache`.

```bash
# 1) PNG stream -> frame cache (resized uint8 memmap; ~16.5 GB for 224 @ 114905 frames)
python -m vtla.tac_encoder.tactile_mae.tools.png_to_frame_cache \
  --src_dir <flat_png_dir> \
  --dataset_root playground/data --dataset_id pretrained_data \
  --camera_key observation.images.cam_finger0 --image_size 224 --num_workers 16

# 2) train directly off the cache (no mp4 / parquet / LeRobot metadata)
torchrun --nproc_per_node=4 -m vtla.tac_encoder.tactile_mae.train \
  --raw_frame_cache --dataset_root playground/data --dataset_ids pretrained_data \
  --camera_keys observation.images.cam_finger0 --image_size 224 \
  --arch vit_l --pretrained_path playground/pretrained_models/checkpoint.pth \
  --sensor_id -1 --use_sensor_token --use_same_patchemb --sensor_token_for_all \
  --batch_size 64 --epochs 20 --output_dir playground/results/tac_mae_pretrain
```

`--raw_frame_cache` reads the pre-built cache directly and splits train/val by
contiguous **row range** (last `--val_ratio` fraction = val), since the stream has
no episodes. `--image_size` must match what the cache was built with (the cache
signature folder, e.g. `all_224_v1`, encodes it). Contact filtering is not
available in this mode (it needs per-frame decode from a LeRobot dataset).

## Contact-frame filtering (optional)

By default every tactile frame is used. With `--contact_filter`, training keeps
only **contact** frames and subsamples the rest:

- contact score = **max per-channel std** of the frame (0-255 scale); idle gel
  frames are ~0.1, contact frames rise to several units;
- `score > --contact_std_threshold` (default `0.5`) ⇒ contact (kept);
- otherwise kept with probability `--noncontact_keep_ratio` (default `0.05`).

Scores are computed once and cached at `<dataset>/meta/contact_std.npz`. In
`scripts/train_enc.sh` enables this by default; tune via env:
`CONTACT_FILTER=0` (off), `CONTACT_STD_THRESHOLD`, `NONCONTACT_KEEP_RATIO`.

## Notes
- ViT-L `from CLIP` uses the HF CLIP dir; ViT-B `from CLIP` reads the
  **open_clip** `CLIP-ViT-B-16` weights directly (auto-remapped in the loader).
- `from anytouch` is ViT-L only (no released ViT-B weights).
- Only tactile camera streams are decoded (top/wrist views are skipped) for speed.
