# Tactile MAE

该模块实现 AnyTouch stage-1 风格的 masked autoencoder，用 StarVTLA 的 LeRobot 触觉视频或 raw frame cache 预训练触觉图像 backbone。它只训练 image path，不包含文本对齐、跨传感器标签或 AnyTouch 后续对比学习阶段。

## 模型

| 组件 | ViT-L | ViT-B |
| --- | --- | --- |
| encoder | ViT-L/14，1024 hidden，24 layers | ViT-B/16，768 hidden，12 layers |
| projection | 1024 -> 768 | 768 -> 512 |
| decoder | 8 layers，512 hidden，16 heads，2048 MLP | 同左 |
| sensor token | 10 slots，每 slot 5 tokens | 同左 |
| 默认 mask | 75% random patches | 同左 |
| loss | masked-patch MSE + 可选 visible loss | 同左 |

Patch embedding 遵循 AnyTouch stage-1 的 `use_same_patchemb` 路径：图像重复成 3 帧后进入 `video_patch_embedding` Conv3d。Encoder 使用 Transformers CLIP building blocks，decoder 位于 [models/vit_decoder.py](models/vit_decoder.py)，参数命名保持与 AnyTouch checkpoint 一致。

## 推荐训练入口

从仓库根目录运行：

```bash
bash scripts/train_enc.sh \
  <dataset_ids> <scratch|clip|anytouch> <vit_b|vit_l> \
  <num_processes> <batch_size> <epochs>
```

示例：

```bash
bash scripts/train_enc.sh \
  "dataset_a dataset_b" clip vit_b 4 128 100
```

输出位于：

```text
playground/results/backbones/<timestamp>_tacmae_<arch>_from_<init_mode>/
```

包装脚本负责识别输入模式、预热数据 cache、选择 camera key、启动单进程 Python 或多进程 torchrun，并把配置和日志保存到 run 目录。完整参数和环境变量见 [Workflow Scripts](../../../scripts/README.md#触觉-backbone-训练)。

## 初始化

统一通过 `pretrained_path` 自动识别权重命名空间：

| 模式 | 来源 | 加载范围 |
| --- | --- | --- |
| `scratch` | 空 | 全部随机初始化 |
| `clip` | 本地 CLIP 目录 | encoder + projection |
| `anytouch` | AnyTouch `.pth` 或转换目录 | 完整 MAE strict load |

默认本地路径：

```text
ViT-B CLIP: playground/pretrained_models/CLIP-ViT-B-16-DataComp.XL-s13B-b90K
ViT-L CLIP: playground/pretrained_models/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
AnyTouch:   playground/pretrained_models/AnyTouch-ViT-L-16
```

公开 AnyTouch 完整权重只支持 ViT-L，因此 `init_mode=anytouch` 与 `arch=vit_b` 会被拒绝。

将原始 AnyTouch checkpoint 转为 HF 风格目录：

```bash
python -m vtla.tac_encoder.tactile_mae.tools.convert_anytouch_to_hf \
  --src playground/pretrained_models/checkpoint.pth \
  --out playground/pretrained_models/anytouch_mae_vitl \
  --arch vit_l
```

## Sensor ID

AnyTouch 有 10 个 sensor slots，每个 slot 5 个 token。已知 ID 包括：

| ID | 传感器 |
| ---: | --- |
| `0` | early GelSight |
| `1` | DIGIT |
| `2` | GelSight OF-Real |
| `3` | GelSight Mini |
| `4` | DuraGel |
| `-1` | agnostic，映射到 slot 9 |

当前触觉默认使用 `SENSOR_ID=-1`。只有明确需要复用某类 sensor token 时才改为 `3` 或未占用 slot。

## LeRobot 数据模式

包含 `meta/info.json` 的输入按 LeRobot 数据集处理。脚本只解码触觉 camera key，不读取 top/wrist RGB。未显式设置 `TACTILE_KEYS` 时，包装脚本使用当前默认触觉 key；单臂或不同命名的数据建议直接从数据集 metadata 传入：

```bash
TACTILE_KEYS='[observation.images.left_cam_finger0,observation.images.left_cam_finger1]' \
  bash scripts/train_enc.sh <dataset_id> clip vit_b 1 128 100
```

首次运行会构建 dataset cache。多个 LeRobot 数据集可以联合训练，但不能与 raw frame cache 混合。

## 接触帧筛选

包装脚本默认启用 contact filter：

```text
score = max(per-channel pixel std), scale 0..255
contact = score > CONTACT_STD_THRESHOLD
```

接触帧全部保留，非接触帧按 `NONCONTACT_KEEP_RATIO` 随机保留。默认：

```text
CONTACT_FILTER=1
CONTACT_STD_THRESHOLD=0.5
NONCONTACT_KEEP_RATIO=0.05
CONTACT_STRIDE=1
```

结果缓存到 `<dataset>/meta/contact_std.npz`。关闭筛选：

```bash
CONTACT_FILTER=0 bash scripts/train_enc.sh <dataset_id>
```

## Raw Frame Cache

不含 LeRobot metadata 的连续触觉图像流可以先转成 decode-once frame cache：

```bash
python -m vtla.tac_encoder.tactile_mae.tools.png_to_frame_cache \
  --src_dir <flat_png_dir> \
  --dataset_root playground/data \
  --dataset_id pretrained_data \
  --camera_key observation.images.cam_finger0 \
  --image_size 224 \
  --num_workers 16
```

然后运行：

```bash
RAW_FRAME_CACHE=1 \
IMAGE_SIZE=224 \
  bash scripts/train_enc.sh pretrained_data clip vit_b 4 128 100
```

Raw 模式按连续行范围划分 train/validation，最后 `VAL_RATIO` 比例作为 validation。`IMAGE_SIZE` 必须和 cache 签名一致。因为没有逐帧 LeRobot 解码来源，contact filter 会自动关闭。

## 直接调用

需要绕过包装脚本时：

```bash
torchrun --nproc_per_node=4 \
  -m vtla.tac_encoder.tactile_mae.train \
  --arch vit_l \
  --pretrained_path playground/pretrained_models/AnyTouch-ViT-L-16 \
  --dataset_root playground/data \
  --dataset_ids <dataset_id> \
  --camera_keys observation.images.left_cam_finger0 observation.images.left_cam_finger1 \
  --sensor_id -1 \
  --use_sensor_token \
  --use_same_patchemb \
  --sensor_token_for_all \
  --mask_ratio 0.75 \
  --batch_size 64 \
  --epochs 100 \
  --warmup_epochs 1 \
  --weight_decay 0.1 \
  --blr 1e-5 \
  --amp_dtype bfloat16 \
  --output_dir playground/results/backbones/tacmae_run
```

包装脚本当前还设置 `visible_loss_weight=0.1`，并对 sensor-token beta 从 `0.0` 调度到 `0.75`。复现实验时应一并记录这些参数。

## Eval 和可视化

```bash
bash vtla/tac_encoder/tactile_mae/scripts/eval.sh \
  <checkpoint> <vit_b|vit_l> <dataset_ids...>
```

输出目录包含：

| 文件 | 内容 |
| --- | --- |
| `metrics.txt` | validation masked-patch MSE |
| `reconstruction.png` | 原图、mask、重建和 pasted 对比 |
| `tsne.png` | CLS 特征的 t-SNE |

## 目录

```text
tactile_mae/
├── models/      MAE、decoder、position embedding 和构建逻辑
├── data/        LeRobot 触觉数据集
├── engine/      训练循环、scheduler 和分布式工具
├── tools/       checkpoint 与 frame-cache 转换
├── scripts/     直接训练和评估脚本
├── config.py
├── train.py
└── eval.py
```
