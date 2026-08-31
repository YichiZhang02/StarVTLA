# FastWAM

`fastwam` 是 StarVTLA 的 world-action model。它以 Wan2.2-TI2V-5B 为视频生成骨干，并加入
action DiT，使未来视频与 action chunk 在同一个训练目标中联合生成。

共享的数据集、`robot_type`、state/action 表示和相机路由约定见
[VTLA Training](../../README.md)。

## 最小环境

当前仓库验证基线为 Python 3.10.19、PyTorch 2.7.1+cu128、torchvision
0.22.1+cu128 和 CUDA 12.8。FastWAM 的直接依赖包括 `transformers==5.5.0`、
`einops==0.8.1`、`safetensors==0.7.0`、Pillow 和 `rich==15.0.0`，均由根目录
`requirements.txt` 安装。

```bash
conda create -n starvtla-fastwam python=3.10 -y
conda activate starvtla-fastwam
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

安装检查：

```bash
python -c "import torch, transformers, einops, safetensors, rich; from vtla.frameworks.fastwam.modeling_fastwam import FastWAMPolicy; print(torch.__version__)"
```

FastWAM 默认使用 BF16 和大型 Wan 组件，正式训练需要支持 BF16 的 CUDA GPU；CPU 不适合作为
训练或完整推理环境。

## 预训练资产

默认目录必须至少包含：

```text
playground/pretrained_models/Wan2.2-TI2V-5B/
├── diffusion_pytorch_model-*.safetensors
├── Wan2.2_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
├── google/umt5-xxl/
│   ├── tokenizer.json
│   └── spiece.model
└── interpolated_dit/
    └── InterpolatedDiT_from_official_Wan2.2_alphascale_1024hdim.pt
```

若缺少插值 DiT，先执行：

```bash
python tools/generate_interpolated_dit.py \
  --wan-dir playground/pretrained_models/Wan2.2-TI2V-5B \
  --device cuda --dtype bfloat16
```

工具只生成插值 DiT，不下载其余 Wan2.2 资产。

## 文本 Embedding

训练前每个数据集都需要本地任务文本缓存：

```bash
python tools/precompute_world_model_text_embeddings.py \
  --dataset-root playground/data/<dataset_id> \
  --world-model wan22 \
  --device cuda
```

输出是 `text_embeddings/wan22/manifest.json` 和 `embeddings.safetensors`。`train.sh` 会检查并在
缺失时自动生成；mixture 会逐个处理成员数据集。训练时固定
`load_text_encoder=false`，因此每步训练不会加载 T5。

## 时间布局与默认值

默认 `n_obs_steps=33`、`video_frame_stride=4`，得到 9 帧视频条件；`chunk_size=32`，每个视频
transition 对应 4 个 action。每个相机先缩放到 `224x224`，随后沿宽度拼接。

| 配置 | 默认值 |
| --- | ---: |
| `chunk_size` / `n_action_steps` | `32` / `16` |
| `num_inference_steps` | `10` |
| `loss_lambda_video` | `1.0` |
| `loss_lambda_action` | `1.0` |
| `dtype` | `bfloat16` |

## 训练

```bash
VISUALIZATION_ENABLED=true \
  bash train.sh <dataset_id> fastwam 1 1 50000 \
  true none absolute_joint absolute_joint 6 none
```

`tactile_mode=encode` 可把 Tactile MAE context 注入 video DiT 或 action DiT：

```bash
TACTILE_INSERT_LOCATION=decoder \
TACTILE_NUM_FRAMES=3 \
TACTILE_ENCODER_PATH=<backbone_checkpoint> \
  bash train.sh <dataset_id> fastwam 1 1 50000 \
  true encode absolute_joint absolute_joint 6 none
```

FastWAM 也支持 `tactile_mode=as_image`，此时触觉历史由 Wan VAE 编码，并且必须设置
`TACTILE_INSERT_LOCATION=encoder`。训练可视化默认每 1000 step 输出预测视频、GT、动作误差与
耗时到 run 目录的 `visualizations/`。

## 推理

```bash
bash inference.sh <run_id> <step> async <robot_type> 16 0
```

部署加载 StarVTLA checkpoint 中保存的模型权重和资产路径。迁移 checkpoint 到另一台机器时，
Wan VAE 等未写入 policy checkpoint 的底座资产路径也必须可用。
