# Dream-Tac

StarVTLA 的 Dream-Tac policy 使用 Cosmos-Predict2-2B-Video2World 作为生成模型底座，同时
沿用 StarVTLA 的 LeRobot dataset、sensor routing、processor、Accelerate 训练循环、optimizer、
checkpoint 和部署接口。Cosmos Policy 源码位于 `vendor/cosmos_policy/`，运行时不依赖
`ref_repo/` 或系统安装的 `cosmos-policy` 包。

## 环境安装

Dream-Tac 是可选 policy，因此它的额外依赖以注释形式记录在仓库根目录
`requirements.txt`，不会被默认的 `pip install -r requirements.txt` 安装。当前验证环境为：

```text
Python 3.10
PyTorch 2.7.1+cu128
CUDA 12.8
cuDNN 9.7
NVIDIA A100 (sm_80)
```

先安装 StarVTLA 的通用依赖和与当前 CUDA 匹配的 PyTorch，再安装 Dream-Tac 依赖：

```bash
python -m pip install \
  peft==0.20.0 loguru==0.7.3 sentencepiece==0.2.1 \
  boto3==1.42.70 ftfy==6.3.1 better-profanity==0.7.0 \
  mediapy==1.2.2 webdataset==0.2.111 hydra-core==1.3.2 \
  "transformer-engine[pytorch]==2.2.0"

python -m pip install --no-deps megatron-core==0.10.0
```

Megatron Core 仍由 Cosmos 的 Hydra model loader、并行状态和 tokenizer 路径使用。这里使用
`--no-deps`，是为了避免 `megatron-core==0.10.0` 的开放式依赖解析安装 ModelOpt 或替换已有
PyTorch。Transformer Engine 的 `[pytorch]` extra 会同时安装 Python/common 包和 PyTorch CUDA
扩展。CUDA runtime、cuBLAS 和 cuDNN 应由当前 PyTorch wheel 或系统 CUDA 环境提供，本项目不固定
安装 `nvidia-cudnn-cu12`、`nvidia-cublas-cu12` 或 `nvidia-cuda-runtime-cu12`。

安装后可检查实际训练路径所需的导入：

```bash
python - <<'PY'
from vtla.frameworks.dream_tac.runtime import ensure_dream_tac_importable

ensure_dream_tac_importable()
import transformer_engine.pytorch  # noqa: F401
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

print(load_model_from_checkpoint)
PY
```

### Transformer Engine 编译回退

若 `transformer_engine.pytorch` 导入失败，通常是 PyTorch CUDA 扩展没有适配当前环境。先确认
PyTorch、CUDA、编译器和 GPU 架构一致，再重新安装扩展。下面的命令使用 Python wheel 中的
cuDNN 头文件和动态库；如果使用系统 cuDNN，应把 `CUDNN_ROOT` 改成实际路径。

```bash
cudnn_root=$(python -c \
  "from pathlib import Path; import nvidia.cudnn; print(Path(nvidia.cudnn.__path__[0]))")

CUDNN_ROOT="${cudnn_root}" \
CPATH="${cudnn_root}/include${CPATH:+:${CPATH}}" \
LIBRARY_PATH="${cudnn_root}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
LD_LIBRARY_PATH="${cudnn_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
MAX_JOBS=4 \
TORCH_CUDA_ARCH_LIST=8.0 \
NVTE_FRAMEWORK=pytorch \
  python -m pip install --force-reinstall --no-deps transformer-engine-torch==2.2.0
```

`TORCH_CUDA_ARCH_LIST=8.0` 对应 A100，H100 使用 `9.0`。`runtime.py` 仍会在导入
Transformer Engine 前预加载 Python wheel 内的 cuDNN 动态库；这只是运行时动态库发现，不负责
安装或编译依赖。

## 预训练资产

`train.sh` 固定从以下目录初始化新 Dream-Tac 训练：

```text
playground/pretrained_models/Cosmos-Predict2-2B-Video2World/
├── model-480p-16fps.pt        # 也支持 model.pt 或 model/model.pt
├── text_encoder/
│   ├── config.json
│   └── *.safetensors
└── tokenizer/
    ├── tokenizer_config.json
    ├── tokenizer.json         # 或 spiece.model
    └── tokenizer.pth          # Cosmos VAE
```

`pretrained_path` 统一表示权重来源。指向 Cosmos-Predict2 目录时，从底座初始化新布局；指向
StarVTLA Dream-Tac checkpoint 时，先根据 checkpoint 中保存的 Cosmos 路径构造 core，再加载
StarVTLA 保存的可训练参数。完整 `--resume` 还会恢复 optimizer、scheduler 和训练 step。

每个 checkpoint 都保存 slot layout fingerprint。基于 Dream-Tac checkpoint 继续训练时，camera、
tactile、state slot 的数量或顺序不一致会被拒绝；新布局必须重新从 Cosmos 底座初始化。

## 动态 Slot 布局

布局由 `wrist_only`、`wrist_camera_keys`、`top_camera_keys`、`tactile_mode`、`tactile_keys` 和
`state_mode` 编译。一个 checkpoint 的布局固定，不保留原版 Franka 12-slot preset。

```text
blank
[current proprio]
current wrist RGB ...
[current top RGB ...]
[current tactile ...]
action chunk
[future proprio]
future wrist RGB ...
[future top RGB ...]
[future tactile ...]
```

方括号项目是否存在由配置决定：

- `state_mode=none`：省略 current/future proprio。
- `wrist_only=true`：省略全部 top RGB。
- `tactile_mode=none`：省略全部 tactile。
- `tactile_mode=as_image`：每个 `tactile_keys` 成为独立图像 slot。
- 双臂不需要特殊 preset；左右腕相机和多路触觉按配置 key 顺序分别生成 slot。

Cosmos temporal compression factor 固定为 4，因此：

```text
pixel_frames = 1 + (slot_count - 1) * 4
```

action slot 位于 current 与 future 两组之间，之前的 slot 作为条件，action 以及之后的 slot 参与
future prediction。不同模态在布局中各占一个 temporal slot；低维 state/action 作为独立张量交给
Cosmos core 并注入对应 latent slot，因此不要求它们与 RGB 具有相同的原始 token 数量。

## State、Action 和触觉

Dream-Tac 支持 StarVTLA 已有的数据表示：

```text
state_mode:
  none | absolute_joint | episode_joint
  absolute_rot6d | episode_rot6d
  absolute_quat | episode_quat

action_mode:
  absolute_joint | relative_joint
  absolute_rot6d | relative_rot6d
  absolute_quat | relative_quat
```

单双臂的维数从 dataset feature schema 解析，不硬编码原版 Franka 的 6D state 或 7D action。
归一化、episode/relative 表示和部署端 action 恢复由 StarVTLA 通用 processor 完成。

Dream-Tac 仅支持 `tactile_mode=none|as_image`。在 `as_image` 下，模型的 current tactile 图像
slot 只输入当前时刻 `t`，future tactile slot 使用 `t + action_gap + 20` 的图像作为监督。数据层还
采样 `t-1`，但它只用于计算接触变化的 scalar self-attention gate，不会成为额外图像 slot。
Dream-Tac 不使用通用触觉历史窗口，所以 `TACTILE_NUM_FRAMES` 和 `TACTILE_FRAME_OFFSET` 不控制
上述采样；policy 配置中的 `tactile_num_frames=1`、`tactile_frame_offset=1` 固定不变。

`action_gap` 决定 action GT 起点，固定的 `chunk_size=20` 对应：

```text
action[t + action_gap : t + action_gap + 20]
future observation at t + action_gap + 20
```

`20` 是 action chunk 长度，不是 EDM 去噪次数。默认推理生成一个 20-step chunk，EDM 使用
`num_inference_steps=5`，部署队列默认执行其中 `n_action_steps=8` 个动作。

## 文本 Embedding 预计算

每个数据集需要在训练前生成本地 T5-11B 缓存：

```bash
python tools/precompute_world_model_text_embeddings.py \
  --dataset-root playground/data/<dataset_id> \
  --world-model dream_tac \
  --pretrained-path playground/pretrained_models/Cosmos-Predict2-2B-Video2World \
  --device cuda
```

脚本根据 `pretrained_path` 自动加载 `text_encoder/` 和 `tokenizer/`，并强制使用本地文件，不访问
Hugging Face。产物为：

```text
playground/data/<dataset_id>/text_embeddings/dream_tac/
├── manifest.json
└── embeddings.safetensors
```

`train.sh` 会复用完整缓存，缺失时自动执行同一预计算。Mixture 会为每个成员分别检查缓存，并在
加载时校验 task embedding 冲突。

## 训练

典型的多视角、触觉、rot6d 训练：

```bash
VISUALIZATION_ENABLED=true bash train.sh <dataset_id> dream_tac 4 1 50000 \
  false as_image absolute_rot6d relative_rot6d 6 strong
```

wrist-only、无触觉、joint 训练：

```bash
bash train.sh <dataset_id> dream_tac 4 1 50000 \
  true none absolute_joint absolute_joint 0 none
```

相机和触觉 key 默认根据数据集 feature 自动分类，也可以显式覆盖：

```bash
TOP_CAM='[observation.images.cam_top]' \
WRIST_CAM='[observation.images.left_cam_wrist,observation.images.right_cam_wrist]' \
TACTILE_KEYS='[observation.images.left_finger,observation.images.right_finger]' \
  bash train.sh <dataset_id> dream_tac 4 1 50000 \
    false as_image absolute_rot6d relative_rot6d 6 strong
```

## Loss 和推理

Cosmos core 先计算逐 slot weighted EDM loss，StarVTLA 再按 future slot 类型汇总。默认总 loss 为：

```text
2.00 * action_loss
+ 1.00 * future_rgb_loss
+ 1.00 * future_tactile_loss
+ 0.25 * future_state_loss
```

相应配置为 `action_loss_weight`、`rgb_loss_weight`、`tactile_loss_weight` 和
`state_loss_weight`。被当前布局省略的模态 loss 为 0。

生成 action 时使用与训练一致的 slot layout，从 action latent 解码固定 20-step chunk。
`action_start_offset` 和 `n_action_steps` 决定每次生成后实际加入部署队列的区间。

## Visualization Eval

`VISUALIZATION_ENABLED=true` 会启用训练期可视化。默认每 1000 step 保存 4 个样本，使用固定
seed 和 5 个生成步骤，输出到 run 目录下：

```text
visualizations/step_<step>/
├── *.png
└── metrics.json
```

结果包括每个 future RGB/tactile slot 的预测与 GT 对比、PSNR/SSIM、normalized action
MAE/RMSE 和推理耗时。相关配置为 `visualization_freq`、`visualization_num_samples`、
`visualization_num_inference_steps` 和 `visualization_seed`。

## 源码和许可证

StarVTLA adapter 位于当前目录，Cosmos Policy 上游来源、commit、Apache-2.0 许可证和本地兼容
修改记录见 [vendor/README.md](vendor/README.md)。

## 常见问题

- `ModuleNotFoundError: peft`：Dream-Tac 可选依赖未安装，按“环境安装”执行。
- `float != c10::BFloat16`：确认通过 DreamTacPolicy 调用 Cosmos core；adapter 会为训练、推理和
  visualization 建立与 core precision 一致的 autocast。
- T5 请求 Hugging Face 或出现 SSL 错误：确认 `pretrained_path/text_encoder` 和
  `pretrained_path/tokenizer` 文件完整，并重新运行本地预计算命令。
- slot fingerprint 不一致：当前 sensor 配置与 checkpoint 不同，需要恢复原布局或从 Cosmos
  底座开始新训练。
