# StarVLA-GR00T

`starvla_groot` 是 StarVTLA 注册的视觉语言动作 policy，由 Qwen vision-language backbone 和 GR00T flow-matching DiT action head 组成。

## 结构

```text
图像 + 任务文本
  -> Qwen AutoModelForImageTextToText
  -> multimodal hidden states
  -> GR00T flow-matching action head
  -> action chunk
```

主要文件：

| 文件 | 作用 |
| --- | --- |
| `configuration_starvla_groot.py` | 注册配置和输入输出契约 |
| `modeling_starvla_groot.py` | policy 前向、loss 和 action 采样 |
| `processor_starvla_groot.py` | pre/postprocessor 构建 |
| `qwen_vl_interface.py` | Qwen 多模态接口 |
| `action_head/` | GR00T flow-matching DiT |

配置使用 Draccus dataclass，不需要 YAML。共享的数据集、`robot_type`、state/action 和触觉约定见 [VTLA Training](../../README.md)。

## 最小环境

当前仓库验证基线为 Python 3.10.19、PyTorch 2.7.1+cu128、torchvision
0.22.1+cu128 和 CUDA 12.8。该 policy 的核心额外依赖是 `transformers==5.5.0`、
`diffusers==0.35.2` 和 `safetensors==0.7.0`，均已固定在仓库根目录的
`requirements.txt`。

```bash
conda create -n starvtla-groot python=3.10 -y
conda activate starvtla-groot
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

安装检查：

```bash
python -c "import torch, transformers, diffusers; from vtla.frameworks.starvla_groot.modeling_starvla_groot import StarvlaGrootPolicy; print(torch.__version__, transformers.__version__)"
```

默认 `dtype=bfloat16`，正式训练需要支持 BF16 的 CUDA GPU。非 CUDA 12.8 环境应从 PyTorch
官方索引选择匹配 wheel，并保持 torch/torchvision 版本配对。默认 attention 实现是 PyTorch
`sdpa`，不要求安装 FlashAttention；只有显式设置 `attn_implementation=flash_attention_2` 时才需要
额外安装与当前 torch/CUDA ABI 匹配的 `flash-attn`。

## 训练

推荐使用根目录包装脚本，它会从数据集自动推断相机 key，并把数据集 `robot_type` 写入 checkpoint：

```bash
bash train.sh <dataset_id> starvla_groot 1 4 20000 \
  true none absolute_joint absolute_joint 6 none
```

EE action 示例：

```bash
bash train.sh <processed_dataset_id> starvla_groot 1 4 20000 \
  true none absolute_rot6d relative_rot6d 6 none
```

基础 VLM 默认读取：

```text
playground/pretrained_models/Qwen3.5-0.8B
```

常用 policy override：

```text
--policy.action_model_type=DiT-B|DiT-L
--policy.repeated_diffusion_steps=8
--policy.num_inference_timesteps=4
--policy.train_expert_only=true
--policy.freeze_vision_encoder=true
--policy.gradient_checkpointing=false
```

`train_expert_only=true` 会冻结 VLM，仅训练 action head；需要视觉 backbone 适应当前相机域时保持为 `false`。

## 触觉

`tactile_mode=as_image` 将触觉作为额外图像视角，`tactile_mode=encode` 使用 Tactile MAE 生成 context。多帧触觉会按时间窗口拼接到 hidden states。

```bash
TACTILE_NUM_FRAMES=3 \
TACTILE_FRAME_OFFSET=2 \
TACTILE_ENCODER_PATH=playground/pretrained_models/AnyTouch-ViT-L-16 \
  bash train.sh <dataset_id> starvla_groot 1 4 20000 \
  true encode absolute_joint absolute_joint 6 none
```

## Transformers 兼容性

VLM 通过 `AutoModelForImageTextToText.from_pretrained()` 加载，本地 `transformers` 必须认识 checkpoint 的 `model_type`。Qwen3.5 权重若来自开发版 Transformers，应使用包含相同 `qwen3_5` 实现的版本；稳定版不识别该架构时会在加载阶段失败。

## 推理

```bash
bash inference.sh <run_id> <step>
```

推理从 checkpoint 自动恢复 policy 配置、任务文本和 `robot_type`。机器人构型与 B/ISF FK/IK 不能手动错配。
