# ACT

`act` 是 StarVTLA 中从头训练的 Action Chunking Transformer。它用 ResNet 编码一个或多个
视觉视角，再由 Transformer 一次预测固定长度的 action chunk；默认启用 VAE 训练目标。

共享的数据集、`robot_type`、state/action 表示和相机路由约定见
[VTLA Training](../../README.md)。

## 最小环境

当前仓库验证基线为 Python 3.10.19、PyTorch 2.7.1+cu128、torchvision
0.22.1+cu128 和 CUDA 12.8。ACT 的直接建模依赖是 `torch`、`torchvision`、`einops` 和
`numpy`，训练入口还会使用根目录 `requirements.txt` 中的数据与训练依赖。

从仓库根目录安装：

```bash
conda create -n starvtla-act python=3.10 -y
conda activate starvtla-act
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

CPU 仅适合导入检查和小规模调试；正式训练应使用 CUDA。若机器不是 CUDA 12.8，保留
`torch==2.7.1` 与 `torchvision==0.22.1` 的对应关系，并从 PyTorch 官方索引选择匹配的 wheel。

安装检查：

```bash
python -c "import torch, torchvision, einops; from vtla.frameworks.act.modeling_act import ACTPolicy; print(torch.__version__)"
```

## 结构与默认值

```text
RGB / tactile images -> shared ResNet-18 -> Transformer encoder
state + action target -> VAE encoder      -> Transformer decoder -> action chunk
```

| 配置 | 默认值 | 含义 |
| --- | ---: | --- |
| `image_resolution` | `224x224` | 所有视角进入 ResNet 前的统一尺寸 |
| `chunk_size` | `32` | 每次预测的动作数 |
| `n_action_steps` | `16` | 每次部署实际执行的动作数 |
| `vision_backbone` | `resnet18` | torchvision 视觉骨干 |
| `use_vae` | `true` | 是否使用 ACT 的条件 VAE 目标 |
| `kl_weight` | `10.0` | KL loss 权重 |

默认 `pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1`。首次初始化时 torchvision
可能联网下载权重；离线机器应提前缓存该权重，或设置
`--policy.pretrained_backbone_weights=null` 从头训练视觉骨干。

## 训练

```bash
bash train.sh <dataset_id> act 1 32 20000 \
  true none absolute_joint absolute_joint 6 strong
```

EE action 示例：

```bash
bash train.sh <processed_dataset_id> act 1 32 20000 \
  true none absolute_rot6d relative_rot6d 6 strong
```

ACT 支持 `tactile_mode=as_image` 和 `encode`。`as_image` 把各路触觉帧作为额外视觉视角；
`encode` 把 Tactile MAE token 加入 Transformer encoder。启用 `encode` 时必须提供
`TACTILE_ENCODER_PATH`。

## 输入输出约束

- 至少需要一个经过路由的图像 feature，或 `observation.environment_state`。
- `state_mode=none` 可省略机器人 state；action feature 始终必需。
- `n_obs_steps` 固定为 `1`。
- `action_start_offset + n_action_steps` 不能超过 `chunk_size`。
- 启用 `temporal_ensemble_coeff` 时，`n_action_steps` 必须为 `1`。

## 推理

```bash
bash inference.sh <run_id> <step> async <robot_type> 16 0
```

checkpoint 会保存完整 policy 配置、processor、数据统计、任务文本和 `robot_type`。推理时不需要
再次提供 ResNet 初始化权重。
