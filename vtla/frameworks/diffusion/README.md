# Diffusion Policy

`diffusion` 是 StarVTLA 中从头训练的 conditional diffusion policy。它用 ResNet 提取多视角
图像特征，将视觉与 proprioception 作为 global condition，再用 1D conditional U-Net 去噪得到
action horizon。

共享的数据集、`robot_type`、state/action 表示和相机路由约定见
[VTLA Training](../../README.md)。

末端 state/action 统一使用 TCP rot6d：绝对位姿在基座系，`relative_rot6d` 为当前 TCP 系位移与
零中心相对旋转，反归一化后旋转全零表示不旋转，夹爪为绝对指令。整个 chunk 共用当前 TCP
锚点，`state_mode=none` 也需保留该隐藏锚点。关节模式不变，不再提供 `ee_frame` 或 quaternion
模型模式。旧 EE 数据需迁移、旧 EE checkpoint 需重训；定义和工具见
[TCP 数据与动作约定](../../../tools/TCP_ACTIONS.md)。

## 最小环境

当前仓库验证基线为 Python 3.10.19、PyTorch 2.7.1+cu128、torchvision
0.22.1+cu128 和 CUDA 12.8。该 policy 的额外核心依赖是 `diffusers==0.35.2` 与
`einops==0.8.1`，均已固定在根目录 `requirements.txt`。

```bash
conda create -n starvtla-diffusion python=3.10 -y
conda activate starvtla-diffusion
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

非 CUDA 12.8 环境应从 PyTorch 官方索引选择匹配 wheel，并保持 torch/torchvision 版本配对。
安装检查：

```bash
python -c "import torch, torchvision, diffusers, einops; from vtla.frameworks.diffusion.modeling_diffusion import DiffusionPolicy; print(torch.__version__)"
```

## 结构与默认值

```text
两步观测 -> shared ResNet -> visual features --+
robot state ------------------------------------+-> global condition
noisy action horizon + diffusion timestep ------+-> conditional 1D U-Net -> action horizon
```

| 配置 | 默认值 | 含义 |
| --- | ---: | --- |
| `n_obs_steps` | `2` | 当前帧和前一帧观测 |
| `horizon` | `32` | 去噪并预测的动作窗口 |
| `n_action_steps` | `16` | 每次部署执行的动作数 |
| `resize_imgs_to` | `224x224` | 多相机统一尺寸 |
| `num_train_timesteps` | `100` | 训练扩散步数 |
| `noise_scheduler_type` | `DDPM` | `DDPM` 或 `DDIM` |

视觉 ResNet 默认从头初始化，不需要额外预训练资产。

## 训练

```bash
bash train.sh <dataset_id> diffusion 1 64 20000 \
  true none absolute_joint absolute_joint 6 strong
```

EE action 示例：

```bash
bash train.sh <processed_dataset_id> diffusion 1 64 20000 \
  true none absolute_rot6d relative_rot6d 6 strong
```

触觉推荐使用 `tactile_mode=encode`，它把 Tactile MAE 输出作为 global condition：

```bash
TACTILE_ENCODER_PATH=<backbone_checkpoint> \
  bash train.sh <dataset_id> diffusion 1 32 20000 \
  true encode absolute_joint absolute_joint 6 strong
```

`as_image` 只支持单帧触觉；当 `TACTILE_NUM_FRAMES>1` 时配置会直接拒绝，因为普通相机已经占用
Diffusion 的共享 `n_obs_steps` 时间轴。

## 输入输出约束

- 至少需要一个经过路由的图像 feature，或 `observation.environment_state`。
- `horizon` 必须能被 `2 ** len(down_dims)` 整除；默认 `32` 与三层 U-Net 匹配。
- 可执行窗口上界为 `horizon - n_obs_steps + 1`。
- `action_gap` 同时作用于 horizon 的监督时间对齐；离线评估会从 checkpoint 恢复它。

相对 TCP 统计必须覆盖完整 horizon，起点为 `1-n_obs_steps+action_gap`。
上述默认 `n_obs_steps=2, horizon=32, action_gap=6` 的 EE 示例需在已迁移数据上先运行：

```bash
python tools/rebuild_relative_ee_stats.py \
  --root playground/data/<processed_dataset_id> --horizon 32 --offset-start 5
```

若 gap=0，应改为 `--offset-start -1`。整个预测窗口以最新观测 TCP 为固定锚点。
部署执行窗口会跳过 horizon 的历史位置；不要把原始 horizon 第 0 项当成首个执行目标。

## 推理

```bash
bash inference.sh <run_id> <step> async <robot_type> 16 0
```

需要缩短延迟时可在训练配置或 checkpoint 配置中设置 `num_inference_steps`，但应在离线评估后再
部署；减少反向扩散步数会改变动作质量。
