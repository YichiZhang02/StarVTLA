# UniVTAC HDF5 → LeRobot v3

转换入口：`tools/convert_univtac_to_lerobot.py`。源数据保持不变；每个任务生成独立的本地 LeRobot 数据集。

## 输入与输出

- 输入：`playground/data/UniVTAC/isaac45/<task>/hdf5/*.hdf5`，共八个任务、800 个源 episode。
- UniVTAC 源码默认位于 `playground/simulations/UniVTAC`，脚本从其 `instructions/<task>.json` 读取 `seen` 任务文本。
- 输出：`playground/data/univtac_isaac45_<task>_lerobot/`。其下有标准 LeRobot v3 的 `meta/`、`data/`、`videos/`，以及转换清单 `univtac_conversion.json`。
- 运行全部任务：

```bash
HF_HOME=/tmp/starvtla_hf_univtac_convert \
python tools/convert_univtac_to_lerobot.py
```

只转换一个任务或少量 episode：

```bash
python tools/convert_univtac_to_lerobot.py --tasks lift_can --max-episodes 1
```

已有完整或部分输出时需显式加 `--resume`；脚本会检查输出与转换清单中的 episode 数和基本配置。不要将源 HDF5 目录作为 `--output-base`。

## 字段映射

| UniVTAC HDF5 | LeRobot 特征 | 处理 |
| --- | --- | --- |
| `embodiment/joint[t, :7]` | `observation.state[:7]` | Franka 七关节，float32 |
| `embodiment/joint[t, 7:9]` | `observation.state[7]` | 两 finger 位置取均值，单位 m |
| `embodiment/joint[t+1]` | `action` | 同样的 9→8 映射；下一**实测** qpos |
| `step[t]` | `observation.sim_step` | 保留源仿真步号 |
| `observation/head/rgb` | `observation.images.cam_top` | JPEG 解码、BGR→RGB、224×224 |
| `observation/wrist/rgb` | `observation.images.cam_wrist` | 同上 |
| 左、右 `tactile/*/rgb_marker` | `observation.images.left_cam_finger0`、`right_cam_finger0` | 同上 |

源数据每两步采样一次，仿真步长为 1/120 秒，因此输出 `fps=60`。每个源 episode 的最后一帧没有下一帧动作目标，故不写入。`lift_can/62.hdf5` 的 `step` 有一次回退；转换器在回退点切成两个连续 episode，避免跨断点配对，因此预计产出 **801 个 LeRobot episode、152079 帧**。

UniVTAC 源文件没有原始控制命令列。`action[t] = observation.state[t+1]` 沿用其 HDF5 批量读取代码的训练配对方式，但标签表示下一次采样时**达到的状态**，不是发往控制器的命令。后续做在线评测时必须考虑这个差别。

本次按 policy 所需字段转换。`actor`、`atom`、EE、触觉 `depth`、`marker`、`pose`、纯 `rgb` 不进入输出；源文件仍在 HDF5 中，完整列表写入每个数据集的 `univtac_conversion.json`。视频采用 H.264，源 JPEG 经解码及缩放后会重新编码，不是无损逐像素复制。

## 核验

转换完成后检查每个 `meta/info.json` 的 `total_episodes`、`total_frames`、`fps=60`、`robot_type=franka_panda`。可通过本仓库 `vtla.datasets.lerobot_dataset.LeRobotDataset` 读回首帧，核对 `action` 与源 HDF5 下一帧关节值。`HF_HOME` 应指向可写目录，避免 Hugging Face datasets 读取时在只读缓存目录创建锁文件。
