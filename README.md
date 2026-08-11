# tac_infra

A private infra for VTLA.

包含三块: **backbone 预训练**(触觉 MAE encoder)、**VTLA 策略训练**(act / diffusion / pi05 / starvla_groot)、以及在睿尔曼双臂上的 **deployment**(数据采集 + 推理)。

所有运行期路径均为**相对路径**, 统一挂在 `playground/` 下, 跨机器可移植:

```
playground/
├── pretrained_models/   # 预训练权重 (CLIP / AnyTouch / pi05_base / 底座 VLM)
├── data/                # 训练 / 预训练数据集 (LeRobot 格式或裸 frame_cache)
├── results/
│   ├── backbones/       # train_enc.sh 输出 (触觉 MAE checkpoint + 日志)
│   └── models/          # train.sh 输出 (VTLA 策略 checkpoint + 日志)
└── eval/                # 推理录制的评测数据集
```

---

# Quick start

## 1) 环境

```bash
conda create -n vtla python=3.10 -y
conda activate vtla
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 2) 下载预训练权重 -> `playground/pretrained_models/`

训练前需把以下权重放到 `playground/pretrained_models/`(目录名保持下表一致, 脚本按名引用)。
推荐用 `huggingface-cli download <repo_id> --local-dir playground/pretrained_models/<目录名>`。

| 目录名 | 用途 | 来源 |
| --- | --- | --- |
| `CLIP-ViT-L-14-DataComp.XL-s13B-b90K` | 触觉 MAE 的 CLIP 初始化 (ViT-L) | https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K |
| `CLIP-ViT-B-16-DataComp.XL-s13B-b90K` | 触觉 MAE 的 CLIP 初始化 (ViT-B, 需 HF 格式) | https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K |
| `AnyTouch-ViT-L-16` | 触觉 MAE 的 AnyTouch 初始化 / VTLA 触觉 encoder (仅 ViT-L) | AnyTouch 官方发布 (见 https://github.com/GeWu-Lab/AnyTouch ); 下载后转成 HF 目录或直接用 `.pth` |
| `pi05_base` | pi05 策略底座 (完整 policy 检查点) | https://huggingface.co/lerobot/pi05_base |
| `Qwen3.5-0.8B` | starvla_groot 底座 VLM (HF 模型目录) | https://huggingface.co/Qwen/Qwen3.5-0.8B |

> 说明: `pi05_base` 是**完整 policy 检查点**(含 `model.safetensors` + 预/后处理器), 通过 `--policy.pretrained_path` 加载整个 policy; `Qwen3.5-0.8B` 是 starvla_groot 的**底座 VLM**, 通过 `--policy.base_vlm` 传入, policy 本身从零训练。
> AnyTouch 只有 ViT-L 权重; CLIP-B-16 需要 HF 格式(bundled 的 open_clip 权重需先转换)。

## 3) 放置数据集 -> `playground/data/`

每个数据集放成 `playground/data/<dataset_id>/`, 训练脚本按 `dataset_id` 引用。支持两类:

- **LeRobot 数据集**: 含 `meta/info.json`(VTLA 训练 / 触觉 MAE 都可用), 如 `rm_umi_dual_pen_open`。
- **裸 frame_cache**: 无 LeRobot meta、单路相机(仅触觉 MAE 预训练用), 如 `pretrained_data`。

```
playground/data/
├── rm_umi_dual_pen_open/      # LeRobot 数据集
│   └── meta/info.json
└── pretrained_data/           # 裸 frame_cache (无 meta)
```

---

# Git usage

```bash
# 开始写代码前
git pull --rebase origin main

# 写完后
git add .
git commit -m "..."
git push origin main

# 另一台服务器同步
git pull origin main

# 强制同步 (丢弃本地改动)
git fetch origin
git reset --hard origin/main
```

---

# Training

## Backbone 预训练: 触觉 MAE (`train_enc.sh`)

基于 `vtla/tac_encoder/tactile_mae`, 直接吃 `playground/data/` 下的数据训练 AnyTouch stage1 风格的触觉 MAE encoder, checkpoint 与日志一起输出到 `playground/results/backbones/<run>/`。

```bash
# 用法: bash train_enc.sh <dataset_ids> <init_mode> <arch> <num_processes> <batch_size> <epochs>
bash train_enc.sh rm_umi_dual_pen_open clip vit_l
```

位置参数:

| 序号 | 参数 | 默认 | 说明 |
| --- | --- | --- | --- |
| 1 | `dataset_ids` | `pretrained_data` | 一个或多个数据集; 多个用引号包成空格分隔串, 如 `"ds1 ds2"`。**LeRobot 与裸 frame_cache 不能混跑** |
| 2 | `init_mode` | `clip` | `scratch`(随机) / `clip`(CLIP 初始化) / `anytouch`(AnyTouch 完整 MAE, 仅 ViT-L) |
| 3 | `arch` | `vit_b` | `vit_l` / `vit_b` |
| 4 | `num_processes` | `4` | >1 用 torchrun 多卡, =1 用单卡 python |
| 5 | `batch_size` | `128` | |
| 6 | `epochs` | `100` | |

常用环境变量: `GPU_ID`(默认 `0,1,2,3`)、`TACTILE_KEYS`(触觉图 key 列表 `[k1,k2,...]`)、`SENSOR_ID`、`MASK_RATIO`、`CONTACT_FILTER`(接触帧筛选, 默认开)、`IMAGE_SIZE`、`RAW_FRAME_CACHE`(强制裸缓存模式)。

> 数据模式自动识别: 含 `meta/info.json` 的为 LeRobot(先 warm-up 缓存再训练); 否则为裸 frame_cache(跳过 warm-up、关闭 contact_filter, 须用与构建缓存时一致的 `IMAGE_SIZE`)。
>
> 日志: 训练期间先写到系统临时目录, 正常跑完才搬进 `playground/results/backbones/<run>/` 与 checkpoint 同目录; 训练中途失败/中断则丢弃临时日志(不落盘)。

## VTLA 策略训练 (`train.sh`)

训练 act / diffusion / pi05 / starvla_groot 策略, checkpoint 与日志一起输出到 `playground/results/models/<run>/`(日志训练期间先写临时目录, 正常跑完才搬进与 checkpoint 同目录; 中途失败/中断则丢弃, 不落盘)。

```bash
# 用法: bash train.sh <dataset_id> <policy_type> <num_processes> <batch_size> <steps> \
#                      <wrist_only> <tactile_mode> <state_mode> <action_mode>
bash train.sh rm_umi_dual_pen_open diffusion
```

位置参数:

| 序号 | 参数 | 默认 | 说明 |
| --- | --- | --- | --- |
| 1 | `dataset_id` | `rm_umi_dual_pen_open` | `playground/data/<dataset_id>` |
| 2 | `policy_type` | `diffusion` | `act` / `diffusion` (从零) ・ `pi05` (载 `pi05_base`) ・ `starvla_groot` (载底座 VLM) |
| 3 | `num_processes` | `1` | accelerate 进程数 |
| 4 | `batch_size` | `32` | |
| 5 | `steps` | `10_000` | |
| 6 | `wrist_only` | `false` | `true` 只用 wrist 相机; `false` 用 top + wrist |
| 7 | `tactile_mode` | `none` | `none`(触觉不进模型) / `as_image`(触觉当图像输入) / `encode`(触觉 encoder) |
| 8 | `state_mode` | `joint` | `none` / `joint`(关节角) / `episode_rot6d` / `absolute_rot6d` / `episode_quat` / `absolute_quat` |
| 9 | `action_mode` | `joint` | `joint`(关节角) / `rot6d`(EE 相对动作, rot6d) / `quat`(EE 相对动作, 四元数) |

触觉 encoder (仅 `tactile_mode=encode`): 通过 `TACTILE_ENCODER_PATH`(默认 `playground/pretrained_models/AnyTouch-ViT-L-16`)指定 tactile-MAE 权重作为 encoder 初始化, `arch`/`sensor_id`/`image_size` 从 checkpoint 自动读取。`encode` 模式下 encoder + query token 会**随 policy 一起训练**(非冻结)。

#### 触觉时序窗口 (`tactile_num_frames` / `tactile_frame_offset`)

两个新参数，让 policy 看到多帧触觉历史（而不只是当前帧），与 RGB 观测窗口**完全独立**，`encode` 和 `as_image` 模式均生效：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TACTILE_NUM_FRAMES` | `1` | 每步喂给模型的触觉帧数（含当前帧）。`1` = 单帧 = 完全向后兼容。 |
| `TACTILE_FRAME_OFFSET` | `1` | 相邻两帧的采样间隔（帧数）。`1` = 相邻帧；`k` = 每隔 k 帧采样一次。 |

采样的 delta 索引为 `[-(F-1)*off, ..., -off, 0]`，例如 `F=3, off=2` 采样 `[-4, -2, 0]`（相对当前步）。

**训练**：dataset 的 `delta_timestamps` 自动按上述索引拉取触觉帧，每个 finger 相机仍只解码一次视频（按 delta 多取帧，无额外 I/O）。触觉 key 收到的 tensor 形状为 `[B, F, C, H, W]`（`F=1` 时退化为 `[B, C, H, W]`，与现有行为完全一致）。

**推理**：preprocessor pipeline 中自动插入 `TactileTemporalWindowStep`，维护每路 finger 的滑动帧缓冲；episode 切换时调用 `reset()` 清空。

**各 policy 行为**：

| Policy | encode 多帧 | as_image 多帧 |
|---|---|---|
| ACT | ✅ 展平为额外 token（positional embed 自动扩大） | ✅ 每帧作为独立相机输入 |
| pi05 | ✅ 展平为额外 prefix token | ✅ 每帧作为独立相机 |
| starvla_groot | ✅ 展平后 cat 到 hidden states | ✅ 每帧作为独立相机 |
| diffusion | ✅ 展平到 global conditioning（独立于 n_obs_steps） | ❌ 不支持（diffusion 将所有相机对齐到 n_obs_steps 轴，多帧触觉无法加入该轴；请改用 encode 模式） |

示例（3 帧触觉，间隔 1 帧，encode 模式）：

```bash
TACTILE_NUM_FRAMES=3 TACTILE_FRAME_OFFSET=1 \
  bash train.sh rm_umi_dual_pen_open act 8 16 50000 false encode
```

示例（4 帧触觉，间隔 2 帧，as_image 模式，pi05）：

```bash
TACTILE_NUM_FRAMES=4 TACTILE_FRAME_OFFSET=2 \
  bash train.sh rm_umi_dual_pen_open pi05 8 16 50000 false as_image
```

相机 / 触觉 key (draccus 列表 `[k1,k2,...]`, 按数据集命名用环境变量覆盖):

- `TOP_CAM` (默认 `[observation.images.cam_top]`)
- `WRIST_CAM` (默认 `[observation.images.left_cam_wrist,observation.images.right_cam_wrist]`)
- `TACTILE_KEYS` (默认四路 finger: `left/right × finger0/1`)

示例 (双臂 + 触觉作为图像输入, 4 卡):

```bash
bash train.sh rm_umi_dual_pen_open diffusion 4 32 20000 false as_image joint
```

### 末端位姿 (EE) 模式

除关节角外，state / action 还支持末端位姿 (end-effector)，同时兼容两种旋转表达：

#### 旋转格式

| 格式 | 每臂维度 | 双臂总维度 | 旋转部分 |
|---|---|---|---|
| `rot6d` (默认) | 10 | 20 | 旋转矩阵前两列 (Zhou 2019), 连续且唯一 |
| `quat` | 8 | 16 | 四元数 `[x, y, z, w]` (scalar-last/ROS 惯例) |

#### 坐标系 × 旋转格式 → `state_mode` 可选值

| state_mode | 坐标系 | 旋转格式 | 说明 |
|---|---|---|---|
| `episode_rot6d` | episode 相对 | rot6d | 相对每个 episode 首帧 T0⁻¹·Tt |
| `absolute_rot6d` | 机器人 base | rot6d | 直接 FK 结果 Tt，保留绝对工作空间位置 |
| `episode_quat` | episode 相对 | quat | 同上，旋转用四元数 |
| `absolute_quat` | 机器人 base | quat | 同上，旋转用四元数 |

旧名称 `episode_ee` / `absolute_ee` 作为别名仍然被接受（等同于 `episode_rot6d` / `absolute_rot6d`）。

#### `action_mode` 可选值

| action_mode | 旋转格式 | 说明 |
|---|---|---|
| `rot6d` | rot6d | 动作是相对当前观测的 EE 位姿 St⁻¹·S_{t+k}，rot6d 表示 |
| `quat` | quat | 同上，旋转用四元数 |

旧名称 `relative_ee` 等同于 `rot6d`。

**约束**：`state_mode` 与 `action_mode` 的旋转格式必须一致（同为 rot6d 或同为 quat）。

**前置（一次性）**：先把关节数据集离线转出 EE 列（FK + 四元数转换，无需连机械臂）。  
`convert_joints_to_eepose.py` 会**同时生成** rot6d 和 quat 两套列及各自的归一化统计，原关节列保持不变：

```bash
python tools/convert_joints_to_eepose.py --root playground/data/rm_umi_dual_pen_open
#   --horizon 32   # relative action 统计的最大 chunk 步长; 训练 chunk_size 须 <= 该值
```

生成的列：

| 列名 | 维度 | 说明 |
|---|---|---|
| `observation.state_episode_ee` | 20 | rot6d, episode 相对 |
| `action_episode_ee` | 20 | rot6d, episode 相对（state 的未来轨迹） |
| `observation.state_absolute_ee` | 20 | rot6d, base 坐标系 |
| `action_absolute_ee` | 20 | rot6d, base 坐标系 |
| `observation.state_episode_quat` | 16 | quat, episode 相对 |
| `action_episode_quat` | 16 | quat, episode 相对 |
| `observation.state_absolute_quat` | 16 | quat, base 坐标系 |
| `action_absolute_quat` | 16 | quat, base 坐标系 |
| `action_relative_ee` | 20 | stats-only，rot6d 相对动作归一化统计 |
| `action_relative_quat` | 16 | stats-only，quat 相对动作归一化统计 |

EE 模式训练示例（rot6d，pi05）：

```bash
bash train.sh rm_umi_dual_pen_open pi05 1 32 10000 false none episode_rot6d rot6d
```

EE 模式训练示例（quat，diffusion）：

```bash
bash train.sh rm_umi_dual_pen_open diffusion 4 32 20000 false none episode_quat quat
```

> 推理 (EE 模式) 的硬件下发链路 (FK 读 state + IK/`rm_movep_canfd` 发 action + 首帧绝对位姿缓存) 仍在接入中, 见 `TODO`。训练侧已全部可用。

---

# Tools

仓库根的 `tools/` 放训练/数据相关的离线小工具。

## 降分辨率数据集 (`tools/downscale_dataset_videos.py`)

训练时相机帧在 CPU 上逐步解码, 而 RGB 相机原生分辨率很大(如 1920×1080), policy 又会把每帧 resize 到 ~224。解码全分辨率帧再丢掉像素会让 data loader 成为瓶颈(快模型会被 GPU 拖到周期性 stall)。该脚本**非破坏地**生成一份降分辨率副本: 只重编码大的 8-bit RGB 相机视频到 `SIZE×SIZE`(默认 256, 给 224 裁剪留余量), 帧数/fps/时间戳不变、用小 GOP 保证随机 seek 便宜; **触觉 finger 相机(16-bit 无损 .mkv)原样拷贝不动**, 同时 patch `meta/info.json` 里对应 feature 的 shape/分辨率/codec。

```bash
python tools/downscale_dataset_videos.py \
    --src playground/data/rm_umi_dual_pen_open \
    --dst playground/data/rm_umi_dual_pen_open_256 \
    --size 256
```

之后训练用 `--dataset.root=<dst>`(repo_id 不变)。降分辨率目标自动选取(全局 .mp4 路径、8-bit、非无损、短边 > size), 可用 `--cameras` 强制指定; 常用参数 `--crf`(质量)、`--gop`(seek 速度)、`--jobs`(并行)、`--verify`(ffprobe 校验帧数一致)。需要 `ffmpeg`/`ffprobe` 在 PATH 上。

## 鱼眼去畸变数据集 (`tools/undistort_dataset_videos.py`)

UMI 腕部相机录制的是全幅鱼眼(1920×1080, Kalibr equidistant/OpenCV fisheye 模型), 训练需要的是去畸变后的正方形中心裁剪(896×896, 即 `..._umistyle` → `..._umistyle_undist`)。该脚本**非破坏地**生成去畸变副本: 每帧 `解码 → cv2.fisheye 去畸变(新内参=K, 不额外缩放) → 居中裁 896×896(不再 resize) → 重编码`, 帧数/fps/时间戳不变、小 GOP 保证随机 seek 便宜; **只处理腕部相机, 触觉 finger 相机(16-bit 无损 .mkv)及其它相机原样拷贝不动**, 同时 patch `meta/info.json` 的 shape/分辨率/codec。流程与 `ugripper/zxd_fisheye/undistort_wrist.py` 逐像素一致。

标定文件已内置在 `tools/calib/x5_{left,right}_intrinsics.json`(从 `ugripper/zxd_fisheye` 拷入), 默认自动加载, 一般无需指定 `--calib`。

```bash
# 快速可视化自检: 每个腕相机抽 1 帧, 输出 原图/去畸变带裁剪框/最终896 三张 PNG + 短 clip 到 tools/undistort_test/
python tools/undistort_dataset_videos.py --src playground/data/rm_umi_dual_pen_open --test

# 全量处理
python tools/undistort_dataset_videos.py \
    --src playground/data/rm_umi_dual_pen_open \
    --dst playground/data/rm_umi_dual_pen_open_undist
```

标定 JSON 格式 (Kalibr equidistant):

```json
{
    "distortion_model": "equidistant",
    "camera_matrix":     [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "distortion_coeffs": [k1, k2, k3, k4],
    "resolution":        [width, height]
}
```

默认去畸变 `observation.images.{left,right}_cam_wrist`(可用 `--cameras` 覆盖)。`--calib` 不填用内置标定; 也可给单个 json 或 `--calib left_cam_wrist=l.json right_cam_wrist=r.json`。常用参数 `--crop`(裁剪边长, 默认 896)、`--crf`、`--gop`、`--jobs`、`--verify`(ffprobe 校验帧数一致)。`meta/stats.json` 的逐通道图像统计**保持不变**(去畸变+裁剪是几何 warp, 基本保持均值/方差, 与 downscale 同理)。需要 `ffmpeg`/`ffprobe` 及 `opencv-python`。

> **顺序很重要: 先去畸变(原生分辨率)再降采样, 不要反过来。** 在原生 1920×1080 上去畸变并裁 896, 再降到训练分辨率, 才能保留细节和正确的中心裁剪 FOV; 直接对已降采样的 256 数据集去畸变会糊且 FOV 不对(挤压后的全鱼眼视野, 而非 896 中心裁剪)。
>
> ```bash
> # 1) 原始数据集 → 896 去畸变
> python tools/undistort_dataset_videos.py \
>     --src playground/data/rm_umi_dual_260617_pen_place_cap_notop \
>     --dst playground/data/rm_umi_dual_260617_pen_place_cap_notop_undist
>
> # 2) 896 去畸变 → 256 (复用降分辨率工具)
> python tools/downscale_dataset_videos.py \
>     --src playground/data/rm_umi_dual_260617_pen_place_cap_notop_undist \
>     --dst playground/data/rm_umi_dual_260617_pen_place_cap_notop_undist_256 --size 256
> ```
>
> A100 等数据中心卡**无 NVENC 硬编**, 用默认 `libx264`(CPU); 仅消费级/带 NVENC 的卡才用 `--codec hevc_nvenc`。

---

# Deployment

在睿尔曼 RM75b 双臂上做数据采集与策略推理。所有命令从仓库根运行(脚本会自动 `cd` 到根)。

## 1) 厂商 SDK 位置

硬件封装在 `import` 厂商库前统一调用 `deployment/hardware/_sdk_paths.py` 的 `ensure_*` 把 SDK 加到 `sys.path`。SDK 统一放在 `deployment/sdk/` 下:

```
deployment/sdk/
├── Robotic_Arm/          # 睿尔曼机械臂 SDK     -> import Robotic_Arm.*   (已内置)
│   └── libs/linux_x86/libapi_c.so
├── dm_lingkong_grip/     # 领控电爪客户端       -> import dm_lingkong_grip_sdk.*
├── fish_camera_client/   # 鱼眼相机 gRPC 客户端 -> 扁平 import
└── dmrobotics/           # Flux 触觉传感器 SDK  -> import dmrobotics.*
```

> 仓库已内置 `Robotic_Arm`(含 linux_x86 的 `libapi_c.so`)。其余三个 SDK(电爪 / 鱼眼相机 / 触觉)需按上面的目录名放入 `deployment/sdk/`; 目录缺失时 `_sdk_paths` 静默跳过, 由对应硬件 import 时报错。

## 2) Robot config

机器人配置类在 `deployment/robots/<robot_type>/config_<robot_type>.py`, 通过 `@RobotConfig.register_subclass("<robot_type>")` 注册, 用 `--robot.type=<robot_type>` 选择。已注册的 type 见 `deployment/robots/`(如 `realman_ugripper_dual`、`realman_ugripper_dual_notac`、`realman_ugripper_dual_notop_notac` 等)。

以 `realman_ugripper_dual` 为例, 上机前**重点核对**(详见 [config_realman_ugripper_dual.py](deployment/robots/realman_ugripper_dual/config_realman_ugripper_dual.py)):

- **从臂 IP/端口**: `left_follower_ip=192.168.1.200` / `right_follower_ip=192.168.1.201`, `follower_tcp_port=8080`
- **每臂板子 IP**(鱼眼/触觉/夹爪代理): `left_board_ip=192.168.1.10` / `right_board_ip=192.168.1.11`
- **本机 IP**: `pc_host=192.168.1.120`(触觉 UDP 回传用)
- **夹爪满行程**: `left_gripper_itinerary`/`right_gripper_itinerary`(左右传动比不同, 已实测覆盖)
- **触觉/鱼眼输出尺寸与编码范围**: `tactile_width/height`、`tactile_depth/deform_min/max`(须与训练时一致)
- **启用的手臂与硬件开关**: `arms=["left","right"]`、`use_tactile`、`cameras`(顶部相机, 空字典即无 top)

config 字段都可在命令行用 `--robot.<字段>=<值>` 覆盖, 如 `--robot.use_tactile=false`、`--robot.left_follower_ip=...`。`--robot.id` 用于区分同型号的不同机器人(影响标定文件目录)。

## 3) Pre-test (硬件自检)

```bash
# 1) 存在性检查 (默认, 最安全, 不连硬件)
python -m deployment.tools.hardware_check

# 2) 图像检查 (实际连相机/触觉抓一帧, 可存图)
python -m deployment.tools.hardware_check --stage camera --show

# 3) 主从同步 (⚠️ 会驱动从臂, 必须显式确认)
python -m deployment.tools.hardware_check --stage teleop --confirm-move
```

## 4) 数据采集

推荐用 `collect.sh`(自动按时间命名 `local/<时间戳>_<name>`, 默认存到 `playground/data/<repo_id 末段>`):

```bash
# 用法: bash collect.sh <name> <single_task> <num_episodes>
bash collect.sh rm_tactile_demo "抓笔" 30
```

或直接调脚本:

```bash
python -m deployment.collect \
    --robot.type=realman_ugripper_dual \
    --robot.id=realman_dual \
    --teleop.type=bi_realman_ugripper_leader \
    --dataset.repo_id=local/$(date +%Y%m%d_%H%M%S)_grab_pen \
    --dataset.single_task="抓笔" \
    --dataset.num_episodes=50 \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=15 \
    --robot.use_tactile=true \
    --dataset.push_to_hub=false
```

> 采集中可中途按右键提前保存当前 episode; `episode_time_s` 为单集最长录制秒数, `reset_time_s` 为集间复位场景时间。

## 5) 模型推理

推荐用 `inference.sh`(按 `pretrained_id` + `step` 自动拼 checkpoint 路径, `--match_policy` 自动对齐硬件 + 任务):

```bash
# 用法: bash inference.sh <pretrained_id> <step> [n_action_steps] [action_start_offset]
bash inference.sh rm_umi_dual_pen_open_diffusion_wristonly_false_tactile_none_state_joint 5000
```

`pretrained_id` 即 `playground/results/models/` 下的目录名, 实际加载 `playground/results/models/<pretrained_id>/checkpoints/<step 补零6位>/pretrained_model`。

位置参数 (`$3`/`$4` 留空则完全用 checkpoint config.json 里的值, 不覆盖):

| 序号 | 参数 | 默认 | 说明 |
| --- | --- | --- | --- |
| 3 | `n_action_steps` | *(checkpoint 默认)* | 每次推理执行几个 action 再重规划(`chunk[:n]`)。调小可提高响应性但推理更频繁。 |
| 4 | `action_start_offset` | *(checkpoint 默认, 通常 0)* | 执行前丢掉 chunk 前 m 个动作 (`chunk[m:]`)。实际执行 `chunk[m : m+n]`。用于补偿模型推理延迟 —— 推理期间机器人已执行了若干步, 丢掉对应的陈旧开头动作。须满足 `m + n ≤ chunk_size`。 |

```bash
# 覆盖 n_action_steps=8 (默认 chunk 仍从头开始)
bash inference.sh <id> 5000 8

# 同时覆盖 n_action_steps=8 且丢掉 chunk 前 4 个动作 -> 执行 chunk[4:12]
bash inference.sh <id> 5000 8 4
```

或直接调脚本:

```bash
# 自动对齐 (match_policy 从 checkpoint 读取硬件/任务配置)
python -m deployment.inference \
    --robot.type=realman_ugripper_dual \
    --policy.path=<path to pretrained_model> \
    --dataset.repo_id=local/eval_$(date +%Y%m%d_%H%M%S)_pen \
    --match_policy=true

# 手动模式 (match_policy=false 时, 硬件/任务需自己给)
python -m deployment.inference \
    --robot.type=realman_ugripper_dual \
    --policy.path=<path to pretrained_model> \
    --dataset.repo_id=local/eval_pen \
    --match_policy=false \
    --robot.use_tactile=false \
    --dataset.single_task="Grasp the cap and pull it off the pen."
```

> 推理录制的评测数据集默认落到 `playground/eval/<repo_id 末段>`。

### home_joints 设置

`--robot.home_joints` 指定每次推理开始前机械臂复位的目标关节位置 (弧度)。初始值可用工具脚本从当前硬件实时读取:

```bash
# 连接双臂, 读取当前关节位置, 输出可直接贴入 inference.sh 的参数行
python -m deployment.tools.read_home_joints

# 只读左臂 / 只读右臂
python -m deployment.tools.read_home_joints --side left
python -m deployment.tools.read_home_joints --side right

# 自定义 IP (默认 left=192.168.1.200, right=192.168.1.201)
python -m deployment.tools.read_home_joints --left-ip 192.168.1.200 --right-ip 192.168.1.201
```

stdout 直接输出 `--robot.home_joints='...'` 字符串, 复制后替换 `inference.sh` 第 26 行即可; stderr 逐关节打印数值方便核对。

### 腕部鱼眼去畸变 (训练-推理一致)

若模型是在去畸变数据集上训练的(腕部=鱼眼去畸变+居中裁 896), 推理时必须对原生鱼眼帧做**相同**变换, 否则几何不一致(训练-推理 gap)。`--match_policy` 会**自动判定并开启**, 无需手动:

1. **marker 优先**: 经 checkpoint 的 `train_config.json` 找到训练集, 若其 `meta/info.json` 有 `undistort` 标记(由 `tools/undistort_dataset_videos.py` 写入) → 开启, 并采用其中的 `crop`;
2. **名称兜底**: 训练集不可访问时, 看 `repo_id`/`root` 是否含 `undist`;
3. 都不满足 → 关闭。

去畸变在 `get_observation` 内对腕部帧即时完成(`deployment/hardware/wrist_cameras/undistort.py`, 与 tools 逐像素一致, 标定内置在 `.../calib/x5_{left,right}_intrinsics.json`), 输出 896×896, 最终 224 由 policy 的 `resize_imgs_to` 完成; 录下的评测视频也是去畸变 896, 与 policy 输入一致。

手动覆盖(`match_policy=false` 或想强制):

```bash
--robot.undistort_wrist=true     # 强制开 (false 强制关; 默认 auto)
--robot.undistort_crop=896       # 裁剪边长 (须与训练 crop 一致)
```

> 采集(`collect`)/遥操作无 policy, `auto` 即**关闭**, 存原生鱼眼, 之后用 `tools/undistort_dataset_videos.py` 离线去畸变 —— 切勿在采集端就去畸变(会丢失原始数据)。

---

## 关节数据集预处理流水线 (`process_joint_data.sh`)

对从机械臂关节角采集的原始数据集做**三步串联预处理**: 去畸变 → 降分辨率 → 关节转末端位姿(FK)。非破坏, 逐级生成新副本, 原始数据集不动。

```bash
# 用法: bash process_joint_data.sh <dataset_id> [size] [horizon]
bash process_joint_data.sh rm_umi_dual_260707_pen_in_case_1
```

位置参数:

| 序号 | 参数 | 默认 | 说明 |
| --- | --- | --- | --- |
| 1 | `dataset_id` | *(必填)* | `playground/data/<dataset_id>` |
| 2 | `size` | `256` | 降分辨率目标边长(给 224 裁剪留余量) |
| 3 | `horizon` | `32` | `action_relative_ee` 统计的最大 chunk 步长; 训练 `chunk_size` 须 ≤ 该值 |

产物:
```
<id>                 (原始, 不动)
  -> <id>_undist           鱼眼去畸变 + 居中裁 896
  -> <id>_undist_<size>    降到 size×size  ← 训练用这个 (就地加 EE 列)
```

可选环境变量: `CROP`(去畸变裁剪边长, 默认 `896`)、`JOBS`(ffmpeg 并行数, 默认 `12`)、`CAMERAS`、`CALIB`(同 `undistort_dataset_videos.py`)。已存在的中间目录自动跳过(幂等)。

## UMI 数据集预处理流水线 (`process_umi_data.sh`)

与 `process_joint_data.sh` 完全同流程, 但第 3 步换成 `convert_umi_to_eepose`(UMI 数据本身已存末端位姿, 跳过 FK; 若 `meta/episodes` 缺失会自动从帧数重建)。其余参数、产物目录命名规则、环境变量完全一致。

```bash
# 用法: bash process_umi_data.sh <dataset_id> [size] [horizon]
bash process_umi_data.sh rm_umi_dual_260707_pen_in_case_1
```

> **dataset_id 必填**(`process_joint_data.sh` 有内置默认值; 本脚本要求显式指定以避免误操作)。

## 多数据集合并 (`merge_datasets.sh`)

对多个特征不一致的数据集取特征**交集对齐**后合并成单一训练数据集。典型场景: 无触觉数据 + 有触觉相机数据同时训练(后者多出 finger 相机 feature, 直接 aggregate 报错)。

```bash
# 用法: bash merge_datasets.sh <out_id> <src_id_1> <src_id_2> [<src_id_3> ...]
bash merge_datasets.sh merged_notac_undist_256 \
    rm_umi_dual_260706_pen_in_case_notac_undist_256 \
    rm_umi_dual_260706_pen_in_case_1_notac_undist_256
```

产物 `playground/data/<out_id>` 非破坏生成(输出目录已存在时报错拒绝覆盖)。合并后按 10 MB 分片重编码视频(保持小 mp4 布局, 避免训练随机取帧慢)。合并结果是 notac 数据集, 用 `tactile_mode=none` 训练。

> 所有 id 均为 `playground/data/` 下的目录名(不含路径)。
