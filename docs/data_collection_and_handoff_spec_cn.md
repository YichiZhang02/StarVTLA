# StarVTLA 关节数据采集、处理与交付规范

> 文档用途：发给数据采集方/数据提供方，指导其将数据整理成 StarVTLA 可直接训练的、与 `scripts/process_joint_data.sh` 最终产物一致的格式。
>
> 规范版本：2026-08-24。本文以仓库当前代码和 `process_joint_data.sh` 为唯一实现基准。

## 1. 最终交付要求（先看本节）

数据提供方应交付一个**完整的 LeRobot v3.0 数据集目录**，而不是若干独立视频、CSV 或图片文件。最终目录必须已经完成：

1. 腕部鱼眼相机去畸变和中心裁剪；
2. 触觉数据从规范的 `uint16` 定点编码转换为训练用 `uint8`；
3. 所有训练视频缩放到 `256 x 256`；
4. 关节数据通过正确的 RealMan 正运动学（FK）生成 episode-relative 和 absolute EE pose；
5. 全局统计与逐 episode 统计更新完成。

当前默认交付参数如下。除非接收方书面指定其他值，不得自行修改。

| 项目 | 要求 |
| --- | --- |
| 数据标准 | LeRobot `v3.0` |
| 统一数据频率 | `30 Hz`（每步 `1/30 s = 0.033333... s`） |
| 最终视频分辨率 | 所有视觉和触觉视频均为 `256 x 256 x 3` |
| 腕部去畸变裁剪 | 原始标定分辨率上去畸变，再中心裁剪 `896 x 896` |
| 视频输出 | MP4，H.264，`yuv420p`，无音频，`30 fps` |
| 数值列 | Parquet 中的 fixed-size list，元素为 `float32` |
| 关节单位 | 弧度（rad） |
| EE 位置单位 | 米（m），在对应机器人基座坐标系或 episode 首帧局部坐标系中 |
| 夹爪 | 归一化到 `[0, 1]`，`0=闭合/最紧`，`1=张开`；relative 表示中仍保持绝对值 |

标准处理命令：

```bash
bash scripts/process_joint_data.sh <dataset_id> 256
```

这里省略的 `horizon` 和 `action_gap` 只用于生成训练所需的 relative-action 归一化统计，**不属于采集格式或逐帧处理后数据格式**。脚本省略这两个参数时使用自己的默认统计窗口；接收方可在训练前按实际训练配置重新计算。

源数据放在 `playground/data/<dataset_id>/`；有触觉时最终目录为：

```text
playground/data/<dataset_id>_undist_uint8_256/
```

无触觉时最终目录为：

```text
playground/data/<dataset_id>_undist_256/
```

### 1.1 必须随数据一起提供的信息

除最终数据集目录外，交付包中还必须有一份纯文本 `DELIVERY.md`，至少记录：

- 数据集名称、采集日期、任务描述、episode 数量和总帧数；
- 使用的仓库 commit ID；
- 完整处理命令，包括 `size`、`CROP`、`CAMERAS` 和 `CALIB`；若处理时生成过 relative-action 统计，可附带记录当时的 `horizon/action_gap`，但它们不是交付格式要求；
- 机器人型号、机械臂侧别、末端工具安装情况；
- 相机型号、序列号及每个 camera key 对应的物理相机；
- 腕部相机内参/畸变标定 JSON；若不是本项目现有相机，必须同时交付标定文件；
- 是否含触觉、触觉型号、finger0/finger1 的物理安装位置和朝向；
- 采集异常、丢帧、重复帧、网络中断、失败 episode 的处理记录；
- 最终目录的 SHA-256 清单。

建议同时保留并交付未经处理的原始数据集。最终触觉视频不可逆，原始 `uint16` MKV 是数值审计和重新量化的唯一可信来源。

## 2. 支持的机器人和硬性兼容条件

当前 `process_joint_data.sh` 只支持以下两种严格定义的机器人构型：

| `robot_type` | 构型 | FK 变体 | 必须存在的臂 |
| --- | --- | --- | --- |
| `rm_base_umi_dual` | RealMan RM75-B 双臂 | `base` | right + left |
| `rm_isf_umi_left` | RealMan RM75-ISF 单左臂 | `isf` | left |

`meta/info.json` 中的 `robot_type` 决定 FK 模型，不存在命令行覆盖。以下做法均不允许：

- 将其他机器人数据的 `robot_type` 字符串改成上述值；
- 用 ISF FK 处理 B 型机器人，或反过来；
- 将只有一条臂的数据标记成 `rm_base_umi_dual`；
- 使用 TCP/tool pose 冒充本项目的 flange pose；
- 将角度制关节值直接写入要求为弧度制的列。

本项目 FK 使用 RM75-E 模型，输入 7 个关节角，内部由 rad 转 degree 后调用 RealMan SDK `rm_algo_forward_kinematics(..., flag=0)`。输出位姿是**机器人基座坐标系下的 flange 位姿，不附加 TCP 外参**。

若数据来自其他机器人、不同连杆参数、不同 base 坐标定义或不同 TCP 定义，即使数组维度相同，也不能直接视为兼容数据。必须先增加对应的机器人配置和 FK 适配，并由接收方验证。

## 3. 频率、时间轴与同步方式

### 3.1 逻辑数据频率

最终数据集只有一条统一逻辑时间轴，固定为 `30 Hz`。每个 episode 内：

```text
frame_index = 0, 1, 2, ..., L-1
timestamp   = frame_index / 30.0
```

因此每个 episode 的 `timestamp` 都从 `0.0` 重新开始。例如：

```text
frame 0 -> 0.000000 s
frame 1 -> 0.033333 s
frame 2 -> 0.066667 s
```

`index` 是跨全部 episode 连续递增的全局帧号；`episode_index` 从 0 连续递增；`frame_index` 在每个 episode 内重新从 0 开始。

### 3.2 传感器采集频率与落盘频率的区别

项目当前硬件采用“异步采集、30 Hz 主循环取最新值”的方式：

- 腕部相机源通常约 `50-59 fps`，主循环在每个 30 Hz 时刻取最新一帧；
- Flux 触觉源原生约 `110 fps`，当前接收端限制到 `30 fps`，主循环取最新一帧；
- 机器人关节状态由后台线程持续更新，主循环读取最近状态；
- 最终所有模态都按统一的 `30 Hz` 逻辑时间轴写入。

数据提供方不必复制相同的内部线程实现，但最终必须满足以下结果：

- 每个逻辑时刻都有 state、action 和每个必需相机的有效样本；
- 所有模态对应同一物理时刻，或使用明确、稳定的最近邻同步规则；
- 不得把不同频率的数据仅通过修改视频 FPS 标签伪装成 30 Hz；
- 不得在没有说明的情况下大量重复旧帧或线性插值动作；
- 建议原始视觉/触觉源频率不低于 30 Hz，并记录设备时间戳供审计。

当前 LeRobot 表中的 `timestamp` 是规则化的逻辑时间，不是设备原始硬件时间。若有设备时间戳，建议在补充审计文件中保留。

### 3.3 state 与 action 的时间语义

在第 `t` 行：

- `observation.state[t]` 是当前测得的机器人关节状态；
- `action[t]` 是该控制周期的目标/监督动作，字段顺序必须与 state 完全一致；
- 不要默认把 `action[t]` 设置成 `observation.state[t+1]`；
- drag 示教模式下，关节 action 通常取当前实测关节，夹爪 action 取当时的夹爪目标；普通遥操作模式下 action 是遥操作器产生并经过动作处理器的关节目标。

若外部系统记录的是其他时刻定义（例如“上一周期下发值”或“下一周期目标”），必须先转换为上述同周期语义，并在 `DELIVERY.md` 说明。

## 4. 处理前：原始数据格式

### 4.1 目录结构

原始数据必须是完整的 LeRobot v3.0 数据集：

```text
<dataset_id>/
├── data/
│   └── chunk-000/
│       └── file-000.parquet
├── videos/
│   ├── observation.images.<camera_key>/
│   │   └── chunk-000/
│   │       ├── file-000.mp4（普通相机）
│   │       └── ...
│   └── observation.images.<tactile_key>/
│       └── chunk-000/
│           ├── file-000.mkv（原始触觉）
│           └── ...
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.parquet
    ├── tactile_encoding.json（有触觉时必须存在）
    └── episodes/
        └── chunk-000/
            └── file-000.parquet
```

文件可以按 `info.json` 中的大小限制拆成多个 `file-NNN`，不能假设一个 episode 对应一个视频文件。episode 到数据/视频文件及时间区间的映射以 `meta/episodes/*.parquet` 为准。

### 4.2 单左臂关节布局

当 `robot_type = "rm_isf_umi_left"` 时，`observation.state` 与 `action` 都是 8 维 `float32`，顺序必须完全一致：

```text
0  left_main_joint1     rad
1  left_main_joint2     rad
2  left_main_joint3     rad
3  left_main_joint4     rad
4  left_main_joint5     rad
5  left_main_joint6     rad
6  left_main_joint7     rad
7  left_main_gripper    [0,1]，0=闭合，1=张开
```

### 4.3 双臂关节布局

当 `robot_type = "rm_base_umi_dual"` 时，`observation.state` 与 `action` 都是 16 维 `float32`。规范顺序为 right arm 在前、left arm 在后：

```text
0..6    right_main_joint1..7
7       right_main_gripper
8..14   left_main_joint1..7
15      left_main_gripper
```

虽然转换器按字段名识别关节索引，但交付时仍必须使用上述标准顺序，避免其他训练或部署组件产生歧义。`action.names` 必须与 `observation.state.names` 逐项完全相同。

### 4.4 Parquet 基础列

处理前每个 `data/**/*.parquet` 至少包含：

| 列名 | Arrow 类型 | 含义 |
| --- | --- | --- |
| `observation.state` | `fixed_size_list<float32>[8或16]` | 当前实测关节和夹爪状态 |
| `action` | `fixed_size_list<float32>[8或16]` | 当前控制周期动作/监督目标 |
| `timestamp` | `float32` | episode 内逻辑时间，单位秒 |
| `frame_index` | `int64` | episode 内帧号，从 0 连续递增 |
| `episode_index` | `int64` | episode 编号，从 0 连续递增 |
| `index` | `int64` | 全数据集全局帧号，从 0 连续递增 |
| `task_index` | `int64` | 指向 `meta/tasks.parquet` 的任务索引 |

不得出现 NaN、Inf、维度变化、缺失关节或同一列中混用 degree/radian。

### 4.5 原始腕部 RGB 相机

当前标准腕部相机 key：

- 单左臂：`observation.images.left_cam_wrist`
- 双臂：再增加 `observation.images.right_cam_wrist`

当前本项目原始腕部视频特征为：

| 项目 | 值 |
| --- | --- |
| 数组语义 | RGB，HWC，`uint8` |
| 原始分辨率 | `1920 x 1080`（宽 x 高），metadata shape 写作 `[1080, 1920, 3]` |
| 原始视频示例 | MP4 / AV1 / `yuv420p` / 30 fps / 无音频 |
| 畸变模型 | OpenCV fisheye / Kalibr `equidistant` |

原始视频编码不强制必须是 AV1，只要 ffmpeg 能稳定解码；但分辨率、相机内参和畸变参数必须相互匹配。若使用不同相机，不得套用仓库内置的 X5 标定文件。

标定 JSON 必须包含：

```json
{
  "distortion_model": "equidistant",
  "camera_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
  "distortion_coeffs": [0, 0, 0, 0],
  "resolution": [1920, 1080]
}
```

其中数值必须是该物理相机的真实标定结果，不能使用上面的占位零值。

### 4.6 原始触觉数据

每条臂默认有两路 Flux 触觉：

```text
observation.images.<side>_cam_finger0
observation.images.<side>_cam_finger1
```

原始触觉格式是 `tactile_u16_fixed_v1`：

| 项目 | 值 |
| --- | --- |
| shape | `[288, 384, 3]`，即 HWC |
| dtype | little-endian `uint16`（`<u2`） |
| 容器 | Matroska `.mkv` |
| 编码 | FFV1 lossless |
| pixel format | `gbrp16le` |
| 帧率 | 30 fps |
| 通道顺序 | `[depth, deform_x, deform_y]` |

物理场到 uint16 的编码公式为：

```text
raw_depth    = clip(round(depth * 1000), 0, 65535)
raw_deform_x = clip(round(deform_x * 1000 + 30000), 0, 65535)
raw_deform_y = clip(round(deform_y * 1000 + 30000), 0, 65535)
```

反解公式为：

```text
depth    = raw_depth / 1000
deform_x = (raw_deform_x - 30000) / 1000
deform_y = (raw_deform_y - 30000) / 1000
```

有触觉时，`meta/tactile_encoding.json` 必须准确记录 encoding、dtype、layout、channels、scale、offset、fps 和 camera keys；每个触觉 feature 也必须在 `info.json` 中包含：

```json
{
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mkv",
  "external_video": true,
  "tactile_encoding": "tactile_u16_fixed_v1",
  "storage_dtype": "uint16"
}
```

## 5. 从处理前到处理后的转换

处理顺序不可交换：

```text
原始 LeRobot v3.0 数据集
  -> (1) 腕部 RGB：原始分辨率鱼眼去畸变 + 896 中心裁剪
  -> (2) 触觉：uint16 固定范围量化到 uint8
  -> (3) 所有 MP4：缩放到 256 x 256
  -> (4) joints -> EE pose，并更新统计
```

### 5.1 腕部相机去畸变

对每路腕部相机执行：

1. 在标定分辨率（当前为 `1920 x 1080`）上构造 `cv2.fisheye.initUndistortRectifyMap`；
2. 使用原相机矩阵 `K` 作为 new camera matrix，不额外 zoom；
3. `cv2.remap` 使用线性插值，边界用常数填充；
4. 对去畸变结果做 `896 x 896` 中心裁剪；
5. 编码为 H.264 / `yuv420p` / CRF 18 / GOP 4，保持帧率和时间顺序。

对于当前 `1920 x 1080` 输入，中心裁剪左上角为：

```text
x0 = (1920 - 896) // 2 = 512
y0 = (1080 - 896) // 2 = 92
```

必须先在原始分辨率去畸变，再裁剪和缩小。先缩小再去畸变会改变几何映射、视场并降低清晰度，不符合本项目输入分布。

### 5.2 触觉 uint16 -> uint8

只转换 metadata 中明确标记为 `tactile_u16_fixed_v1` 的视频。三通道分别计算。

depth 通道：

```text
d = clip(raw_depth, 0, 1000)
u8_depth = round(255 * d / 1000)
```

deformation 两个通道：

```text
delta = clip(raw_deformation - 30000, -1000, 1000)

if delta < 0:
    u8_deformation = 128 - round(128 * abs(delta) / 1000)
else:
    u8_deformation = 128 + round(127 * delta / 1000)
```

关键映射点：

| 语义 | uint16 输入 | uint8 输出 |
| --- | ---: | ---: |
| depth 最小/中点/最大 | `0 / 500 / 1000` | `0 / 128 / 255` |
| deformation 下限/中心/上限 | `29000 / 30000 / 31000` | `0 / 128 / 255` |

超出范围的值饱和到端点。这一量化使用项目固定范围，不使用单个 episode 或整个数据集的 min/max。

量化中间产物使用 MP4 / `libx264rgb` / CRF 0 / `gbrp`；随后缩放阶段会再次编码为标准 H.264 / `yuv420p`。最终触觉数据由于 RGB-YUV 转换和 4:2:0 色度抽样而**不是逐像素可逆数据**。

### 5.3 视频缩放

所有 MP4 视频（腕部 RGB、其他 RGB、已转成 uint8 的触觉）统一缩放到 `256 x 256`：

- OpenCV `INTER_LANCZOS4`；
- 不保持原长宽比，直接缩放成正方形；
- H.264 / `yuv420p` / GOP 4；
- 普通 RGB 使用 CRF 18；
- 触觉使用 CRF 0，但由于 `yuv420p` 本身仍不是数值无损；
- 不改变逻辑 fps 和帧顺序；
- 删除音频。

腕部图像已是 `896 x 896`，因此没有长宽比变化；原始触觉 `384 x 288` 会被直接变换到 `256 x 256`，这是当前训练管线的既定行为。

### 5.4 关节到末端位姿（FK）

对 `observation.state` 和 `action` 分别做 FK，不能用 state 的 FK 结果替代 action 的 FK 结果。

单臂 7 关节输入记作 `q`（rad）。FK 得到基座系 flange 位姿：

```text
T(q) = [R(q)  p(q)]
       [ 0      1 ]
```

其中 `p=[x,y,z]`，单位为米，`R` 是 `3 x 3` 旋转矩阵。

rot6d 使用旋转矩阵前两列按列展开：

```text
rot6d(R) = concat(R[:, 0], R[:, 1])
         = [r00, r10, r20, r01, r11, r21]
```

四元数统一使用 scalar-last 顺序：

```text
[qx, qy, qz, qw]
```

每臂输出布局：

```text
rot6d: [x, y, z, rot6d_0..5, gripper]  # 10 维
quat:  [x, y, z, qx, qy, qz, qw, gripper]  # 8 维
```

双臂输出始终 right 在前、left 在后，因此 rot6d 为 20 维、quat 为 16 维。

### 5.5 episode-relative joint

设 episode 首帧 state 为 `q0`，当前 state 为 `qt`：

```text
observation.state_episode_joint[t, joint_dims] = qt - q0
observation.state_episode_joint[t, gripper_dims] = current_gripper
```

只对关节角相减，夹爪不相减。因此首帧关节值应接近全 0，首帧夹爪仍是实际绝对值。

### 5.6 episode-relative EE pose

每个 episode、每条臂都以**该 episode 首帧 observation.state 的 FK pose** 作为基准 `T0=(R0,p0)`。

对任意 state 或 action 的 FK pose `Tt=(Rt,pt)`：

```text
p_episode = R0^T * (pt - p0)
R_episode = R0^T * Rt
```

然后分别保存为 rot6d 和 quaternion。夹爪仍保存原绝对值。

注意：`action_episode_ee` 的基准也是首帧 **state** pose，不是首帧 action pose。

### 5.7 absolute EE pose

absolute 列直接保存机器人基座坐标系下的 FK 结果：

```text
p_absolute = pt
R_absolute = Rt
```

不做 episode 首帧平移或旋转消除。

### 5.8 relative action 统计

`action_relative_ee` 和 `action_relative_quat` 不作为逐帧 Parquet 列保存，只写入全局 `meta/stats.json`。它们是训练归一化统计，不是采集数据或逐帧派生数据。对每个有效的 `(t,k)`：

```text
k in [action_gap, action_gap + horizon - 1]

p_rel = R_state(t)^T * (p_action(t+k) - p_state(t))
R_rel = R_state(t)^T * R_action(t+k)
gripper_rel = gripper_action(t+k)  # 仍是绝对夹爪值
```

`action_relative_joint` 的统计同理：

```text
joint_rel = action_joint(t+k) - state_joint(t)
gripper_rel = action_gripper(t+k)
```

每个 episode 尾部没有足够未来帧的 `(t,k)` 组合自动跳过，不跨 episode 取 action。

`action_gap` 与 `horizon` 只改变上述 relative-action 统计分布，不会改变原始帧、视频、基础 Parquet 列或 9 个逐帧派生列。它们不作为数据格式验收条件。接收方在训练 relative-action 模型前，应使用该次训练自己的窗口参数重新计算这些统计；absolute-action 训练不使用它们。

### 5.9 统计内容

9 个新增逐帧数值特征都计算以下统计：

```text
min, max, mean, std, count, q01, q10, q50, q90, q99
```

- `meta/stats.json` 保存全数据集统计；
- `meta/episodes/*.parquet` 保存逐 episode 派生特征统计；
- relative action 统计仅保存在全局 `meta/stats.json`；
- 数值特征的 `std` 是总体标准差（除数为样本数 `N`，即 NumPy 默认 `ddof=0`）；`q01/q10/q50/q90/q99` 分别是 1%、10%、50%、90%、99% 分位数；
- `count` 对逐帧特征是参与统计的帧数，对 relative-action 统计是所有 episode 内有效 `(t,k)` 样本数；
- 去畸变和 resize 工具保留原数据集已有的普通图像通道统计，不会按变换后的每个像素重新扫描计算；
- 外部触觉视频在当前采集写入流程中不参与普通 RGB 图像统计，其数值定义由 `tactile_encoding.json` 给出；
- 不允许删除、复制旧统计冒充新特征统计或手工填零。

## 6. 处理后：最终数据格式

### 6.1 最终目录

```text
<dataset_id>_undist_uint8_256/
├── data/chunk-NNN/file-NNN.parquet
├── videos/observation.images.<camera_key>/chunk-NNN/file-NNN.mp4
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.parquet
    ├── tactile_encoding.json
    └── episodes/chunk-NNN/file-NNN.parquet
```

没有触觉时可以没有 `tactile_encoding.json`，目录名也不含 `_uint8`。

### 6.2 最终 Parquet 列

基础列原样保留，并新增 9 列：

| 列名 | 单臂维度 | 双臂维度 | 含义 |
| --- | ---: | ---: | --- |
| `observation.state_episode_joint` | 8 | 16 | 相对 episode 首帧的 joint state，gripper 绝对 |
| `observation.state_episode_ee` | 10 | 20 | state 的 episode-relative rot6d EE |
| `action_episode_ee` | 10 | 20 | action 的 episode-relative rot6d EE |
| `observation.state_absolute_ee` | 10 | 20 | state 的基座系 absolute rot6d EE |
| `action_absolute_ee` | 10 | 20 | action 的基座系 absolute rot6d EE |
| `observation.state_episode_quat` | 8 | 16 | state 的 episode-relative quaternion EE |
| `action_episode_quat` | 8 | 16 | action 的 episode-relative quaternion EE |
| `observation.state_absolute_quat` | 8 | 16 | state 的基座系 absolute quaternion EE |
| `action_absolute_quat` | 8 | 16 | action 的基座系 absolute quaternion EE |

所有新增列都是 Arrow `fixed_size_list<float32>[N]`。

### 6.3 最终视频特征

所有用于训练的视频 feature 必须满足：

```text
dtype: video
shape: [256, 256, 3]       # metadata 顺序为 H, W, C
container: mp4
codec: h264
pixel format: yuv420p
fps: 30
audio: false
```

处理后的触觉 feature 还必须包含：

```json
{
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "external_video": true,
  "tactile_encoding": "tactile_u8_linear_v1",
  "storage_dtype": "uint8"
}
```

`meta/tactile_encoding.json` 中应明确 `authoritative=false`、`lossless=false`、`reversible=false`。播放器显示出来的 RGB 颜色不是通道定义依据，通道语义只以 metadata 为准。

### 6.4 `meta/info.json` 关键字段

必须包含并自洽：

```json
{
  "codebase_version": "v3.0",
  "fps": 30,
  "robot_type": "rm_isf_umi_left",
  "total_episodes": 0,
  "total_frames": 0,
  "total_tasks": 0,
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "ee_num_arms": 1,
  "ee_arm_sides": ["left"],
  "undistort": {
    "model": "equidistant",
    "crop": 896,
    "cameras": {
      "observation.images.left_cam_wrist": "<实际标定文件名>.json"
    }
  }
}
```

上例中的计数必须替换成真实值。双臂数据应为：

```json
{
  "robot_type": "rm_base_umi_dual",
  "ee_num_arms": 2,
  "ee_arm_sides": ["right", "left"]
}
```

### 6.5 task 和 episode metadata

`meta/tasks.parquet` 至少包含：

```text
task_index: int64
task: string
```

任务描述应简洁、一致，建议使用训练和推理时会使用的英文命令。例如：

```text
insert the object to the hole
```

`meta/episodes/*.parquet` 必须包含：

- `episode_index`、`tasks`、`length`；
- `data/chunk_index`、`data/file_index`；
- `dataset_from_index`、`dataset_to_index`；
- 每个视频 feature 对应的 chunk/file/from_timestamp/to_timestamp；
- 原始特征和全部新增特征的逐 episode 统计。

不要根据视频文件名猜 episode 边界，也不要要求每个视频文件的总帧数等于某个 Parquet 文件的行数；文件可能聚合多个 episode。加载时必须使用 episode metadata 的区间映射。

## 7. 数据提供方操作步骤

### 7.1 准备环境

使用接收方提供的同一仓库版本，确认 `python`、PyArrow、OpenCV、SciPy、PyTorch、PyAV、ffmpeg/ffprobe 和仓库内 RealMan SDK 可用。

记录 commit：

```bash
git rev-parse HEAD
```

### 7.2 放置并初检原始数据

```bash
test -f playground/data/<dataset_id>/meta/info.json
python -m json.tool playground/data/<dataset_id>/meta/info.json >/dev/null
```

先人工确认：

- `robot_type` 正确；
- `fps=30`；
- state/action 名称、顺序、单位一致；
- 相机 key、物理相机和标定文件一一对应；
- 触觉 metadata 与真实文件一致；
- episode/task/index 连续且没有空 episode。

### 7.3 使用正确标定运行处理

使用仓库默认 X5 相机和默认标定时：

```bash
bash scripts/process_joint_data.sh <dataset_id> 256
```

使用不同腕部相机或重新标定结果时，显式指定 camera keys 和标定文件。例如单左臂：

```bash
CAMERAS="observation.images.left_cam_wrist" \
CALIB="/absolute/path/to/left_intrinsics.json" \
CROP=896 \
bash scripts/process_joint_data.sh <dataset_id> 256
```

双臂且两路标定不同时，`CALIB` 参数的具体传法应按工具支持的 `camera=path` 形式配置并先做测试帧检查。不得在未查看去畸变结果的情况下批量处理。

### 7.4 处理后的最低人工检查

至少随机检查：

- 每个 camera key 的 3 个 episode（开头、中间、结尾）；
- 腕部图像是否方向正确、没有错误裁剪、严重黑边或错误标定；
- finger0/finger1 是否接反，depth/deform 通道是否符合 metadata；
- 关节运动方向与视频动作一致；
- 每个 episode 首帧的 episode-relative xyz 接近 0、旋转接近单位旋转；
- absolute EE 位置是否处于合理工作空间；
- quaternion 范数是否接近 1；
- 没有 NaN、Inf、空视频或零帧文件。

## 8. 验收与拒收条件

### 8.1 自动验收要点

接收方应检查：

1. `info.json` 声明值与真实 Parquet/video 一致；
2. `total_frames == 所有 data parquet 行数之和`；
3. `total_episodes == episode metadata 行数`；
4. episode/index/frame_index 连续且边界正确；
5. `timestamp == frame_index / 30`（允许 float32 误差）；
6. 每个 `task_index` 都能在 `tasks.parquet` 找到；
7. state/action names 完全一致，shape 与机器人构型一致；
8. 9 个派生 Parquet 列全部存在，shape/dtype 正确；
9. `stats.json` 包含所有派生列以及 `action_relative_joint/ee/quat`；
10. 所有 MP4 可解码，真实分辨率、fps、codec、pix_fmt 与 metadata 一致；
11. 每个 episode 的视频 metadata 时间范围有效并可取到对应帧；
12. `ee_num_arms`、`ee_arm_sides`、`robot_type` 相互一致；
13. 触觉 manifest、feature metadata 和实际编码一致；
14. 随机重算 FK 后与交付的 EE 列在 float32 容差内一致。

可用以下命令抽查真实视频：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 \
  <video.mp4>
```

预期关键输出：

```text
codec_name=h264
width=256
height=256
pix_fmt=yuv420p
r_frame_rate=30/1
```

### 8.2 直接拒收情形

出现以下任一情况应退回数据提供方修正：

- 只提供散装视频/CSV，没有完整 LeRobot metadata；
- 不是 30 Hz，或只是修改 FPS 标签而没有正确重采样/同步；
- `robot_type` 与真实机器人不符；
- 关节单位、顺序或夹爪方向不明；
- action 的时序含义不明；
- 腕部相机使用了错误内参或先缩小后去畸变；
- 触觉被自动拉伸 min/max、通道交换或使用有损格式保存原始 uint16；
- 最终视频不是 `256 x 256` H.264/yuv420p/30 fps；
- 缺少 EE 派生列或统计；
- 存在 NaN、Inf、损坏视频、空 episode、大量未说明的重复帧；
- 通过重命名字段或修改 metadata 掩盖不兼容硬件。

## 9. 交付清单模板

数据提供方交付前逐项确认：

```text
[ ] 完整 LeRobot v3.0 最终数据集目录
[ ] meta/info.json
[ ] meta/stats.json
[ ] meta/tasks.parquet
[ ] meta/episodes/**/*.parquet
[ ] meta/tactile_encoding.json（有触觉时）
[ ] data/**/*.parquet
[ ] videos/**/*.mp4
[ ] DELIVERY.md
[ ] 实际使用的相机标定 JSON
[ ] SHA256SUMS
[ ] 可选但强烈建议：完整原始 uint16/原始鱼眼数据集
```

`DELIVERY.md` 推荐模板：

```markdown
# Dataset delivery

- Dataset ID:
- Collection date:
- Git commit:
- Robot type:
- Arm sides:
- Task(s):
- Episodes:
- Total frames:
- Logical fps: 30
- Joint unit: rad
- Gripper convention: 0 closed, 1 open
- Processing command: bash scripts/process_joint_data.sh <id> 256
- CROP: 896
- CAMERAS:
- CALIB:
- Wrist camera serial numbers:
- Tactile sensor IDs and finger mapping:
- Known dropped/repeated frames:
- Failed/removed episodes:
- Other anomalies:
```

生成校验和：

```bash
find <final_dataset_dir> -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS
```

## 10. 接口边界总结

数据提供方的责任不是“提供看起来相似的视频和关节数组”，而是交付一个在以下五个层面都一致的数据集：

1. **物理定义一致**：机器人构型、关节单位、基座/flange 坐标系、夹爪方向一致；
2. **时间定义一致**：所有模态在 30 Hz 逻辑时间轴同步，state/action 语义一致；
3. **视觉定义一致**：正确标定去畸变、896 中心裁剪、256 正方形输出；
4. **触觉定义一致**：固定 `uint16` 编码和固定范围 `uint8` 量化，通道顺序明确；
5. **存储定义一致**：完整 LeRobot v3.0 目录、Parquet schema、视频编码、metadata 和统计全部自洽。

只要其中任一层不一致，即使文件可以被 Python 打开，也不能认为数据已经达到 `process_joint_data.sh` 的最终格式。
