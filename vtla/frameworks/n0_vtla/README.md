# N0-VTLA in StarVTLA

独立的 `n0_vtla` policy。所有模型代码在 StarVTLA 内；训练、推理、保存和恢复不读取
`ref_repo`，不导入外部 `n0vtla` 包，也不修改已安装的 Transformers。PaliGemma/Gemma
复用本项目的 `pi05` / `pi_gemma`，触觉 encoder / predictor 的归属与许可见 NOTICE / LICENSE。

末端 state/action 统一使用 TCP rot6d：绝对位姿在基座系，`relative_rot6d` 为当前 TCP 系位移与
零中心相对旋转，反归一化后旋转全零表示不旋转，夹爪为绝对指令。整个 chunk 共用当前 TCP
锚点，`state_mode=none` 也需保留该隐藏锚点。关节模式不变，不再提供 `ee_frame` 或 quaternion
模型模式。旧 EE 数据需迁移、旧 EE checkpoint 需重训；定义和工具见
[TCP 数据与动作约定](../../../tools/TCP_ACTIONS.md)。

使用 `relative_rot6d` 时，数据统计必须匹配实际 `chunk_size` 和 `action_gap`。
本模型默认 chunk 为 50；gap=6 时，已迁移数据的统计重建命令为：

```bash
python tools/rebuild_relative_ee_stats.py \
  --root playground/data/<dataset_id> --horizon 50 --action-gap 6
```

这只重建统计，不迁移旧位姿；旧数据先用 [迁移工具](../../../tools/README.md#tcp-数据迁移与统计量重建)。

## 模型

RGB 和 PI0.5 风格的任务/状态 prompt → PaliGemma；episode 初始基线与当前触觉的差分
→ 冻结 DINOv2 → cross-attention predictor → 默认 5 个 latent → Gemma 动作专家。
训练采用动作 flow matching；推理默认 10 步 Euler 采样，单次请求只计算一次触觉 latent
和 prefix KV。没有跨动作块的 Transformer 历史或执行中的高频触觉闭环。

默认 predictor 为原版公开后训练配置的 `tactile_kv`，保留 `joint_kv`、零初始化 z gate
和可选 `g_to_expert` 参数，导入时必须与权重匹配。初始 gate 为零并不保证插入 token 后
与无触觉模型逐位一致；它只控制注入 token 的数值。

## 配置

- 仅支持 `tactile_mode=as_image`；`none` / `encode` 均报错。触觉不会进入 RGB 视觉塔。
- 复用 `wrist_only`、相机 keys、`tactile_keys`、`ee_num_arms` 以及现有 state/action modes。
- 动作：`absolute_joint`、`relative_joint`、`absolute_rot6d`、`relative_rot6d`。有效宽度由 dataset feature schema 决定。
- 状态：`none`、absolute/episode joint、absolute/episode TCP rot6d。无状态模型不生成
  state prompt，但相对动作仍需要隐藏的当前动作锚点，由公共 processor 管理。
- `chunk_size=50`、`n_action_steps=50`、`action_start_offset=0`。短执行窗口应与后训练匹配。
- 支持 `action_gap`；训练 sampler 排除 episode 最后 `action_gap` 帧，避免全 padding 的目标块。
- `max_action_dim=32`、`max_state_dim=32`。模型输入 padding 不参与动作 loss；episode 尾部
  `action_is_pad` 也会被屏蔽。扩大动作容器必须显式适配投影层。
- `tactile_num_frames=1`、`tactile_frame_offset=1` 固定，不能用于改变基线采样。
- `tactile_image_size=224`，须为 DINOv2 patch size 的倍数。输入是 uint8 RGB 或 [0,1]
  浮点图像；物理力场需先通过一致的、明确的数据表示转换为图像。
- 首期不支持 RTC、torch.compile、streaming dataset。支持标准 indexed dataset 及同 schema mixture。

## 动作语义与权重

**本适配器使用 StarVTLA 的动作语义。** `relative_rot6d` 是当前 TCP 系下的位置差和减去单位 rot6d 的相对旋转；原版基础模型的位置/rot6d 逐元素增量不是相同表示。导入原生参数后应使用当前
数据集的正确统计后训练，不能直接把基础权重当成任意机器人可部署策略。
相对关节动作需要 dataset 的 action feature names 正确标记 gripper，以便保留绝对夹爪指令。

两个路径用途不同：

- `base_model_path`：本地原生 N0-VTLA `model.safetensors` 或含此文件的目录。初始化时严格
  检查键名/形状，兼容旧 `tactile_prior` 命名及 PaliGemma 的模块层级迁移。
- `pretrained_path`：StarVTLA 保存的完整 `n0_vtla` policy checkpoint，用于恢复/部署。
  加载时不再次访问 `base_model_path`，DINOv2 由配置直接构建、再加载 checkpoint 内的参数，
  不会自动下载 DINOv2。当前实现不提供原版服务端的增量解码复现模式。

默认拒绝无权重初始化，只有显式 `allow_random_init=true` 才允许随机模型（用于测试/研究）。
改变 `max_action_dim` 导致原版投影不匹配时，可显式设置 `reinitialize_action_projections=true`；
这会重新初始化动作输入/输出层，其他参数仍严格匹配。调整动作语义即使没有形状变化也需要后训练。

Tokenizer 使用 `paligemma_tokenizer_path`；未指定时使用标准 PaliGemma tokenizer 名称。
离线运行应指向本地 tokenizer（例如现有 pi05_base 的 tokenizer 子目录）。模型恢复不依赖
原生源码或原生权重路径，但部署时仍须提供 processor 配置中记录的 tokenizer 资产。

## 训练

```bash
N0_VTLA_BASE_PATH=/path/to/n0-vtla-base \
PALIGEMMA_TOKENIZER_PATH=playground/pretrained_models/pi05_base/paligemma-3b-pt-224-tokenizer \
  bash train.sh <dataset_id> n0_vtla 4 1 20000 true as_image absolute_joint relative_rot6d
```

相机、触觉 keys 和单/双臂设置按现有 StarVTLA 数据约定配置。更细的模型参数通过
`python -m vtla.train --policy.type=n0_vtla ...` 设置，例如 `--policy.predictor_arch=joint_kv`。
`PRETRAINED_PATH` 用于加载已保存的 StarVTLA checkpoint，不能指向没有 StarVTLA config 的原版权重。

训练数据在每次随机采样时直接索引当前 episode 的 `dataset_from_index`，输出每路触觉
`[baseline, current]`，形状为 `[B,2,3,H,W]`。不依赖有限负偏移或上一批训练样本，支持
episode 过滤和乱序。默认 augmentation 仅作用于 RGB，触觉基线和当前帧使用同一确定性缩放。
初始帧是否无接触由采集约定决定。

## 推理

使用现有 StarVTLA 部署入口加载保存的 checkpoint；输入经公共 preprocessor，输出经公共
postprocessor 还原绝对控制目标。直接调用模型时的形状：

- RGB：`[B,3,H,W]`；触觉：`[B,3,H,W]`。
- 可显式提供 `<tactile_key>.baseline`，或提供 `[B,2,3,H,W]` 的 baseline/current 对。
- 未显式提供基线时，每个 episode 首次预测捕获当前触觉，后续复用。
- 缺失传感器用同 shape 的占位图及 `<key>_mask=False`；不可静默省略配置的 key。
- `predict_action_chunk()` 返回 `[B,chunk_size,active_action_dim]` 的模型空间动作。
- `select_action()` 按配置执行窗口消费队列；每个 episode 必须 reset policy 和 processors。
- 直接调用 processor/模型时需使用与现有部署循环相同的动作锚点锁定逻辑；公共部署入口已处理。

## 验证

```bash
python -m pytest tests/frameworks/test_n0_vtla.py -q
```

测试覆盖真实小型 Gemma/SigLIP/DINOv2 前向和反向、触觉条件依赖、单步训练/缓存采样一致性、
模型和 processor 保存恢复、动作表示往返、精确 episode 基线及过滤索引、padding loss、
基线 reset、动作队列和严格权重加载。小模型测试不代表已验证原版大模型的任务成功率。

本地原生权重的完整 GPU 数值测试（需要本地 tokenizer，不访问网络）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.test_n0_vtla_weights \
  --weights playground/pretrained_models/n0-vtla-base
```

报告默认写入 `playground/results/n0_vtla_weights_smoke.json`。使用合成输入验证严格加载、
3 路 RGB / 4 路触觉、20 维有效动作、50 步 chunk、10 步采样、反向梯度、固定噪声
可重复性及触觉干预响应。不执行优化器更新，不评估真实任务成功率或与原版策略的数值等价性。
DINOv2 初始化保留原权重的 518 分辨率位置网格，实际 224 触觉输入通过内部位置编码插值处理。
