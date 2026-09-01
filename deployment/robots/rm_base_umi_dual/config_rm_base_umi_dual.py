#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
睿尔曼 RM75b 双臂 (ugripper 集成版) 配置类

每条手臂的设备:
    - 从臂 (Follower): TCP/IP 网线  (左 192.168.1.201 / 右 192.168.1.200, 端口 8080)
    - 领控电爪 (Gripper): gRPC/CAN  (左 192.168.1.11:55551 / 右 192.168.1.10:55551)
    - 手腕鱼眼相机: fish_camera gRPC(50088) + UDP, 原生 1920x1080
    - 触觉传感器 x2: dmrobotics Flux gRPC(50051/50052), 输出 (288,384,3) uint16

主臂 (Leader) 不在本机器人内, 由 rm_leader_dual 遥操作器负责。

observation / action 字段统一加 left_ / right_ 前缀:
    observation.state : left_main_joint1..7, left_main_gripper, right_...
    observation.images: left_cam_wrist, left_cam_finger0/1, right_...
    action            : left_main_joint1..7, left_main_gripper, right_...  (夹爪归一化 [0,1])
"""

from dataclasses import dataclass, field
from typing import ClassVar

from deployment.hardware.top_cameras import OpenCVTopCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("rm_base_umi_dual")
@dataclass
class RmBaseUmiDualConfig(RobotConfig):
    """睿尔曼 RM75b 双臂 (ugripper) 配置。"""

    kinematics_force_type: ClassVar[str] = "base"
    kinematics_sides: ClassVar[tuple[str, ...]] = ("right", "left")
    teleop_type: ClassVar[str] = "rm_leader_dual"

    # ============ 启用的手臂 ============
    # 可选 ["left"], ["right"], 或 ["left", "right"]
    arms: list[str] = field(default_factory=lambda: ["left", "right"])

    # ============ 硬件存在性开关 (物理 rig 有什么就开什么) ============
    # 触觉传感器: True=连接并产出 *_cam_finger0/1 观测; False=完全跳过 (原 _notac 变体)。
    # 顶部相机: 由下方 cameras 配置控制 (空字典 = 无 top, 原 _notop 变体)。
    # 关节 state 永远产出, 不设开关 (控制/安全都需要; 模型是否消费由 checkpoint 决定)。
    use_tactile: bool = True

    # ============ 从臂 (Follower - TCP/IP) ============
    left_follower_ip: str = "192.168.1.201"
    right_follower_ip: str = "192.168.1.200"
    follower_tcp_port: int = 8080

    # ============ 每臂板子 IP (鱼眼/触觉/夹爪代理) ============
    left_board_ip: str = "192.168.1.11"
    right_board_ip: str = "192.168.1.10"

    # ============ 领控电爪 (gRPC/CAN via 板子) ============
    gripper_grpc_port: int = 55551
    gripper_can_interface: str = "can0"
    gripper_can_bitrate: int = 1_000_000
    gripper_speed: int = 40    # 10~100
    gripper_torque: int = 50   # 10~100
    # 真实满行程覆盖 (编码器计数): 解决左右爪传动比不同。
    # SDK grip_init 把 max_itinerary 写死成 25000/90000, 但实测左右爪不同
    # (measure_gripper_stroke.py: 左≈92218, 右≈113972)。None 表示用 SDK 自动值。
    left_gripper_itinerary: int | None = 92218
    right_gripper_itinerary: int | None = 113972

    # ============ 手腕鱼眼相机 (fish_camera gRPC + UDP) ============
    fisheye_grpc_port: int = 50088
    # 每臂占用一个本机 UDP 接收端口 (必须各不相同)
    left_fisheye_udp_port: int = 50100
    right_fisheye_udp_port: int = 50101
    fisheye_width: int = 1920
    fisheye_height: int = 1080
    fisheye_max_datagram: int = 1200

    # ============ 腕部鱼眼去畸变 (消除训练-推理 gap) ============
    # 训练数据若用 tools/undistort_dataset_videos.py 去过畸变(腕部=去畸变+居中裁 896),
    # 推理时必须对原生鱼眼帧做相同变换, 否则几何不一致。
    #   "auto"  : 由 inference.py --match_policy 按 checkpoint 自动判定后改写为 true/false;
    #             机器人层(采集/遥操作, 无 policy)把 auto 视为关闭(采集存原生鱼眼)。
    #   "true"  : 强制开启;  "false" : 强制关闭
    undistort_wrist: str = "auto"
    # 居中裁剪边长 (须与训练 crop 一致, tools 默认 896)
    undistort_crop: int = 896
    # 每臂标定文件 {side: path}; None = 用 deployment 内置 calib/x5_{left,right}_intrinsics.json
    wrist_calib: dict[str, str] | None = None

    # ============ 触觉传感器 (dmrobotics Flux gRPC) ============
    pc_host: str = "192.168.1.102"          # 本机 IP, 用于 UDP 回传
    tactile0_grpc_port: int = 50051
    tactile1_grpc_port: int = 50052
    tactile0_dev_id: int = 0
    tactile1_dev_id: int = 2
    # 每路触觉占用一个本机 UDP 接收端口 (必须各不相同)
    left_tactile0_pc_port: int = 60000
    left_tactile1_pc_port: int = 60001
    right_tactile0_pc_port: int = 60002
    right_tactile1_pc_port: int = 60003
    # 输出尺寸 (须与 dmrobotics getDepth/getDeformation2D 实际输出一致)
    # 实测这批 Flux 传感器输出 384x288 (见 record 日志首帧 shape=(288,384,3))
    tactile_width: int = 384
    tactile_height: int = 288
    # 独立控制触觉源端推送率；不要与鱼眼 stream_max_fps 共用。
    tactile_max_fps: int = 30
    # 输出遵循 tactile_u16_fixed_v1: [depth*1000, dx*1000+30000, dy*1000+30000]。
    # 权威帧由采集引擎以 FFV1/gbrp16le 无损 MKV video feature 保存。

    # ============ 数据流首帧等待超时 ============
    stream_first_frame_timeout: float = 5.0

    # ============ 数据流限速 ============
    # ⚠️ 跳帧式限速的"量化陷阱": 从 ~50fps 的鱼眼源按 N 限速, 会被量化成源帧率的整数
    # 分之一。设 30~49 之间任意值都会变成"每隔一帧"= ~25fps (< 30Hz 消费) -> 手腕重复帧
    # 卡顿。实测 (analyze 鱼眼视频): 设 45 时唯一帧只有 25fps。
    # 因此鱼眼只能取: 0 = 不限速(约 50-59fps), 或 >= 源帧率(>=60)。
    # 保证产出 > 录制 30fps, 主循环每帧都有新帧。28 核机器满速 CPU 仅 ~24%, 充裕。
    # 若想省 CPU 又不卡, 用 640x360/480 鱼眼(满速也很轻), 别回到 30~49 的限速。
    stream_max_fps: float = 0.0

    # ============ 数据流诊断 ============
    # True 时每个 worker 每 ~2s 打印产出fps/解码耗时/UDP到达率, 用于排查瓶颈
    # (板子+网线 vs PC解码)。正常录制请保持 False。
    stream_debug_fps: bool = False

    # ============ 安全与控制 ============
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    use_degrees: bool = False  # False = 弧度制

    # ============ 动作空间 (joint | ee) ============
    # "joint": action = 16 维关节角 (默认, 采集/遥操作不受影响)。
    # "ee"   : action = 20 维末端位姿 (右臂在前, 每臂 [xyz, rot6d, gripper]); send_action 走
    #          rm_movep_canfd 笛卡尔在线控制。由 inference.py --match_policy 按 checkpoint 的
    #          action_mode (relative_ee -> ee) 自动设定, 一般无需手填。
    action_space: str = "joint"
    # rm_movep_canfd 透传参数 (ee 模式下生效)
    canfd_follow: bool = False        # False=非跟随(轨迹模式), True=透传跟随
    canfd_trajectory_mode: int = 0    # 0=轨迹模式
    canfd_radio: int = 0              # 平滑系数 0~100
    # ee 模式笛卡尔单步安全限幅 (当前->目标 这一步, None=不限)。movep 绕过关节 max_relative_target,
    # 故这里是 ee 模式下的主要安全网, 配合控制器自身限幅。
    max_ee_pos_step_m: float | None = 0.05
    max_ee_rot_step_deg: float | None = 15.0
    # 上电自检: ee 模式下检查工具/工作坐标系是否≈单位 (movep 位姿系须= rm_algo_forward_kinematics
    # 的 flange/base 系, 否则有系统性偏移)。True=不符仅告警; 设 False 跳过自检。
    ee_frame_check: bool = True

    # ============ 自动复位 (move_to_home) ============
    # 连接时自动读取当前位置；启动前先把机械臂摆到安全的初始姿态。
    # 复位时夹爪目标开度 (归一化 [0,1], 1.0=张开, 0.0=夹紧)
    home_gripper: float = 1.0
    # 普通 movej 的目标复位耗时 (秒)，用于估算控制器速度百分比。
    home_duration_s: float = 2.0
    # movej 返回后以关节反馈确认到位；容差统一用角度表示。
    home_joint_tolerance_deg: float = 1.0
    home_settle_timeout_s: float = 2.0

    # ============ 额外本地 USB 相机 ============
    # 顶部全景相机 (cam_top), 参考 realman_tactile_shandd_hd_tac16
    cameras: dict[str, OpenCVTopCameraConfig] = field(
        default_factory=lambda: {
            "cam_top": OpenCVTopCameraConfig(
                index_or_path=6,
                fps=30,
                width=1920,
                height=1080,
            ),
        }
    )
    # 需要做"中心裁剪到 4:3"的相机名 (居中裁出 宽:高 = 4:3 的区域)。
    # cam_top 1920x1080 -> 1440x1080。
    crop_4_3_cameras: list[str] = field(default_factory=lambda: ["cam_top"])

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.arms != ["left", "right"]:
            raise ValueError("rm_base_umi_dual requires arms=['left', 'right']")
