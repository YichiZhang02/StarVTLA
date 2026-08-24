#!/usr/bin/env python

"""睿尔曼 RM75b 单右臂 ugripper 配置。"""

from dataclasses import dataclass, field
from typing import ClassVar

from deployment.hardware.top_cameras import OpenCVTopCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("rm_isf_umi_right")
@dataclass
class RmIsfUmiRightConfig(RobotConfig):
    """机械臂、夹爪、腕部相机和两路触觉组成的独立单右臂配置。"""

    kinematics_force_type: ClassVar[str] = "isf"
    kinematics_sides: ClassVar[tuple[str, ...]] = ("right",)
    teleop_type: ClassVar[str] = "rm_leader_right"

    use_tactile: bool = True

    # 机械臂本体。
    follower_ip: str = "192.168.1.200"
    follower_tcp_port: int = 8080

    # 末端板同时代理夹爪、腕部相机和触觉服务。
    board_ip: str = "192.168.1.11"

    # 领控电爪。
    gripper_grpc_port: int = 55551
    gripper_can_interface: str = "can0"
    gripper_can_bitrate: int = 1_000_000
    gripper_speed: int = 40
    gripper_torque: int = 50
    gripper_itinerary: int | None = None

    # 腕部鱼眼相机。
    fisheye_grpc_port: int = 50088
    fisheye_udp_port: int = 50101
    fisheye_width: int = 1920
    fisheye_height: int = 1080
    fisheye_max_datagram: int = 1200

    # 腕部鱼眼去畸变。None 使用内置 x5_right 标定。
    undistort_wrist: str = "auto"
    undistort_crop: int = 896
    wrist_calib: str | None = None

    # 触觉传感器。
    pc_host: str = "192.168.1.102"
    tactile0_grpc_port: int = 50051
    tactile1_grpc_port: int = 50052
    tactile0_dev_id: int = 0
    tactile1_dev_id: int = 2
    tactile0_pc_port: int = 60002
    tactile1_pc_port: int = 60003
    tactile_width: int = 384
    tactile_height: int = 288
    tactile_max_fps: int = 30

    stream_first_frame_timeout: float = 5.0
    stream_max_fps: float = 0.0
    stream_debug_fps: bool = False

    # 安全与控制。
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    use_degrees: bool = False

    # 动作空间与 RealMan CAN-FD 透传参数。
    action_space: str = "joint"
    canfd_follow: bool = False
    canfd_trajectory_mode: int = 0
    canfd_radio: int = 0
    max_ee_pos_step_m: float | None = 0.05
    max_ee_rot_step_deg: float | None = 15.0
    ee_frame_check: bool = True

    # 自动复位。
    home_joints: dict[str, float] | None = None
    home_gripper: float = 1.0
    home_duration_s: float = 4.0

    # 额外本地 USB 相机；新 rig 默认没有顶部相机。
    cameras: dict[str, OpenCVTopCameraConfig] = field(default_factory=dict)
    crop_4_3_cameras: list[str] = field(default_factory=list)
