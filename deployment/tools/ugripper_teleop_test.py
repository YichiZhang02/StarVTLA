#!/usr/bin/env python
"""
睿尔曼 RM75b 双臂遥操作测试脚本

支持单臂和双臂遥操作（带夹爪）：
- 主臂 (Leader) → 从臂 (Follower)
- 主臂夹爪 → 从臂夹爪 (领控夹爪通过 gRPC/CAN)

配置:
- 左臂从臂 IP: 192.168.1.200, 夹爪 gRPC: 192.168.1.10:55551
- 右臂从臂 IP: 192.168.1.201, 夹爪 gRPC: 192.168.1.11:55551
- 左主臂串口: /dev/ttyRealmanBaseLeaderL
- 右主臂串口: /dev/ttyRealmanBaseLeaderR

使用方法:
    python ugripper_遥操测试.py          # 交互式选择
    python ugripper_遥操测试.py -m left  # 仅左臂
    python ugripper_遥操测试.py -m right # 仅右臂
    python ugripper_遥操测试.py -m dual  # 双臂

按 Ctrl+C 停止遥操作
"""

import logging
import time
import sys
import os
import argparse
import numpy as np
import signal

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加路径 (本文件位于 deployment/tools/, 上一级即 deployment/)
_DEPLOY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_DEPLOY_ROOT, 'sdk'))                              # Robotic_Arm 模块
sys.path.insert(0, os.path.join(_DEPLOY_ROOT, 'sdk', 'dm_lingkong_grip'))  # dm_lingkong_grip_sdk 包
sys.path.insert(0, os.path.dirname(_DEPLOY_ROOT))                                  # tac_infra 根 (供 import deployment.*)

# 全局停止标志
stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    print("\n\n正在停止遥操作...")
    stop_flag = True

signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# 配置参数
# ============================================================
# 左臂配置
LEFT_FOLLOWER_IP = "192.168.1.200"
LEFT_FOLLOWER_PORT = 8080
LEFT_LEADER_PORT = "/dev/ttyRealmanBaseLeaderL"
LEFT_LEADER_PORT_ALT = "/dev/ttyUSB0"  # 备用端口（没有udev规则时）
LEFT_GRIPPER_SERVER = "192.168.1.10:55551"

# 右臂配置
RIGHT_FOLLOWER_IP = "192.168.1.201"
RIGHT_FOLLOWER_PORT = 8080
RIGHT_LEADER_PORT = "/dev/ttyRealmanBaseLeaderR"
RIGHT_LEADER_PORT_ALT = "/dev/ttyUSB1"  # 备用端口
RIGHT_GRIPPER_SERVER = "192.168.1.11:55551"

# CAN 配置
CAN_INTERFACE = "can0"
CAN_BITRATE = 1000000

# 主臂夹爪范围 (实测): MIN=夹紧, MAX=张开
LEADER_GRIPPER_MIN = 0.066
LEADER_GRIPPER_MAX = 0.971
GRIPPER_GAIN = 1.0

# 从臂电爪参数 (新 SDK)
GRIPPER_SPEED = 40    # 10~100
GRIPPER_TORQUE = 50   # 10~100, 越小越不容易夹坏物体

# 行程挡位强制覆盖 (解决左右爪自动判挡不一致 / 传动比问题)
# grip_init 会按实测行程是否 >40000 自动把 max_itinerary 锁成 25000 或 90000。
# 若某侧判错, 在此按"正常侧"实测到的值强制覆盖, 例如:
#   "right": {"speed_coe": 1000, "max_itinerary": 25000}
# 留空 {} 表示用 SDK 自动判挡。先跑一次看两侧打印值再决定填什么。
GRIPPER_FORCE_ITINERARY: dict = {
    # 实测值 (measure_gripper_stroke.py): 右爪真实行程比 SDK 写死的 90000 大 ~27%
    "left":  {"speed_coe": 3600, "max_itinerary": 92218},
    "right": {"speed_coe": 3600, "max_itinerary": 113972},
}


class ArmController:
    """单臂控制器，包含主臂、从臂和夹爪"""
    
    def __init__(self, side: str):
        """
        Args:
            side: "left" 或 "right"
        """
        self.side = side
        self.name = "左臂" if side == "left" else "右臂"
        
        # 设备
        self.follower = None
        self.follower_handle = None
        self.leader = None
        self.gripper = None
        
        # 配置
        if side == "left":
            self.follower_ip = LEFT_FOLLOWER_IP
            self.follower_port = LEFT_FOLLOWER_PORT
            self.leader_port = LEFT_LEADER_PORT
            self.leader_port_alt = LEFT_LEADER_PORT_ALT
            self.gripper_server = LEFT_GRIPPER_SERVER
        else:
            self.follower_ip = RIGHT_FOLLOWER_IP
            self.follower_port = RIGHT_FOLLOWER_PORT
            self.leader_port = RIGHT_LEADER_PORT
            self.leader_port_alt = RIGHT_LEADER_PORT_ALT
            self.gripper_server = RIGHT_GRIPPER_SERVER
        
        # 状态
        self.is_connected = False
        self.gripper_initialized = False
    
    def connect_follower(self) -> bool:
        """连接从臂"""
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
            
            print(f"   连接{self.name}从臂 ({self.follower_ip}:{self.follower_port})...")
            self.follower = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            self.follower_handle = self.follower.rm_create_robot_arm(
                self.follower_ip, self.follower_port
            )
            
            if self.follower_handle.id < 0:
                print(f"   ❌ {self.name}从臂连接失败!")
                return False
            
            print(f"   ✅ {self.name}从臂连接成功")
            return True
        except Exception as e:
            print(f"   ❌ {self.name}从臂连接异常: {e}")
            return False
    
    def connect_leader(self) -> bool:
        """连接主臂"""
        try:
            from deployment.hardware.leader_arms import RealmanLeader as LeaderArm
            
            # 尝试主端口，如果不存在则尝试备用端口
            port = None
            if os.path.exists(self.leader_port):
                port = self.leader_port
            elif os.path.exists(self.leader_port_alt):
                port = self.leader_port_alt
            else:
                print(f"   ❌ {self.name}主臂 ({self.leader_port}) 未连接!")
                return False
            
            print(f"   连接{self.name}主臂 ({port})...")
            self.leader = LeaderArm(
                port=port,
                baudrate=460800,
                hex_data="55 AA 02 00 00 67",
            )
            self.leader.connect()
            print(f"   ✅ {self.name}主臂连接成功")
            return True
        except Exception as e:
            print(f"   ❌ {self.name}主臂连接异常: {e}")
            return False
    
    def connect_gripper(self) -> bool:
        """连接领控夹爪 (gRPC/CAN)"""
        try:
            from dm_lingkong_grip_sdk import LingkongGrip
            
            print(f"   连接{self.name}夹爪 ({self.gripper_server})...")
            self.gripper = LingkongGrip(
                server_address=self.gripper_server,
                interface=CAN_INTERFACE,
                bitrate=CAN_BITRATE
            )
            
            if not self.gripper.init_status:
                print(f"   ❌ {self.name}夹爪 gRPC 连接失败!")
                return False
            
            print(f"   ✅ {self.name}夹爪 gRPC 连接成功")
            return True
        except Exception as e:
            print(f"   ❌ {self.name}夹爪连接异常: {e}")
            return False
    
    def _clear_and_enable(self) -> None:
        """清错误 + 使能闭环 —— 新 SDK grip_init 前的恢复步骤 (见 test_grip_new_api.py)。

        否则若电机处于失能/错误态, grip_init 会卡在 "Read open position" 失败。
        """
        self.gripper.client.recv_can_async(self.gripper._on_message_received, 1000)
        time.sleep(0.3)
        self.gripper.client.send_can(0x141, [0x9B, 0, 0, 0, 0, 0, 0, 0])  # 清错误标志
        time.sleep(0.2)
        self.gripper.client.send_can(0x141, [0x88, 0, 0, 0, 0, 0, 0, 0])  # 电机使能
        time.sleep(0.2)

    def init_gripper(self) -> bool:
        """初始化夹爪（会自动夹紧）"""
        if self.gripper is None:
            return False

        try:
            print(f"   初始化{self.name}夹爪...")
            self._clear_and_enable()
            if self.gripper.grip_init(time_out=6000):
                # 设温和的力矩/速度, 避免遥操作夹坏物体 (grip_init 内部默认 torque=90)
                self.gripper.set_torque_limit(GRIPPER_TORQUE)
                self.gripper.set_speed(GRIPPER_SPEED)
                # 诊断: 打印行程自动判挡结果, 用于对比左右爪 (传动比问题排查)
                print(
                    f"   ✅ {self.name}夹爪初始化成功  "
                    f"clamp_pos={self.gripper._clamp_pos} "
                    f"open_pos={self.gripper._open_pos} "
                    f"max_itinerary={self.gripper._max_itinerary} "
                    f"speed_coe={self.gripper._speed_coe}"
                )
                # 可选: 强制覆盖行程挡位 (当自动判挡对右爪判错时)
                force = GRIPPER_FORCE_ITINERARY.get(self.side)
                if force is not None:
                    self.gripper._speed_coe = force["speed_coe"]
                    self.gripper._max_itinerary = force["max_itinerary"]
                    self.gripper._open_pos = self.gripper._clamp_pos - force["max_itinerary"]
                    print(
                        f"   ⚙️  {self.name}已强制覆盖行程: "
                        f"max_itinerary={self.gripper._max_itinerary} "
                        f"speed_coe={self.gripper._speed_coe} "
                        f"open_pos={self.gripper._open_pos}"
                    )
                self.gripper_initialized = True
                return True
            else:
                print(f"   ❌ {self.name}夹爪初始化失败")
                return False
        except Exception as e:
            print(f"   ❌ {self.name}夹爪初始化异常: {e}")
            return False
    
    def connect_all(self) -> bool:
        """连接所有设备（从臂、主臂、夹爪）"""
        # 从臂必须连接
        if not self.connect_follower():
            return False
        
        # 主臂必须连接
        if not self.connect_leader():
            return False
        
        # 夹爪必须连接并初始化
        if not self.connect_gripper():
            return False
        
        if not self.init_gripper():
            return False
        
        self.is_connected = True
        return True
    
    def read_leader_position(self):
        """读取主臂位置，返回 (关节弧度列表, 夹爪位置)"""
        if self.leader is None:
            return None, None
        
        pos = self.leader.read_position()
        joints_rad = pos[:7]
        gripper = pos[7]
        return joints_rad, gripper
    
    def send_follower_position(self, joints_rad):
        """发送从臂位置（透传模式）"""
        if self.follower is None:
            return
        
        joints_deg = [np.rad2deg(j) for j in joints_rad]
        self.follower.rm_movej_canfd(joints_deg, False, 0)
    
    def send_gripper_position(self, position):
        """
        发送夹爪位置
        
        Args:
            position: 主臂夹爪位置 (约 0.066~0.971 范围)
        """
        if self.gripper is None or not self.gripper_initialized:
            return

        # 归一化: 0=主臂夹紧, 1=主臂张开
        normalized = (position - LEADER_GRIPPER_MIN) / (LEADER_GRIPPER_MAX - LEADER_GRIPPER_MIN)
        normalized = 0.5 + (normalized - 0.5) * GRIPPER_GAIN
        normalized = max(0.0, min(1.0, normalized))  # 限制在 0~1

        # 新 SDK: move_to_pos 0=夹紧 / 1000=张开 (方向与旧版相反, 不再取反)
        gripper_pos = int(normalized * 1000)
        self.gripper.move_to_pos(gripper_pos)
    
    def disconnect(self):
        """断开所有连接"""
        if self.follower:
            self.follower.rm_delete_robot_arm()
            print(f"   {self.name}从臂已断开")
        
        if self.leader:
            self.leader.disconnect()
            print(f"   {self.name}主臂已断开")
        



def teleop_loop(arms: list, target_hz: int = 30):
    """
    遥操作主循环
    
    Args:
        arms: ArmController 列表
        target_hz: 目标频率
    """
    global stop_flag
    
    target_period = 1.0 / target_hz
    loop_count = 0
    start_time = time.time()
    loop_times = []
    
    print(f"\n开始遥操作 (目标 {target_hz}Hz)...")
    print(f"活动臂: {', '.join([arm.name for arm in arms])}")
    print("按 Ctrl+C 停止\n")
    
    try:
        while not stop_flag:
            loop_start = time.perf_counter()
            
            for arm in arms:
                # 读取主臂位置
                joints_rad, gripper = arm.read_leader_position()
                
                if joints_rad is not None:
                    # 发送到从臂
                    arm.send_follower_position(joints_rad)
                    
                    # 发送夹爪命令
                    if gripper is not None:
                        arm.send_gripper_position(gripper)
            
            loop_count += 1
            loop_time = time.perf_counter() - loop_start
            loop_times.append(loop_time * 1000)
            
            # 每秒打印一次状态
            if loop_count % target_hz == 0:
                elapsed = time.time() - start_time
                fps = loop_count / elapsed
                avg_loop_ms = np.mean(loop_times[-target_hz:])
                
                status_parts = []
                for arm in arms:
                    _, gripper = arm.read_leader_position()
                    if gripper is not None:
                        status_parts.append(f"{arm.name}夹爪:{gripper:.2f}")
                
                status_str = " | ".join(status_parts)
                print(f"\r[{loop_count:5d}] FPS: {fps:.1f} | {status_str} | 循环: {avg_loop_ms:.2f}ms", end="")
            
            # 控制频率
            elapsed = time.perf_counter() - loop_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        pass
    
    # 统计
    elapsed = time.time() - start_time
    avg_fps = loop_count / elapsed if elapsed > 0 else 0
    avg_loop_ms = np.mean(loop_times) if loop_times else 0
    
    print(f"\n\n统计:")
    print(f"   总循环次数: {loop_count}")
    print(f"   运行时间: {elapsed:.1f}s")
    print(f"   平均频率: {avg_fps:.1f} Hz")
    print(f"   平均循环耗时: {avg_loop_ms:.2f} ms")


def run_teleop(mode: str):
    """
    运行遥操作
    
    Args:
        mode: "left", "right", 或 "dual"
    """
    print("=" * 60)
    print(f"  睿尔曼 RM75b 遥操作 - {mode.upper()} 模式")
    print("=" * 60)
    
    arms = []
    
    # 1. 创建控制器
    if mode in ["left", "dual"]:
        left_arm = ArmController("left")
        arms.append(left_arm)
    
    if mode in ["right", "dual"]:
        right_arm = ArmController("right")
        arms.append(right_arm)
    
    # 2. 连接设备
    print("\n1. 连接设备...")
    print("\n⚠️  夹爪初始化时会自动夹紧，请确保无障碍物！")
    input("按 Enter 继续...")
    
    for arm in arms:
        if not arm.connect_all():
            print(f"\n❌ {arm.name}连接失败，退出")
            # 断开已连接的设备
            for a in arms:
                a.disconnect()
            return
    
    # 3. 读取初始位置
    print("\n2. 读取初始位置...")
    for arm in arms:
        joints_rad, gripper = arm.read_leader_position()
        print(f"   {arm.name}主臂: {[f'{j:.3f}' for j in joints_rad]}")
        print(f"   {arm.name}夹爪: {gripper:.3f}")
    
    # 4. 开始遥操作
    print("\n3. 开始遥操作...")
    teleop_loop(arms, target_hz=30)
    
    # 5. 清理
    print("\n4. 清理...")
    for arm in arms:
        arm.disconnect()
    
    print("\n✅ 遥操作完成!")


def main():
    parser = argparse.ArgumentParser(description="双臂遥操作测试脚本（带夹爪）")
    parser.add_argument("-m", "--mode", type=str, default=None,
                        choices=["left", "right", "dual"],
                        help="遥操作模式: left=仅左臂, right=仅右臂, dual=双臂")
    args = parser.parse_args()
    
    if args.mode:
        mode = args.mode
    else:
        print("\n" + "=" * 40)
        print("选择遥操作模式:")
        print("=" * 40)
        print("  1. 仅左臂")
        print("  2. 仅右臂")
        print("  3. 双臂")
        print("=" * 40)
        choice = input("请选择 (1/2/3): ").strip()
        
        if choice == "1":
            mode = "left"
        elif choice == "2":
            mode = "right"
        elif choice == "3":
            mode = "dual"
        else:
            print("无效选择，退出")
            return
    
    run_teleop(mode)


if __name__ == "__main__":
    main()
