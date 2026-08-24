#!/usr/bin/env python

"""RealMan ugripper 单右主臂遥操作器配置。"""

from dataclasses import dataclass

from deployment.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rm_leader_right")
@dataclass
class RmLeaderRightConfig(TeleoperatorConfig):
    """通过稳定的 udev 串口软链接读取单右主臂。"""

    port: str = "/dev/ttyRealmanISFLeaderR"
    baudrate: int = 460800
    hex_data: str = "55 AA 02 00 00 67"

    use_degrees: bool = False
    async_read: bool = True

    # 主臂夹爪原始读数范围：min=夹紧，max=张开。
    leader_gripper_min: float = 0.066
    leader_gripper_max: float = 0.971
    gripper_gain: float = 1.0

    id: str | None = "rm_leader_right"

    def __post_init__(self) -> None:
        if self.leader_gripper_max <= self.leader_gripper_min:
            raise ValueError("leader_gripper_max 必须大于 leader_gripper_min")
        if self.gripper_gain <= 0:
            raise ValueError("gripper_gain 必须大于 0")
