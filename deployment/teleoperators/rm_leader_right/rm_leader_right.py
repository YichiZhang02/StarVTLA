#!/usr/bin/env python

"""RealMan ugripper 单右主臂遥操作器。"""

from deployment.teleoperators.rm_leader_left.rm_leader_left import (
    RmLeaderLeft,
)

from .config_rm_leader_right import RmLeaderRightConfig


class RmLeaderRight(RmLeaderLeft):
    """输出与 rm_isf_umi_right 对齐的 8 维 right_* 动作。"""

    config_class = RmLeaderRightConfig
    name = "rm_leader_right"
    SIDE = "right"
    SIDE_LABEL = "右"

    def __init__(self, config: RmLeaderRightConfig):
        super().__init__(config)
