#!/usr/bin/env python

"""睿尔曼 RM75b 单右臂 ugripper LeRobot 适配器。"""

from deployment.robots.rm_isf_umi_left.rm_isf_umi_left import RmIsfUmiLeft

from .config_rm_isf_umi_right import RmIsfUmiRightConfig


class RmIsfUmiRight(RmIsfUmiLeft):
    """使用 right_* 特征名的睿尔曼 RM75b 单右臂 ugripper。"""

    config_class = RmIsfUmiRightConfig
    name = "rm_isf_umi_right"
    kinematics_force_type = RmIsfUmiRightConfig.kinematics_force_type
    SIDE = "right"
    SIDES = (SIDE,)

    def __init__(self, config: RmIsfUmiRightConfig):
        super().__init__(config)
