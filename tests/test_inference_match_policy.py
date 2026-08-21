import unittest
from types import SimpleNamespace

from deployment.inference import _resolve_robot_type
from deployment.robots import RobotConfig
from deployment.robots.rm_base_umi_dual.config_rm_base_umi_dual import (
    RmBaseUmiDualConfig,
)
from deployment.robots.rm_isf_umi_left.config_rm_isf_umi_left import (
    RmIsfUmiLeftConfig,
)

BASE_DUAL_TYPE = RobotConfig.get_choice_name(RmBaseUmiDualConfig)
ISF_LEFT_TYPE = RobotConfig.get_choice_name(RmIsfUmiLeftConfig)


def _cfg(robot, *, robot_type=None):
    policy = SimpleNamespace(
        robot_type=robot_type,
    )
    return SimpleNamespace(robot=robot, policy=policy)


class MatchPolicyRobotTypeTest(unittest.TestCase):
    def test_single_left_checkpoint_replaces_dual_robot(self):
        cfg = _cfg(
            RmBaseUmiDualConfig(
                home_duration_s=2.0,
                max_ee_pos_step_m=0.01,
                use_tactile=False,
            ),
            robot_type=ISF_LEFT_TYPE,
        )

        _resolve_robot_type(cfg)

        self.assertIsInstance(cfg.robot, RmIsfUmiLeftConfig)
        self.assertEqual(cfg.robot.home_duration_s, 2.0)
        self.assertEqual(cfg.robot.max_ee_pos_step_m, 0.01)
        self.assertFalse(cfg.robot.use_tactile)
        self.assertEqual(cfg.robot.follower_ip, "192.168.1.201")
        self.assertEqual(cfg.policy.robot_type, ISF_LEFT_TYPE)

    def test_dual_checkpoint_replaces_left_robot(self):
        cfg = _cfg(
            RmIsfUmiLeftConfig(home_duration_s=3.0),
            robot_type=BASE_DUAL_TYPE,
        )

        _resolve_robot_type(cfg)

        self.assertIsInstance(cfg.robot, RmBaseUmiDualConfig)
        self.assertEqual(cfg.robot.arms, ["left", "right"])
        self.assertEqual(cfg.robot.home_duration_s, 3.0)
        self.assertEqual(cfg.policy.robot_type, BASE_DUAL_TYPE)

    def test_missing_checkpoint_robot_type_is_rejected(self):
        cfg = _cfg(RmBaseUmiDualConfig(), robot_type=None)

        with self.assertRaisesRegex(ValueError, "Missing robot_type"):
            _resolve_robot_type(cfg)

    def test_legacy_checkpoint_robot_type_is_rejected(self):
        cfg = _cfg(RmBaseUmiDualConfig(), robot_type="realman_ugripper_dual")

        with self.assertRaisesRegex(ValueError, "Unsupported robot_type"):
            _resolve_robot_type(cfg)


if __name__ == "__main__":
    unittest.main()
