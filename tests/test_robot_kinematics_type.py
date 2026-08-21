import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from deployment.robots.rm_base_umi_dual.rm_base_umi_dual import RmBaseUmiDual
from deployment.robots.rm_isf_umi_left.rm_isf_umi_left import RmIsfUmiLeft
from deployment.robots import RobotConfig, make_robot_from_config
from vtla.engine.configs import PreTrainedConfig
from vtla.engine.utils.ee_kinematics import make_realman_algo
from vtla.engine.utils.constants import OBS_STATE
from vtla.frameworks.episode_ee_processor import EpisodeEEPreprocessorStep
from vtla.frameworks.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig


class RobotKinematicsTypeTest(unittest.TestCase):
    base_type = RobotConfig.get_choice_name(RmBaseUmiDual.config_class)
    isf_type = RobotConfig.get_choice_name(RmIsfUmiLeft.config_class)

    def test_only_current_robot_types_are_supported(self):
        self.assertEqual(
            RobotConfig.get_kinematics_robot_types(),
            ["rm_base_umi_dual", "rm_isf_umi_left"],
        )
        with self.assertRaisesRegex(ValueError, "Unsupported robot_type"):
            RobotConfig.get_kinematics_force_type("realman_ugripper_dual")

    def test_robot_types_select_expected_force_variant(self):
        self.assertEqual(RobotConfig.get_kinematics_force_type(self.base_type), "base")
        self.assertEqual(RobotConfig.get_kinematics_force_type(self.isf_type), "isf")
        self.assertEqual(RmBaseUmiDual.config_class.kinematics_sides, ("right", "left"))
        self.assertEqual(RmIsfUmiLeft.config_class.kinematics_sides, ("left",))
        self.assertEqual(RmBaseUmiDual.name, self.base_type)
        self.assertEqual(RmIsfUmiLeft.name, self.isf_type)

    def test_dynamic_factory_resolves_renamed_robot_classes(self):
        self.assertIsInstance(make_robot_from_config(RmBaseUmiDual.config_class()), RmBaseUmiDual)
        self.assertIsInstance(make_robot_from_config(RmIsfUmiLeft.config_class()), RmIsfUmiLeft)

    def test_robot_type_rejects_a_mismatched_dataset_layout(self):
        with self.assertRaisesRegex(ValueError, "Dataset arm layout"):
            RobotConfig.validate_kinematics_sides(self.base_type, ("left",))

    def test_isf_fk_contains_the_physical_ee_offset(self):
        joints_deg = [0.0] * 7
        base_pose = np.asarray(
            make_realman_algo(RmBaseUmiDual.kinematics_force_type).rm_algo_forward_kinematics(
                joints_deg, flag=0
            )
        )
        isf_pose = np.asarray(
            make_realman_algo(RmIsfUmiLeft.kinematics_force_type).rm_algo_forward_kinematics(
                joints_deg, flag=0
            )
        )

        np.testing.assert_allclose(isf_pose[:2], base_pose[:2], atol=1e-7)
        self.assertAlmostEqual(float(isf_pose[2] - base_pose[2]), 0.0172, places=5)
        np.testing.assert_allclose(isf_pose[3:], base_pose[3:], atol=1e-7)

    def test_inference_ee_processor_uses_checkpoint_robot_type(self):
        names = [f"left_main_joint{i}" for i in range(1, 8)] + ["left_main_gripper"]
        step = EpisodeEEPreprocessorStep(
            state_feature_names=names,
            relative_to_baseline=False,
            rot_mode="rot6d",
            n_arms=1,
            robot_type=self.isf_type,
        )

        observation = step.observation({OBS_STATE: torch.zeros(8)})

        self.assertAlmostEqual(float(observation[OBS_STATE][2]), 0.8677, places=4)

    def test_policy_config_round_trips_robot_type(self):
        config = StarvlaGrootConfig(robot_type=self.isf_type, device="cpu")
        with TemporaryDirectory() as directory:
            config._save_pretrained(Path(directory))
            loaded = PreTrainedConfig.from_pretrained(directory)

        self.assertEqual(loaded.robot_type, self.isf_type)


if __name__ == "__main__":
    unittest.main()
