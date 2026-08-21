import json
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.spatial.transform import Rotation as R

from deployment.robots.rm_isf_umi_left.config_rm_isf_umi_left import (
    RmIsfUmiLeftConfig,
)
from deployment.robots.rm_base_umi_dual.config_rm_base_umi_dual import (
    RmBaseUmiDualConfig,
)
from deployment.robots.rm_isf_umi_left.rm_isf_umi_left import RmIsfUmiLeft
from deployment.robots import RobotConfig
from vtla.engine.processor.relative_action_processor import (
    ACTION_ANCHOR,
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from vtla.engine.types import TransitionKey
from vtla.engine.utils.constants import ACTION, OBS_STATE
from vtla.engine.utils.ee_kinematics import make_realman_algo
from vtla.frameworks.episode_ee_processor import (
    ActionAnchorPreprocessorStep,
    QuatActionToRot6dStep,
)


DATASET_ROOT = Path(
    "playground/data/20260819_183840_rm_tactile_demo1_undist_uint8_256"
)


@unittest.skipUnless(DATASET_ROOT.exists(), f"processed test dataset missing: {DATASET_ROOT}")
class InferenceRobotActionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        info = json.loads((DATASET_ROOT / "meta/info.json").read_text())
        cls.state_names = info["features"][OBS_STATE]["names"]
        cls.ee_action_names = info["features"]["action_absolute_ee"]["names"]
        columns = [
            OBS_STATE,
            ACTION,
            "observation.state_absolute_ee",
            "action_absolute_ee",
            "observation.state_absolute_quat",
            "action_absolute_quat",
        ]
        table = pq.read_table(
            DATASET_ROOT / "data/chunk-000/file-000.parquet", columns=columns
        )
        cls.rows = table.to_pylist()

    def test_all_action_modes_recover_absolute_robot_targets(self):
        row = self.rows[0]
        raw_state = torch.tensor(row[OBS_STATE], dtype=torch.float32)

        cases = (
            ("absolute_joint", ACTION, ACTION),
            ("relative_joint", ACTION, ACTION),
            ("absolute_rot6d", "action_absolute_ee", "action_absolute_ee"),
            ("relative_rot6d", "action_absolute_ee", "action_absolute_ee"),
            ("absolute_quat", "action_absolute_quat", "action_absolute_ee"),
            ("relative_quat", "action_absolute_quat", "action_absolute_ee"),
        )

        for action_mode, source_key, expected_key in cases:
            with self.subTest(action_mode=action_mode):
                reference, representation = action_mode.split("_", 1)
                target = torch.tensor(row[source_key], dtype=torch.float32).unsqueeze(0)

                anchor_step = ActionAnchorPreprocessorStep(
                    state_feature_names=self.state_names,
                    representation=representation,
                    n_arms=1,
                    # This legacy fixture's EE columns were generated with B FK.
                    robot_type=RobotConfig.get_choice_name(RmBaseUmiDualConfig),
                )
                observation = anchor_step.observation({OBS_STATE: raw_state.clone()})
                anchor = observation[ACTION_ANCHOR].unsqueeze(0)

                relative_step = RelativeActionsProcessorStep(
                    enabled=reference == "relative",
                    exclude_joints=["gripper"],
                    action_names=self.state_names,
                    mode="joint" if representation == "joint" else "pose",
                    n_arms=1,
                    rot_mode=representation if representation != "joint" else "rot6d",
                )
                transition = {
                    TransitionKey.OBSERVATION: {ACTION_ANCHOR: anchor},
                    TransitionKey.ACTION: target,
                }
                encoded = relative_step(transition)
                decoded = AbsoluteActionsProcessorStep(
                    enabled=reference == "relative", relative_step=relative_step
                )(encoded)[TransitionKey.ACTION]

                if representation == "quat":
                    decoded = QuatActionToRot6dStep(n_arms=1).action(decoded)

                expected = torch.tensor(row[expected_key], dtype=torch.float32).unsqueeze(0)
                torch.testing.assert_close(decoded, expected, atol=2e-6, rtol=2e-6)

    def test_all_dataset_ee_targets_have_valid_realman_ik(self):
        from Robotic_Arm.rm_ctypes_wrap import rm_inverse_kinematics_params_t

        algo = make_realman_algo("base")
        max_position_error = 0.0
        max_rotation_error = 0.0

        for index, row in enumerate(self.rows):
            current = np.asarray(row[OBS_STATE][:7], dtype=np.float64)
            target = np.asarray(row[ACTION][:7], dtype=np.float64)
            target_pose = algo.rm_algo_forward_kinematics(
                np.degrees(target).tolist(), flag=0
            )
            params = rm_inverse_kinematics_params_t(
                q_in=np.degrees(current).tolist(), q_pose=target_pose, flag=0
            )
            result, solution = algo.rm_algo_inverse_kinematics(params)
            self.assertEqual(result, 0, f"IK failed at dataset frame {index}")

            recovered_pose = algo.rm_algo_forward_kinematics(solution, flag=0)
            position_error = np.linalg.norm(
                np.asarray(recovered_pose[:3]) - np.asarray(target_pose[:3])
            )
            target_rotation = R.from_quat(
                [target_pose[4], target_pose[5], target_pose[6], target_pose[3]]
            )
            recovered_rotation = R.from_quat(
                [recovered_pose[4], recovered_pose[5], recovered_pose[6], recovered_pose[3]]
            )
            rotation_error = (recovered_rotation.inv() * target_rotation).magnitude()
            max_position_error = max(max_position_error, float(position_error))
            max_rotation_error = max(max_rotation_error, float(rotation_error))

        self.assertLess(max_position_error, 2e-6)
        self.assertLess(max_rotation_error, 2e-6)

    def test_robot_adapter_passes_the_absolute_pose_to_controller_ik(self):
        class FakeFollower:
            def __init__(self):
                self.pose = None

            def send_pose(self, pose, **_kwargs):
                self.pose = pose
                return 0

        row = self.rows[0]
        robot = RmIsfUmiLeft(
            RmIsfUmiLeftConfig(
                action_space="ee",
                use_tactile=False,
                max_ee_pos_step_m=None,
                max_ee_rot_step_deg=None,
            )
        )
        follower = FakeFollower()
        robot._arms["left"].follower = follower
        robot._current_flange = lambda _side: (np.zeros(3), np.eye(3))

        action = dict(zip(self.ee_action_names, row["action_absolute_ee"], strict=True))
        robot._send_action_ee(action)

        target_joints = np.asarray(row[ACTION][:7], dtype=np.float64)
        expected_pose = make_realman_algo("base").rm_algo_forward_kinematics(
            np.degrees(target_joints).tolist(), flag=0
        )
        np.testing.assert_allclose(follower.pose, expected_pose, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
