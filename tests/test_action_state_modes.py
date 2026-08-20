import unittest

import torch

from vtla.engine.processor.relative_action_processor import (
    ACTION_ANCHOR,
    RelativeActionsProcessorStep,
    route_ee_batch,
)
from vtla.engine.types import TransitionKey
from vtla.engine.utils.constants import ACTION, OBS_STATE
from vtla.engine.utils.ee_transforms import ee_to_absolute
from vtla.frameworks.sensor_routing import SensorRoutingMixin
from vtla.frameworks.episode_ee_processor import QuatActionToRot6dStep
from vtla.frameworks.sensor_routing import (
    ACTION_ABSOLUTE_QUAT,
    OBS_STATE_ABSOLUTE_QUAT,
    OBS_STATE_EPISODE_EE,
)


class ActionStateModeTest(unittest.TestCase):
    def test_compound_modes_are_split_and_legacy_modes_migrate(self):
        cases = [
            ("none", "relative_joint", ("absolute", "none", "relative", "joint")),
            ("episode_joint", "absolute_quat", ("episode", "joint", "absolute", "quat")),
            ("absolute_rot6d", "relative_quat", ("absolute", "rot6d", "relative", "quat")),
            ("joint", "relative_ee", ("absolute", "joint", "relative", "rot6d")),
        ]
        for state_mode, action_mode, expected in cases:
            with self.subTest(state_mode=state_mode, action_mode=action_mode):
                config = SensorRoutingMixin(state_mode=state_mode, action_mode=action_mode)
                config.validate_sensor_modes()
                self.assertEqual(
                    (
                        config.state_reference,
                        config.state_representation,
                        config.action_reference,
                        config.action_representation,
                    ),
                    expected,
                )

    def test_invalid_compound_mode_is_rejected(self):
        config = SensorRoutingMixin(state_mode="none", action_mode="delta_joint")
        with self.assertRaisesRegex(ValueError, "Invalid action_mode"):
            config.validate_sensor_modes()

    def test_none_state_still_routes_hidden_relative_joint_anchor(self):
        state = torch.tensor([[1.0, 0.4]])
        action = torch.tensor([[[1.2, 0.8]]])
        batch = route_ee_batch(
            {OBS_STATE: state, ACTION: action},
            state_mode="none",
            action_mode="relative_joint",
        )
        self.assertNotIn(OBS_STATE, batch)
        torch.testing.assert_close(batch[ACTION_ANCHOR], state)

        step = RelativeActionsProcessorStep(
            enabled=True,
            action_names=["left_joint1", "left_gripper"],
            exclude_joints=["gripper"],
        )
        transition = {
            TransitionKey.OBSERVATION: {ACTION_ANCHOR: batch[ACTION_ANCHOR]},
            TransitionKey.ACTION: batch[ACTION],
        }
        converted = step(transition)
        torch.testing.assert_close(converted[TransitionKey.ACTION], torch.tensor([[[0.2, 0.8]]]))
        self.assertNotIn(ACTION_ANCHOR, converted[TransitionKey.OBSERVATION])

    def test_mixed_ee_modes_route_state_action_and_anchor_independently(self):
        episode_rot6d = torch.randn(2, 10)
        absolute_quat_state = torch.randn(2, 8)
        absolute_quat_action = torch.randn(2, 3, 8)
        batch = route_ee_batch(
            {
                OBS_STATE_EPISODE_EE: episode_rot6d,
                OBS_STATE_ABSOLUTE_QUAT: absolute_quat_state,
                ACTION_ABSOLUTE_QUAT: absolute_quat_action,
            },
            state_mode="episode_rot6d",
            action_mode="relative_quat",
        )
        torch.testing.assert_close(batch[OBS_STATE], episode_rot6d)
        torch.testing.assert_close(batch[ACTION], absolute_quat_action)
        torch.testing.assert_close(batch[ACTION_ANCHOR], absolute_quat_state)

    def test_relative_ee_postprocess_recovers_absolute_base_target(self):
        cases = [("rot6d", [1, 0, 0, 0, 1, 0]), ("quat", [0, 0, 0, 1])]
        for rot_mode, identity in cases:
            with self.subTest(rot_mode=rot_mode):
                reference = torch.tensor([[0.4, -0.2, 0.3, *identity, 0.5]], dtype=torch.float32)
                target = torch.tensor([[[0.5, -0.1, 0.35, *identity, 0.8]]], dtype=torch.float32)
                step = RelativeActionsProcessorStep(
                    enabled=True, mode="pose", n_arms=1, rot_mode=rot_mode
                )
                transition = {
                    TransitionKey.OBSERVATION: {ACTION_ANCHOR: reference},
                    TransitionKey.ACTION: target,
                }
                relative = step(transition)[TransitionKey.ACTION]
                recovered = ee_to_absolute(reference, relative, n_arms=1, rot_mode=rot_mode)
                torch.testing.assert_close(recovered, target, atol=1e-5, rtol=1e-5)

    def test_quat_action_is_adapted_to_robot_rot6d_layout(self):
        action = torch.tensor([[0.1, 0.2, 0.3, 0, 0, 0, 1, 0.7]], dtype=torch.float32)
        converted = QuatActionToRot6dStep(n_arms=1).action(action)
        self.assertEqual(converted.shape, (1, 10))
        torch.testing.assert_close(converted[:, :3], action[:, :3])
        torch.testing.assert_close(converted[:, 3:9], torch.tensor([[1, 0, 0, 0, 1, 0.0]]))
        torch.testing.assert_close(converted[:, -1], action[:, -1])


if __name__ == "__main__":
    unittest.main()
