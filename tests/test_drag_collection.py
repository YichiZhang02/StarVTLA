import unittest

import numpy as np

from deployment._record_engine import build_drag_action, record_loop, toggle_drag_gripper
from deployment.hardware.follower_arms.realman_tcp import RealmanTcpFollower


class _FakeRealmanSdk:
    def __init__(self):
        self.calls = []

    def rm_set_force_drag_mode(self, precise):
        self.calls.append(("set_force_drag_mode", precise))
        return 0

    def rm_get_force_drag_mode(self):
        self.calls.append(("get_force_drag_mode",))
        return 0, 1

    def rm_start_multi_drag_teach(self, mode, singular_wall):
        self.calls.append(("start_multi_drag_teach", mode, singular_wall))
        return 0

    def rm_get_joint_degree(self):
        self.calls.append(("get_joint_degree",))
        return 0, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    def rm_stop_drag_teach(self):
        self.calls.append(("stop_drag_teach",))
        return 0

    def rm_movej_canfd(self, joints, follow, trajectory_mode):
        self.calls.append(("movej_canfd", list(joints), follow, trajectory_mode))
        return 0


class _FakeDragRobot:
    name = "fake_drag"
    action_features = {
        "left_main_joint1": float,
        "left_main_joint2": float,
        "left_main_gripper": float,
    }

    def __init__(self):
        self.gripper_commands = []
        self.arm_send_calls = 0

    def get_observation(self):
        return {
            "left_main_joint1": 0.1,
            "left_main_joint2": 0.2,
            "left_main_gripper": 1.0,
        }

    def send_gripper_action(self, value):
        self.gripper_commands.append(value)

    def send_action(self, _action):
        self.arm_send_calls += 1
        raise AssertionError("drag mode must not send an arm action")


class _FakeStreamWriter:
    def __init__(self):
        self.observations = []

    def add_observation(self, observation):
        self.observations.append(observation)


class DragCollectionTest(unittest.TestCase):
    def test_force_drag_exit_latches_measured_joint_position(self):
        sdk = _FakeRealmanSdk()
        follower = RealmanTcpFollower("127.0.0.1", name="test")
        follower._arm = sdk
        follower._connected = True

        follower.start_six_axis_force_drag(precise=True, mode=3, singular_wall=True)
        follower.stop_drag_teach()

        self.assertEqual(
            sdk.calls[:3],
            [
                ("set_force_drag_mode", 1),
                ("get_force_drag_mode",),
                ("start_multi_drag_teach", 3, 1),
            ],
        )
        stop_index = sdk.calls.index(("stop_drag_teach",))
        hold_call = sdk.calls[stop_index + 1]
        self.assertEqual(hold_call[0], "movej_canfd")
        np.testing.assert_allclose(hold_call[1], [10, 20, 30, 40, 50, 60, 70])

    def test_drag_gripper_toggle_uses_configured_close_value(self):
        robot = _FakeDragRobot()
        observation = robot.get_observation()

        target = toggle_drag_gripper(
            robot,
            observation,
            None,
            open_value=1.0,
            close_value=0.25,
        )
        self.assertEqual(target, 0.25)
        target = toggle_drag_gripper(
            robot,
            observation,
            target,
            open_value=1.0,
            close_value=0.25,
        )
        self.assertEqual(target, 1.0)
        self.assertEqual(robot.gripper_commands, [0.25, 1.0])

    def test_drag_action_uses_measured_joints_and_commanded_gripper(self):
        robot = _FakeDragRobot()
        action = build_drag_action(robot, robot.get_observation(), gripper_target=0.3)
        self.assertEqual(
            action,
            {
                "left_main_joint1": 0.1,
                "left_main_joint2": 0.2,
                "left_main_gripper": 0.3,
            },
        )

    def test_drag_record_loop_never_sends_arm_action(self):
        robot = _FakeDragRobot()
        stream_writer = _FakeStreamWriter()
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "start_episode": False,
            "toggle_gripper": False,
        }

        record_loop(
            robot=robot,
            events=events,
            fps=1000,
            teleop_action_processor=lambda value: value[0],
            robot_action_processor=lambda value: value[0],
            robot_observation_processor=lambda value: value,
            record_features={},
            stream_writer=stream_writer,
            control_time_s=0.0001,
            single_task="drag test",
            drag_mode=True,
            drag_gripper_close_value=0.2,
        )

        self.assertEqual(robot.arm_send_calls, 0)
        self.assertGreaterEqual(len(stream_writer.observations), 1)


if __name__ == "__main__":
    unittest.main()
