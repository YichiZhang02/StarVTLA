from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from deployment.robots.rm_base_umi_dual.rm_base_umi_dual import RmBaseUmiDual
from deployment.robots.rm_base_umi_dual.config_rm_base_umi_dual import RmBaseUmiDualConfig
from deployment.robots.rm_isf_umi_left.rm_isf_umi_left import RmIsfUmiLeft
from deployment.robots.rm_isf_umi_left.config_rm_isf_umi_left import RmIsfUmiLeftConfig
from vtla.engine.processor.relative_action_processor import ACTION_ANCHOR
from vtla.engine.utils.constants import OBS_STATE
from vtla.engine.utils.ee_kinematics import flange_to_tcp, mat_to_rot6d, tcp_to_flange
from vtla.frameworks.episode_ee_processor import ActionAnchorPreprocessorStep


XYZ = (-0.01035, 0.0, 0.06471)
RPY = (0.0, 0.0, 180.0)


def test_flange_tcp_round_trip():
    flange_pos = np.array([0.34, -0.12, 0.58])
    flange_rot = R.from_euler("xyz", [21.0, -17.0, 83.0], degrees=True).as_matrix()

    tcp_pos, tcp_rot = flange_to_tcp(flange_pos, flange_rot, XYZ, RPY)
    actual_pos, actual_rot = tcp_to_flange(tcp_pos, tcp_rot, XYZ, RPY)

    np.testing.assert_allclose(actual_pos, flange_pos, atol=1e-12)
    np.testing.assert_allclose(actual_rot, flange_rot, atol=1e-12)


class _FakeAlgo:
    def rm_algo_forward_kinematics(self, joints_deg, flag=0):
        assert flag == 0
        return [0.4, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0]


def test_relative_action_anchor_uses_tcp(monkeypatch):
    monkeypatch.setattr(
        "vtla.frameworks.episode_ee_processor.make_realman_algo", lambda _force_type: _FakeAlgo()
    )
    names = [f"left_main_joint{i}" for i in range(1, 8)] + ["left_main_gripper"]
    step = ActionAnchorPreprocessorStep(
        state_feature_names=names,
        representation="rot6d",
        n_arms=1,
        robot_type="rm_isf_umi_left",
        ee_frame="tcp",
    )

    observation = {OBS_STATE: np.array([0.0] * 7 + [0.65], dtype=np.float32)}
    step.observation(observation)

    expected_pos, expected_rot = flange_to_tcp(
        np.array([0.4, -0.2, 0.3]), np.eye(3), XYZ, RPY
    )
    expected = np.concatenate([expected_pos, mat_to_rot6d(expected_rot), [0.65]])
    np.testing.assert_allclose(observation[ACTION_ANCHOR].numpy(), expected, atol=1e-7)


class _Follower:
    def __init__(self):
        self.pose = None

    def send_pose(self, pose, **kwargs):
        self.pose = pose


def _action(side: str, pos: np.ndarray, rot: np.ndarray) -> dict[str, float]:
    values = {
        f"{side}_ee_x": float(pos[0]),
        f"{side}_ee_y": float(pos[1]),
        f"{side}_ee_z": float(pos[2]),
        f"{side}_gripper": 0.4,
    }
    values.update(
        {f"{side}_ee_rot6d_{i}": float(v) for i, v in enumerate(mat_to_rot6d(rot))}
    )
    return values


def _bare_robot(robot_cls, config, side: str):
    robot = robot_cls.__new__(robot_cls)
    robot.config = config
    follower = _Follower()
    robot._arms = {arm_side: SimpleNamespace(follower=None, gripper=None) for arm_side in config.kinematics_sides}
    robot._arms[side] = SimpleNamespace(follower=follower, gripper=None)
    robot._current_flange = lambda _side: (np.array([0.2, 0.1, 0.5]), np.eye(3))
    return robot, follower


def test_tcp_action_is_converted_to_flange_for_single_arm_sdk():
    config = RmIsfUmiLeftConfig(
        ee_frame="tcp", max_ee_pos_step_m=None, max_ee_rot_step_deg=None
    )
    robot, follower = _bare_robot(RmIsfUmiLeft, config, "left")
    tcp_pos = np.array([0.51, -0.08, 0.42])
    tcp_rot = R.from_euler("xyz", [5.0, 10.0, -20.0], degrees=True).as_matrix()

    sent = robot._send_action_ee(_action("left", tcp_pos, tcp_rot))

    expected_pos, expected_rot = tcp_to_flange(tcp_pos, tcp_rot, XYZ, RPY)
    np.testing.assert_allclose(follower.pose[:3], expected_pos, atol=1e-7)
    actual_rot = R.from_quat(
        [follower.pose[4], follower.pose[5], follower.pose[6], follower.pose[3]]
    ).as_matrix()
    np.testing.assert_allclose(actual_rot, expected_rot, atol=1e-7)
    np.testing.assert_allclose(
        [sent[f"left_ee_{axis}"] for axis in ("x", "y", "z")], tcp_pos, atol=1e-7
    )


def test_flange_mode_preserves_dual_arm_sdk_target():
    config = RmBaseUmiDualConfig(
        ee_frame="flange", max_ee_pos_step_m=None, max_ee_rot_step_deg=None
    )
    robot, follower = _bare_robot(RmBaseUmiDual, config, "right")
    flange_pos = np.array([0.45, 0.06, 0.38])
    flange_rot = R.from_euler("xyz", [-8.0, 13.0, 25.0], degrees=True).as_matrix()

    robot._send_action_ee(_action("right", flange_pos, flange_rot))

    np.testing.assert_allclose(follower.pose[:3], flange_pos, atol=1e-7)
    actual_rot = R.from_quat(
        [follower.pose[4], follower.pose[5], follower.pose[6], follower.pose[3]]
    ).as_matrix()
    np.testing.assert_allclose(actual_rot, flange_rot, atol=1e-7)
