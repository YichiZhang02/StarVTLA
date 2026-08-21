import unittest

import numpy as np

from tools.convert_joints_to_eepose import (
    compute_relative_ee_stats,
    compute_relative_joint_stats,
)
from vtla.frameworks.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig


def _rot6d_pose(x: float) -> np.ndarray:
    return np.array([x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


class ActionGapTest(unittest.TestCase):
    def test_starvla_target_window_is_shifted_for_every_action_mode(self) -> None:
        for action_mode in (
            "absolute_joint",
            "relative_joint",
            "absolute_rot6d",
            "relative_rot6d",
            "absolute_quat",
            "relative_quat",
        ):
            with self.subTest(action_mode=action_mode):
                config = StarvlaGrootConfig(
                    device="cpu",
                    chunk_size=32,
                    n_action_steps=16,
                    action_mode=action_mode,
                    action_gap=6,
                )
                self.assertEqual(config.action_delta_indices, list(range(6, 38)))

    def test_negative_action_gap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "action_gap must be non-negative"):
            StarvlaGrootConfig(device="cpu", action_gap=-1)

    def test_relative_joint_stats_support_gap_zero(self) -> None:
        per_episode = {
            0: {
                "joint_state": [np.array([0.0]), np.array([1.0]), np.array([2.0])],
                "joint_action": [np.array([0.5]), np.array([1.5]), np.array([2.5])],
            }
        }
        stats = compute_relative_joint_stats(
            per_episode,
            horizon=1,
            relative_mask=np.array([True]),
            action_gap=0,
        )
        self.assertEqual(stats["count"].item(), 3)
        self.assertAlmostEqual(stats["mean"].item(), 0.5)

    def test_relative_pose_stats_use_requested_gap_window(self) -> None:
        poses = [_rot6d_pose(float(index)) for index in range(4)]
        per_episode = {0: {"s_abs": poses, "a_abs": poses}}

        stats = compute_relative_ee_stats(
            per_episode, horizon=2, n_arms=1, action_gap=2
        )

        # k=2 contributes two samples and k=3 contributes one sample.
        self.assertEqual(stats["count"].item(), 3)
        self.assertAlmostEqual(stats["mean"][0], 7.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
