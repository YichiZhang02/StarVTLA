from collections import deque
from types import SimpleNamespace
import unittest

import torch

from vtla.engine.common.control_utils import _lock_relative_action_anchor_for_new_chunk
from vtla.engine.processor.relative_action_processor import (
    ACTION_ANCHOR,
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from vtla.engine.types import TransitionKey
from vtla.frameworks.pretrained import PreTrainedPolicy


def _rot6d_pose(x: float) -> torch.Tensor:
    return torch.tensor([[x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])


def _quat_pose(x: float) -> torch.Tensor:
    return torch.tensor([[x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])


def _observe(step: RelativeActionsProcessorStep, state: torch.Tensor) -> None:
    step({TransitionKey.OBSERVATION: {ACTION_ANCHOR: state}})


def _decode(step: AbsoluteActionsProcessorStep, action: torch.Tensor) -> torch.Tensor:
    return step({TransitionKey.ACTION: action})[TransitionKey.ACTION]


class RelativeActionChunkAnchorTest(unittest.TestCase):
    def test_pose_chunk_uses_one_locked_anchor(self) -> None:
        for rot_mode, pose in (("rot6d", _rot6d_pose), ("quat", _quat_pose)):
            with self.subTest(rot_mode=rot_mode):
                relative = RelativeActionsProcessorStep(
                    enabled=True, mode="pose", n_arms=1, rot_mode=rot_mode
                )
                absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)

                _observe(relative, pose(0.0))
                relative.lock_action_anchor()
                self.assertAlmostEqual(_decode(absolute, pose(0.01))[0, 0].item(), 0.01)

                # A newer live state must not become the anchor for an action still in the old chunk.
                _observe(relative, pose(0.01))
                self.assertAlmostEqual(_decode(absolute, pose(0.02))[0, 0].item(), 0.02)

                # Replanning explicitly replaces the old chunk anchor with the latest state.
                relative.lock_action_anchor()
                self.assertAlmostEqual(_decode(absolute, pose(0.02))[0, 0].item(), 0.03)

    def test_relative_joint_chunk_uses_one_locked_anchor(self) -> None:
        relative = RelativeActionsProcessorStep(enabled=True, mode="joint", n_arms=1)
        absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)

        _observe(relative, torch.tensor([[1.0, 2.0]]))
        relative.lock_action_anchor()
        torch.testing.assert_close(
            _decode(absolute, torch.tensor([[0.1, 0.2]])), torch.tensor([[1.1, 2.2]])
        )

        _observe(relative, torch.tensor([[1.1, 2.2]]))
        torch.testing.assert_close(
            _decode(absolute, torch.tensor([[0.2, 0.4]])), torch.tensor([[1.2, 2.4]])
        )

    def test_controller_locks_only_when_action_queue_is_empty(self) -> None:
        relative = RelativeActionsProcessorStep(enabled=True, mode="joint", n_arms=1)
        preprocessor = SimpleNamespace(steps=[relative])
        policy = SimpleNamespace(_action_queue=deque())
        policy.is_action_queue_empty = lambda: len(policy._action_queue) == 0

        _observe(relative, torch.tensor([[1.0]]))
        _lock_relative_action_anchor_for_new_chunk(policy, preprocessor)
        torch.testing.assert_close(relative.get_cached_state(), torch.tensor([[1.0]]))

        policy._action_queue.append(torch.tensor([[0.1]]))
        _observe(relative, torch.tensor([[2.0]]))
        _lock_relative_action_anchor_for_new_chunk(policy, preprocessor)
        torch.testing.assert_close(relative.get_cached_state(), torch.tensor([[1.0]]))

        policy._action_queue.clear()
        _lock_relative_action_anchor_for_new_chunk(policy, preprocessor)
        torch.testing.assert_close(relative.get_cached_state(), torch.tensor([[2.0]]))

    def test_reset_clears_locked_anchor(self) -> None:
        relative = RelativeActionsProcessorStep(enabled=True, mode="joint", n_arms=1)
        _observe(relative, torch.tensor([[1.0]]))
        relative.lock_action_anchor()

        relative.reset()

        self.assertIsNone(relative.get_cached_state())

    def test_policy_queue_detection_supports_common_queue_layouts(self) -> None:
        direct = SimpleNamespace(_action_queue=deque())
        self.assertTrue(PreTrainedPolicy.is_action_queue_empty(direct))
        direct._action_queue.append(torch.tensor(1.0))
        self.assertFalse(PreTrainedPolicy.is_action_queue_empty(direct))

        diffusion = SimpleNamespace(_queues={"action": deque()})
        self.assertTrue(PreTrainedPolicy.is_action_queue_empty(diffusion))
        diffusion._queues["action"].append(torch.tensor(1.0))
        self.assertFalse(PreTrainedPolicy.is_action_queue_empty(diffusion))

        no_queue = SimpleNamespace()
        self.assertTrue(PreTrainedPolicy.is_action_queue_empty(no_queue))


if __name__ == "__main__":
    unittest.main()
