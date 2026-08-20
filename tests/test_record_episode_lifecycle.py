import unittest
from unittest.mock import patch

from deployment import _record_engine
from deployment.collect import CollectConfig, CollectMode
from deployment.inference import InferenceConfig


def _events() -> dict[str, bool]:
    return {
        "start_episode": False,
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }


class RecordEpisodeLifecycleTest(unittest.TestCase):
    def test_collect_modes_exclude_mix(self):
        self.assertEqual({mode.value for mode in CollectMode}, {"teleop", "drag"})

    def test_entrypoints_have_distinct_reset_defaults(self):
        self.assertFalse(CollectConfig.__dataclass_fields__["reset_before_episode"].default)
        self.assertTrue(InferenceConfig.__dataclass_fields__["reset_before_episode"].default)

    def test_collect_preparation_holds_current_position(self):
        events = _events()

        def press_start(_seconds: float) -> None:
            events["start_episode"] = True

        with (
            patch.object(_record_engine, "move_to_home_smooth") as move_home,
            patch.object(_record_engine, "log_say"),
            patch.object(_record_engine.time, "sleep", side_effect=press_start),
        ):
            started = _record_engine.wait_for_episode_start(
                robot=object(),
                events=events,
                episode_label="Episode 1/2",
                fps=30,
                play_sounds=False,
                reset_before_episode=False,
                home_action={},
                home_duration_s=4.0,
            )

        self.assertTrue(started)
        move_home.assert_not_called()

    def test_inference_preparation_resets_before_start(self):
        events = _events()
        robot = object()
        home_action = {"left_main_joint1": 0.0}

        def press_start(_seconds: float) -> None:
            events["start_episode"] = True

        with (
            patch.object(_record_engine, "move_to_home_smooth") as move_home,
            patch.object(_record_engine, "log_say"),
            patch.object(_record_engine.time, "sleep", side_effect=press_start),
        ):
            started = _record_engine.wait_for_episode_start(
                robot=robot,
                events=events,
                episode_label="Episode 1/2",
                fps=30,
                play_sounds=False,
                reset_before_episode=True,
                home_action=home_action,
                home_duration_s=4.0,
            )

        self.assertTrue(started)
        move_home.assert_called_once_with(robot, home_action, 30, 4.0)


if __name__ == "__main__":
    unittest.main()
