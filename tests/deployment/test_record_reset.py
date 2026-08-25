from types import SimpleNamespace

import numpy as np

from deployment import _record_engine


class FakeRobot:
    def __init__(self):
        self.observation_count = 0
        self.actions = []

    def get_observation(self):
        self.observation_count += 1
        return {
            "joint": 4.0,
            "camera": np.full((2, 2, 3), self.observation_count, dtype=np.uint8),
        }

    def send_action(self, action):
        self.actions.append(action.copy())


def test_reset_then_finalize_records_every_interpolated_step(monkeypatch):
    robot = FakeRobot()
    recorded = []
    finalized_at = []
    monkeypatch.setattr(_record_engine, "busy_wait", lambda _seconds: None)

    prepared_at_home = _record_engine.reset_then_finalize_episode(
        robot=robot,
        reset_before_episode=True,
        home_action={"joint": 0.0},
        fps=2,
        home_duration_s=2.0,
        play_sounds=False,
        episode_label="Episode 1",
        record_step=lambda obs, action: recorded.append((obs, action.copy())),
        finalize=lambda: finalized_at.append(len(recorded)),
    )

    assert prepared_at_home is True
    assert robot.observation_count == 4
    assert [action["joint"] for action in robot.actions] == [3.0, 2.0, 1.0, 0.0]
    assert [action for _, action in recorded] == robot.actions
    assert finalized_at == [4]


def test_record_reset_frame_appends_dataset_state_action_and_raw_tactile():
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["joint"],
        },
        "observation.images.camera": {
            "dtype": "video",
            "shape": (2, 2, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["joint"],
        },
    }
    dataset = SimpleNamespace(frames=[])
    dataset.add_frame = dataset.frames.append
    tactile_writer = SimpleNamespace(camera_keys=(), observations=[])
    tactile_writer.add_observation = tactile_writer.observations.append
    obs = {
        "joint": 2.0,
        "camera": np.ones((2, 2, 3), dtype=np.uint8),
    }

    _record_engine.record_reset_frame(
        obs=obs,
        action={"joint": 1.5},
        robot_observation_processor=lambda value: value,
        record_features=features,
        single_task="reset test",
        dataset=dataset,
        tactile_writer=tactile_writer,
    )

    assert len(dataset.frames) == 1
    np.testing.assert_array_equal(dataset.frames[0]["observation.state"], [2.0])
    np.testing.assert_array_equal(dataset.frames[0]["action"], [1.5])
    assert dataset.frames[0]["task"] == "reset test"
    assert tactile_writer.observations == [obs]


def test_record_reset_frame_appends_stream_observation():
    stream_writer = SimpleNamespace(observations=[])
    stream_writer.add_observation = stream_writer.observations.append
    obs = {"joint": 2.0, "camera": np.ones((2, 2, 3), dtype=np.uint8)}

    _record_engine.record_reset_frame(
        obs=obs,
        action={"joint": 1.5},
        robot_observation_processor=lambda value: {**value, "processed": True},
        record_features={},
        single_task="reset test",
        stream_writer=stream_writer,
    )

    assert stream_writer.observations == [{**obs, "processed": True}]
