from types import SimpleNamespace

import numpy as np

from deployment import _record_engine


class FakeRobot:
    def __init__(self, *, settle_timeout_s=2.0):
        self.observation_count = 0
        self.actions = []
        self.joint = 4.0
        self.config = SimpleNamespace(
            home_joint_tolerance_deg=1.0,
            home_settle_timeout_s=settle_timeout_s,
            use_degrees=False,
        )

    def get_observation(self):
        self.observation_count += 1
        return {
            "joint": self.joint,
            "camera": np.full((2, 2, 3), self.observation_count, dtype=np.uint8),
        }

    def send_action(self, action):
        self.actions.append(action.copy())
        self.joint = action["joint"]
        return action.copy()


def test_move_to_home_interpolates_and_confirms_feedback(monkeypatch):
    robot = FakeRobot()
    monkeypatch.setattr(_record_engine, "busy_wait", lambda _seconds: None)

    at_home = _record_engine.move_to_home_smooth(
        robot,
        {"joint": 0.0},
        fps=2,
        duration_s=2.0,
    )

    assert at_home is True
    assert robot.observation_count == 2
    np.testing.assert_allclose(
        [action["joint"] for action in robot.actions],
        [3.41421356, 2.0, 0.58578644, 0.0, 0.0],
    )


def test_move_to_home_timeout_is_not_confirmed(monkeypatch, caplog):
    robot = FakeRobot(settle_timeout_s=0.0)
    robot.send_action = lambda action: robot.actions.append(action.copy()) or action.copy()
    monkeypatch.setattr(_record_engine, "busy_wait", lambda _seconds: None)

    at_home = _record_engine.move_to_home_smooth(
        robot,
        {"joint": 0.0},
        fps=2,
        duration_s=0.5,
    )

    assert at_home is False
    assert "等待关节到位超时" in caplog.text


def _events():
    return {
        "start_episode": False,
        "exit_early": False,
        "rerecord_episode": False,
        "toggle_gripper": 0,
        "stop_recording": False,
    }


def test_up_starts_episode_without_reset(monkeypatch):
    events = _events()
    order = []

    def request_start(_seconds):
        order.append("start_requested")
        events["start_episode"] = True

    monkeypatch.setattr(_record_engine.time, "sleep", request_start)
    monkeypatch.setattr(
        _record_engine,
        "move_to_home_smooth",
        lambda *_args: (_ for _ in ()).throw(AssertionError("up must not reset")),
    )
    monkeypatch.setattr(_record_engine, "log_say", lambda *_args: None)

    started = _record_engine.wait_for_episode_start(
        events=events,
        episode_label="Episode 1",
        play_sounds=False,
        on_prepared=lambda: order.append("prepared"),
    )

    assert started is True
    assert order == ["start_requested", "prepared"]
    assert events["start_episode"] is False


def test_wait_for_inflight_policy_action_has_no_artificial_delay(monkeypatch):
    waits = []
    monkeypatch.setattr(_record_engine, "busy_wait", waits.append)

    _record_engine.wait_for_inflight_policy_action(fps=30)

    assert waits == [0.0]


def test_right_resets_before_save(monkeypatch):
    events = _events()
    events["exit_early"] = True
    order = []

    def reset_with_retry(*_args):
        confirmed = "reset_failed" in order
        order.append("home_confirmed" if confirmed else "reset_failed")
        return confirmed

    monkeypatch.setattr(_record_engine, "move_to_home_smooth", reset_with_retry)
    monkeypatch.setattr(_record_engine, "log_say", lambda *_args: None)

    finalized = _record_engine.reset_then_finalize_episode(
        robot=FakeRobot(),
        events=events,
        reset_before_episode=True,
        home_action={"joint": 0.0},
        fps=30,
        home_duration_s=2.0,
        play_sounds=False,
        episode_label="Episode 1",
        finalize=lambda: order.append("saved"),
    )

    assert finalized is True
    assert order == ["reset_failed", "home_confirmed", "saved"]


def test_left_resets_before_discard(monkeypatch):
    events = _events()
    events["rerecord_episode"] = True
    events["exit_early"] = True
    order = []
    monkeypatch.setattr(
        _record_engine,
        "move_to_home_smooth",
        lambda *_args: order.append("home_confirmed") or True,
    )
    monkeypatch.setattr(_record_engine, "log_say", lambda *_args: None)

    def discard():
        order.append("discarded")
        events["rerecord_episode"] = False

    finalized = _record_engine.reset_then_finalize_episode(
        robot=FakeRobot(),
        events=events,
        reset_before_episode=True,
        home_action={"joint": 0.0},
        fps=30,
        home_duration_s=2.0,
        play_sounds=False,
        episode_label="Episode 1",
        finalize=discard,
    )

    assert finalized is True
    assert order == ["home_confirmed", "discarded"]
    assert events["rerecord_episode"] is False


def test_failed_reset_does_not_finalize(monkeypatch):
    events = _events()
    finalized = []

    def fail_and_stop(*_args):
        events["stop_recording"] = True
        return False

    monkeypatch.setattr(_record_engine, "move_to_home_smooth", fail_and_stop)
    monkeypatch.setattr(_record_engine, "log_say", lambda *_args: None)

    result = _record_engine.reset_then_finalize_episode(
        robot=FakeRobot(),
        events=events,
        reset_before_episode=True,
        home_action={"joint": 0.0},
        fps=30,
        home_duration_s=2.0,
        play_sounds=False,
        episode_label="Episode 1",
        finalize=lambda: finalized.append(True),
    )

    assert result is False
    assert finalized == []


def test_capture_home_uses_first_post_connect_observation_once():
    class StartupRobot:
        JOINT_NAMES = ["main_joint1"]
        GRIPPER_NAME = "main_gripper"

        def __init__(self):
            self._arms = {"left": object()}
            self._home_joints = {"left": [0.25]}
            self.config = SimpleNamespace(
                home_gripper=1.0,
                home_joints={"left_main_joint1": 99.0},
            )
            self.observation_count = 0

        def get_observation(self):
            self.observation_count += 1
            return {"left_main_joint1": 0.25 * self.observation_count}

    robot = StartupRobot()

    home = _record_engine.capture_home_action(robot)

    assert home == {"left_main_joint1": 0.25, "left_main_gripper": 1.0}
    assert robot.observation_count == 0
