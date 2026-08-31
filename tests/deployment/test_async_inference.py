from types import SimpleNamespace

import multiprocessing as mp
import numpy as np
import pytest
import torch

from deployment import _async_inference, inference
from deployment._async_inference import (
    ActionChunkSnapshot,
    AsyncChunkScheduler,
    SharedActionChunkSlot,
    SharedObservationBuffer,
    _begin_episode,
    _end_episode,
)
from deployment._record_engine import DatasetRecordConfig
from deployment.inference import InferenceConfig
from deployment.robots import RmIsfUmiLeftConfig
from vtla.engine.common import control_utils
from vtla.engine.processor.relative_action_processor import RelativeActionsProcessorStep


def _publish_observation(buffer):
    buffer.publish(
        {
            "joint": 1.25,
            "cam": np.full((2, 3, 3), 7, dtype=np.uint8),
            "cam_finger0": np.full((2, 2, 3), 4095, dtype=np.uint16),
        }
    )


def _read_observation(buffer, result_queue):
    snapshot = buffer.read()
    result_queue.put((snapshot.version, snapshot.values["joint"]))


def _chunk(chunk_id, generation, value):
    return ActionChunkSnapshot(
        chunk_id=chunk_id,
        generation=generation,
        observation_version=chunk_id * 10,
        inference_started_at=1.0,
        inference_finished_at=1.1,
        actions=np.full((3, 2), value, dtype=np.float32),
    )


def test_inference_mode_dispatch_preserves_sync_runner(monkeypatch):
    cfg = SimpleNamespace(inference_mode="sync")
    monkeypatch.setattr(inference, "run_record", lambda received: ("sync", received))

    assert inference._run_inference_mode(cfg) == ("sync", cfg)


def test_inference_mode_dispatch_uses_async_runner(monkeypatch):
    cfg = SimpleNamespace(inference_mode="async")
    monkeypatch.setattr(
        _async_inference,
        "run_async_record",
        lambda received: ("async", received),
    )

    assert inference._run_inference_mode(cfg) == ("async", cfg)


def test_inference_mode_config_normalizes_and_validates():
    cfg = InferenceConfig(
        robot=RmIsfUmiLeftConfig(),
        dataset=DatasetRecordConfig(repo_id="local/eval_async"),
        inference_mode="ASYNC",
    )
    assert cfg.inference_mode == "async"

    with pytest.raises(ValueError, match="sync.*async"):
        InferenceConfig(
            robot=RmIsfUmiLeftConfig(),
            dataset=DatasetRecordConfig(repo_id="local/eval_invalid"),
            inference_mode="parallel",
        )


def test_shared_observation_is_coherent_across_spawn_process():
    ctx = mp.get_context("spawn")
    buffer = SharedObservationBuffer.create(
        {
            "joint": float,
            "cam": (2, 3, 3),
            "cam_finger0": (2, 2, 3),
        },
        ctx,
    )
    process = ctx.Process(target=_publish_observation, args=(buffer,))
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    snapshot = buffer.read()
    assert snapshot is not None
    assert snapshot.version == 1
    assert snapshot.values["joint"] == 1.25
    assert snapshot.values["cam"].dtype == np.uint8
    assert snapshot.values["cam_finger0"].dtype == np.uint16
    np.testing.assert_array_equal(snapshot.values["cam"], 7)
    np.testing.assert_array_equal(snapshot.values["cam_finger0"], 4095)

    # The same shared object is passed to the independently spawned inference
    # process after the hardware process has already used it.
    result_queue = ctx.Queue()
    reader = ctx.Process(target=_read_observation, args=(buffer, result_queue))
    reader.start()
    reader.join(timeout=10)
    assert reader.exitcode == 0
    assert result_queue.get(timeout=1) == (1, 1.25)
    result_queue.close()


def test_shared_observation_history_retains_latest_control_frames():
    ctx = mp.get_context("spawn")
    buffer = SharedObservationBuffer.create({"joint": float}, ctx, history_size=3)
    for value in range(1, 6):
        buffer.publish({"joint": float(value)})

    retained = buffer.read_since(after_version=0)
    assert [snapshot.version for snapshot in retained] == [3, 4, 5]
    assert [snapshot.values["joint"] for snapshot in retained] == [3.0, 4.0, 5.0]
    assert buffer.read_since(after_version=5) == []


def test_active_chunk_is_not_replaced_and_next_claim_uses_latest():
    scheduler = AsyncChunkScheduler(["a", "b"])
    assert scheduler.accept(_chunk(1, 4, 1.0), generation=4)

    assert scheduler.pop() == {"a": 1.0, "b": 1.0}
    assert not scheduler.accept(_chunk(2, 4, 2.0), generation=4)
    assert scheduler.pop() == {"a": 1.0, "b": 1.0}
    assert scheduler.pop() == {"a": 1.0, "b": 1.0}

    assert scheduler.accept(_chunk(3, 4, 3.0), generation=4)
    assert scheduler.last_claimed_chunk_id == 3
    assert scheduler.pop() == {"a": 3.0, "b": 3.0}


def test_scheduler_rejects_stale_generation_and_duplicate_chunk():
    scheduler = AsyncChunkScheduler(["a", "b"])
    assert not scheduler.accept(_chunk(1, 1, 1.0), generation=2)
    assert scheduler.accept(_chunk(2, 2, 2.0), generation=2)
    while scheduler.pop() is not None:
        pass
    assert not scheduler.accept(_chunk(2, 2, 2.0), generation=2)


def test_latest_chunk_slot_overwrites_unclaimed_result_and_reset_invalidates_it():
    ctx = mp.get_context("spawn")
    slot = SharedActionChunkSlot.create((3, 2), ctx)
    slot.publish(
        np.ones((3, 2), dtype=np.float32),
        generation=5,
        observation_version=10,
        inference_started_at=1.0,
        inference_finished_at=1.1,
    )
    newest_id = slot.publish(
        np.full((3, 2), 2.0, dtype=np.float32),
        generation=5,
        observation_version=11,
        inference_started_at=1.2,
        inference_finished_at=1.3,
    )

    claimed = slot.claim(after_chunk_id=0, generation=5)
    assert claimed is not None
    assert claimed.chunk_id == newest_id
    np.testing.assert_array_equal(claimed.actions, 2.0)

    slot.clear(generation=6)
    assert slot.claim(after_chunk_id=0, generation=5) is None
    assert slot.claim(after_chunk_id=0, generation=6) is None


def test_episode_generation_barrier_clears_chunk_slot():
    ctx = mp.get_context("spawn")
    generation = ctx.Value("q", 0)
    enabled = ctx.Event()
    slot = SharedActionChunkSlot.create((2, 1), ctx)

    active_generation = _begin_episode(generation, slot, enabled)
    assert active_generation == 1
    assert enabled.is_set()
    slot.publish(
        np.ones((2, 1), dtype=np.float32),
        generation=active_generation,
        observation_version=1,
        inference_started_at=1.0,
        inference_finished_at=1.1,
    )

    inactive_generation = _end_episode(generation, slot, enabled)
    assert inactive_generation == 2
    assert not enabled.is_set()
    assert slot.claim(after_chunk_id=0, generation=active_generation) is None


def test_full_chunk_prediction_slices_once_and_uses_one_relative_anchor(monkeypatch):
    relative_step = RelativeActionsProcessorStep(enabled=True)

    class Preprocessor:
        steps = [relative_step]

        def __call__(self, observation):
            relative_step._last_state = torch.tensor([[0.5, 0.25]])
            return observation

    class Postprocessor:
        def __call__(self, actions):
            return actions + 100

    class Policy:
        config = SimpleNamespace(action_start_offset=2, n_action_steps=3)

        def predict_action_chunk(self, _observation):
            return torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)

    monkeypatch.setattr(
        control_utils,
        "prepare_observation_for_inference",
        lambda observation, *_args: observation,
    )

    actions = control_utils.predict_action_chunk(
        observation={"observation.state": np.array([0.5, 0.25])},
        policy=Policy(),
        device=torch.device("cpu"),
        preprocessor=Preprocessor(),
        postprocessor=Postprocessor(),
        use_amp=False,
    )

    expected = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)[:, 2:5] + 100
    torch.testing.assert_close(actions, expected)
    torch.testing.assert_close(relative_step._locked_state, torch.tensor([[0.5, 0.25]]))
