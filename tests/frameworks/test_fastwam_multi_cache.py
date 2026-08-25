import json

import pytest
import torch
from safetensors.torch import save_file

from vtla.frameworks.fastwam.processor_fastwam import FastWAMPrepareBatchStep


PROMPT = "prompt: {task}"


def _write_cache(path, tasks, model_hash="same-hash"):
    path.mkdir(parents=True)
    tensors = {}
    entries = []
    for slot, (task, value) in enumerate(tasks):
        tensors[f"context.{slot}"] = torch.full((2, 3), value, dtype=torch.bfloat16)
        tensors[f"mask.{slot}"] = torch.ones(2, dtype=torch.bool)
        entries.append({"task": task, "task_index": slot, "slot": slot})
    save_file(tensors, str(path / "embeddings.safetensors"))
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "world_model": "wan22",
                "context_length": 2,
                "embedding_dim": 3,
                "prompt_template": PROMPT,
                "text_encoder_model_hash": model_hash,
                "tasks": entries,
            }
        ),
        encoding="utf-8",
    )


def _make_step(cache_dirs, **kwargs):
    return FastWAMPrepareBatchStep(
        camera_keys=["camera"],
        frame_indices=[0],
        video_size=(4, 4),
        context_len=2,
        text_dim=3,
        prompt_template=PROMPT,
        text_embedding_cache_dirs=[str(path) for path in cache_dirs],
        **kwargs,
    )


def test_fastwam_merges_multiple_cache_directories_and_saved_state_is_self_contained(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_cache(first, [("pick", 1.0)])
    _write_cache(second, [("place", 2.0), ("pick", 1.0)])

    step = _make_step([first, second])

    assert step.task_to_slot == {"pick": 0, "place": 1}
    assert set(step.state_dict()) == {"context.0", "mask.0", "context.1", "mask.1"}

    saved_config = step.get_config()
    saved_config["text_embedding_cache_dirs"] = [str(tmp_path / "missing")]
    restored = FastWAMPrepareBatchStep(**saved_config)
    restored.load_state_dict(step.state_dict())
    assert restored.task_to_slot == step.task_to_slot
    assert torch.equal(restored.state_dict()["context.1"], step.state_dict()["context.1"])


def test_fastwam_rejects_conflicting_duplicate_tasks(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_cache(first, [("pick", 1.0)])
    _write_cache(second, [("pick", 2.0)])

    with pytest.raises(ValueError, match="conflicting embeddings"):
        _make_step([first, second])


def test_fastwam_rejects_mixed_text_encoders(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_cache(first, [("pick", 1.0)], model_hash="first")
    _write_cache(second, [("place", 2.0)], model_hash="second")

    with pytest.raises(ValueError, match="different text encoder model hashes"):
        _make_step([first, second])
