"""Portable random-state snapshots for checkpointing and isolated validation."""

import random
from contextlib import contextmanager

import numpy as np
import torch


def capture_rng(device):
    device = torch.device(device)
    numpy_state = np.random.get_state()
    return dict(
        torch_rng=torch.get_rng_state(),
        python_rng=random.getstate(),
        numpy_rng=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


def restore_rng(state, device):
    device = torch.device(device)
    if state.get("cuda_rng") is not None and device.type != "cuda":
        raise ValueError("A CUDA training RNG checkpoint requires a CUDA resume device")
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if state.get("numpy_rng") is not None:
        name, keys, position, gaussian, cached = state["numpy_rng"]
        np.random.set_state(
            (name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached)
        )
    if state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], device)


@contextmanager
def preserve_rng(device):
    state = capture_rng(device)
    try:
        yield
    finally:
        restore_rng(state, device)
