"""Checkpoint identity shared by fine-tuning and policy initialization."""

import hashlib
from pathlib import Path

import torch

V6_EPOCH1_SHA256 = "85065ebad91912bf7dfd02fdcf16ead2ae76800d812c69c8b8415d257e9cc5e9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_initialization(provenance: dict) -> None:
    step = provenance.get("step")
    if (
        provenance.get("stage") != "taclewm_finetuned"
        or type(step) is not int
        or step < 1
        or provenance.get("initial_weights_sha256") != V6_EPOCH1_SHA256
    ):
        raise ValueError(
            "Expected positive-step fine-tuning initialized from v6 epoch 1"
        )


def fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    tensors = [
        ("parameter:" + name, value) for name, value in module.named_parameters()
    ]
    tensors += [("buffer:" + name, value) for name, value in module.named_buffers()]
    for name, tensor in sorted(tensors):
        digest.update(name.encode())
        digest.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()
