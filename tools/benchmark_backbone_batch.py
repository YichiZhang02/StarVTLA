"""Benchmark a real tactile training recipe on a cached dataset batch."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from vtla.tac_encoder.data.npy_tactile_dataset import (
    WeightedMixtureSampler,
    build_training_dataset,
    resolve_tactile_dataset,
)
from vtla.tac_encoder.training import get_training_recipe


PRETRAINED_PATHS = {
    "anytouch1": Path("playground/pretrained_models/AnyTouch-ViT-L-16/checkpoint.pth"),
    "anytouch2": Path("playground/pretrained_models/AnyTouch2-Model/checkpoint-4frames.pth"),
    "sparsh_vjepa": Path(
        "playground/pretrained_models/Sparsh-VJEPA-Small/vjepa_vitsmall_full.ckpt"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", choices=tuple(PRETRAINED_PATHS))
    parser.add_argument("batch_size", type=int, choices=(32, 64, 128, 256))
    parser.add_argument("--dataset_id", default="backbone_training_data")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dataset_catalog_root", type=Path, default=Path("playground/data"))
    parser.add_argument("--mixture_config", type=Path, default=Path("configs/data_mixtures.yaml"))
    parser.add_argument("--anytouch1_arch", default="vit_l")
    parser.add_argument("--encoder_dim", type=int)
    parser.add_argument("--encoder_depth", type=int)
    parser.add_argument("--encoder_heads", type=int)
    parser.add_argument("--projection_dim", type=int)
    parser.add_argument("--decoder_dim", type=int)
    parser.add_argument("--decoder_depth", type=int)
    parser.add_argument("--decoder_heads", type=int)
    args = parser.parse_args()
    args.pretrained_path = str(PRETRAINED_PATHS[args.model_id])
    return args


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    resolved = resolve_tactile_dataset(
        args.dataset_id,
        dataset_catalog_root=args.dataset_catalog_root,
        mixture_config=args.mixture_config,
    )
    dataset = build_training_dataset(resolved, split="train")
    sampler = WeightedMixtureSampler(dataset, seed=0, rank=rank, world_size=world_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    recipe = get_training_recipe(args.model_id)
    model = recipe.build_model(args).to(device).train()
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = recipe.optimizer(model, args)
    scheduler = recipe.scheduler(optimizer, args, max(1, len(loader)))

    iterator = iter(loader)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if world_size > 1:
        dist.barrier()
    started = time.perf_counter()
    losses = []
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images = batch["images"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = recipe.step(model, images, args)
        output.loss.backward()
        optimizer.step()
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        recipe.after_optimizer_step(unwrapped, step + 1, args.steps)
        scheduler.step()
        losses.append(float(output.loss.detach()))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    peak_allocated = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
    peak_reserved = torch.tensor(torch.cuda.max_memory_reserved(device), device=device)
    if world_size > 1:
        dist.all_reduce(peak_allocated, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak_reserved, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "dataset_id": args.dataset_id,
                    "batch_size_per_gpu": args.batch_size,
                    "world_size": world_size,
                    "steps": args.steps,
                    "seconds_per_step": elapsed / args.steps,
                    "global_samples_per_second": (
                        args.batch_size * world_size * args.steps / elapsed
                    ),
                    "peak_allocated_gib": peak_allocated.item() / 1024**3,
                    "peak_reserved_gib": peak_reserved.item() / 1024**3,
                    "losses": losses,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
