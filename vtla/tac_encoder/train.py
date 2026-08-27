"""Unified runtime dispatching to backbone-specific training recipes."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .config import SUPPORTED_MODEL_IDS
from .data.npy_tactile_dataset import (
    WeightedMixtureSampler,
    build_training_dataset,
    resolve_tactile_dataset,
    save_resolved_definition,
)
from .eval import select_visualization_indices
from .training import get_training_recipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--model_id", required=True, choices=SUPPORTED_MODEL_IDS)
    parser.add_argument(
        "--cache_root",
        type=Path,
        help="Optional centralized cache root; defaults to <dataset_root>/tactile_backbone_cache.",
    )
    parser.add_argument("--dataset_catalog_root", type=Path, default=Path("playground/data"))
    parser.add_argument("--mixture_config", type=Path, default=Path("configs/data_mixtures.yaml"))
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--pretrained_path", default="")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--anytouch1_arch", choices=["vit_b", "vit_l"], default="vit_l")
    parser.add_argument("--encoder_dim", type=int)
    parser.add_argument("--encoder_depth", type=int)
    parser.add_argument("--encoder_heads", type=int)
    parser.add_argument("--projection_dim", type=int)
    parser.add_argument("--decoder_dim", type=int)
    parser.add_argument("--decoder_depth", type=int)
    parser.add_argument("--decoder_heads", type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--vis_per_level", type=int, default=1)
    parser.add_argument("--amp_dtype", choices=["none", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank


def _atomic_checkpoint(state: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def _checkpoint_state(
    model, recipe, optimizer, scheduler, scaler, epoch, best_loss, args, resolved
) -> dict:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "format_version": 2,
        "objective": recipe.objective,
        "encoder": recipe.encoder_state_dict(unwrapped),
        "trainer": recipe.trainer_state_dict(unwrapped),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "model_id": args.model_id,
        "args": vars(args),
        "resolved_data_mixture": resolved.to_dict(),
        "resolved_signature": resolved.signature,
        "load_report": getattr(unwrapped, "load_report", None),
    }


@torch.no_grad()
def _evaluate_recipe(recipe, model, loader, device, args, autocast_dtype) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    member_loss: dict[int, float] = {}
    member_count: dict[int, int] = {}
    count = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        members = batch["member_index"].to(device)
        for member in members.unique(sorted=True):
            selected = images[members == member]
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                output = recipe.step(model, selected, args)
            sample_count = len(selected)
            for key, value in output.metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * sample_count
            loss = float(output.loss.detach())
            index = int(member.item())
            member_loss[index] = member_loss.get(index, 0.0) + loss * sample_count
            member_count[index] = member_count.get(index, 0) + sample_count
            count += sample_count
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    metrics["loss"] = metrics.get("loss", sum(member_loss.values()) / max(count, 1))
    metrics["samples"] = count
    metrics["member_loss"] = {
        str(index): member_loss[index] / member_count[index] for index in sorted(member_count)
    }
    return metrics


def main() -> None:
    args = parse_args()
    recipe = get_training_recipe(args.model_id)
    rank, world_size, local_rank = _distributed()
    main_process = rank == 0
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device if args.device != "cuda" else "cpu")
    output_dir = args.output_dir or Path("playground/results/tactile_backbone") / args.dataset_id / args.model_id
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_tactile_dataset(
        args.dataset_id,
        cache_root=args.cache_root,
        dataset_catalog_root=args.dataset_catalog_root,
        mixture_config=args.mixture_config,
    )
    for member in resolved.members:
        cache = np.load(Path(member.cache_root) / "num_frames.npy", allow_pickle=False).item()
        stride = np.load(Path(member.cache_root) / "frame_stride.npy", allow_pickle=False).item()
        size = np.load(Path(member.cache_root) / "image_size.npy", allow_pickle=False).item()
        if (int(cache), int(stride), int(size)) != (args.num_frames, args.frame_stride, args.image_size):
            raise ValueError(
                f"Cache {member.cache_root} has T/stride/size={(cache, stride, size)}, "
                f"training requested {(args.num_frames, args.frame_stride, args.image_size)}"
            )
    if main_process:
        save_resolved_definition(resolved, output_dir / "resolved_data_mixture.json")

    train_dataset = build_training_dataset(resolved, split="train")
    train_sampler = WeightedMixtureSampler(
        train_dataset,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_dataset = None
    val_loader = None
    if main_process:
        try:
            val_dataset = build_training_dataset(resolved, split="val")
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(0, args.num_workers // 2),
                pin_memory=device.type == "cuda",
            )
        except ValueError as error:
            print(f"[validation disabled] {error}")

    model = recipe.build_model(args).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if main_process:
        print(
            f"model={args.model_id} objective={recipe.objective} "
            f"parameters={parameter_count:,}"
        )
    if world_size > 1:
        # Reconstruction intentionally leaves downstream-only projection/time parameters unused.
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    optimizer = recipe.optimizer(model, args)
    scheduler = recipe.scheduler(optimizer, args, len(train_loader))
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    if device.type != "cuda" and amp_dtype == torch.float16:
        amp_dtype = None
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    start_epoch = 0
    best_loss = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("model_id") != args.model_id:
            raise ValueError("Resume checkpoint model_id does not match")
        if checkpoint.get("resolved_signature") != resolved.signature:
            raise ValueError("Resolved mixture/cache signature changed since the checkpoint was saved")
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        if checkpoint.get("format_version") != 2:
            raise ValueError("Resume requires a format_version=2 recipe checkpoint")
        if checkpoint.get("objective") != recipe.objective:
            raise ValueError("Resume checkpoint objective does not match the selected recipe")
        recipe.restore_state(unwrapped, checkpoint["encoder"], checkpoint["trainer"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", math.inf))

    visualization_indices = []
    if main_process and val_dataset is not None:
        indices_path = output_dir / "visualization_indices.json"
        if indices_path.is_file():
            visualization_indices = json.loads(indices_path.read_text())["indices"]
        else:
            visualization_indices = select_visualization_indices(val_dataset, args.vis_per_level)
            indices_path.write_text(json.dumps({"indices": visualization_indices}, indent=2) + "\n")

    log_path = output_dir / "train_log.jsonl"
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        seen = 0
        metric_totals: dict[str, float] = {}
        sampled_members = np.zeros(len(resolved.members), dtype=np.int64)
        for batch_index, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                output = recipe.step(model, images, args)
            scaler.scale(output.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
            recipe.after_optimizer_step(
                unwrapped,
                epoch * len(train_loader) + batch_index + 1,
                args.epochs * len(train_loader),
            )
            scheduler.step()
            running += float(output.loss.detach()) * len(images)
            for key, value in output.metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + float(value.detach()) * len(images)
            seen += len(images)
            sampled_members += np.bincount(
                batch["member_index"].numpy(), minlength=len(resolved.members)
            )
        if world_size > 1:
            stats = torch.tensor([running, seen], dtype=torch.float64, device=device)
            dist.all_reduce(stats)
            running, seen = stats.tolist()
            sampled_tensor = torch.tensor(sampled_members, dtype=torch.int64, device=device)
            dist.all_reduce(sampled_tensor)
            sampled_members = sampled_tensor.cpu().numpy()
            keys = sorted(metric_totals)
            if keys:
                metric_tensor = torch.tensor(
                    [metric_totals[key] for key in keys], dtype=torch.float64, device=device
                )
                dist.all_reduce(metric_tensor)
                metric_totals = dict(zip(keys, metric_tensor.cpu().tolist()))
        record = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "train_metrics": {
                key: value / max(seen, 1) for key, value in sorted(metric_totals.items())
            },
            "lr": scheduler.get_last_lr()[0],
            "sampled_members": sampled_members.tolist(),
        }
        if world_size > 1:
            dist.barrier()
        should_eval = args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0
        if main_process and should_eval and val_loader is not None:
            unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
            metrics = _evaluate_recipe(recipe, unwrapped, val_loader, device, args, amp_dtype)
            record["validation"] = metrics
            recipe.save_visualization(
                unwrapped,
                val_dataset,
                visualization_indices,
                output_dir / "recon_vis" / f"recon_epoch{epoch + 1:04d}.png",
                device,
                args,
                amp_dtype,
            )
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                _atomic_checkpoint(
                    _checkpoint_state(
                        model, recipe, optimizer, scheduler, scaler, epoch, best_loss, args, resolved
                    ),
                    output_dir / "best.pth",
                )
        if main_process:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
                _atomic_checkpoint(
                    _checkpoint_state(
                        model, recipe, optimizer, scheduler, scaler, epoch, best_loss, args, resolved
                    ),
                    output_dir / f"checkpoint_epoch{epoch + 1:04d}.pth",
                )
            _atomic_checkpoint(
                _checkpoint_state(
                    model, recipe, optimizer, scheduler, scaler, epoch, best_loss, args, resolved
                ),
                output_dir / "last.pth",
            )
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
