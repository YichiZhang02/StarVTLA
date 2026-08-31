#!/usr/bin/env python
"""Precompute dataset-local text embeddings for supported world models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vtla.frameworks.dream_tac.runtime import resolve_cosmos_text_assets
from vtla.frameworks.fastwam.core.helpers.io import hash_model_file
from vtla.frameworks.fastwam.core.helpers.loader import _load_registered_model
from vtla.frameworks.fastwam.core.wan_video_text_encoder import HuggingfaceTokenizer


WAN22_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def read_dataset_tasks(dataset_root: Path) -> list[dict]:
    path = dataset_root / "meta" / "tasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing LeRobot task metadata: {path}")
    table = pq.read_table(path)
    columns = table.to_pydict()
    task_column = "task" if "task" in columns else "__index_level_0__"
    if task_column not in columns:
        raise ValueError(f"Cannot find task text in {path}; columns={table.column_names}")
    indices = columns.get("task_index", list(range(len(columns[task_column]))))
    tasks = []
    seen = set()
    for task_index, task in zip(indices, columns[task_column], strict=True):
        task = str(task)
        if task not in seen:
            seen.add(task)
            tasks.append({"task_index": int(task_index), "task": task})
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--world-model", choices=["wan22", "dream_tac"], default="wan22")
    parser.add_argument(
        "--text-encoder-path",
        type=Path,
        default=Path(
            "playground/pretrained_models/Wan2.2-TI2V-5B/"
            "models_t5_umt5-xxl-enc-bf16.pth"
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(
            "playground/pretrained_models/Wan2.2-TI2V-5B/google/umt5-xxl"
        ),
    )
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument(
        "--pretrained-path",
        help=(
            "Cosmos Predict2 pretrained path used by Dream-Tac. Its text_encoder/ and "
            "tokenizer/ subdirectories are resolved automatically."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def encode_wan22(args: argparse.Namespace, tasks: list[dict], output_dir: Path) -> None:
    if not tasks:
        raise ValueError(f"Dataset contains no tasks: {args.dataset_root}")
    manifest_path = output_dir / "manifest.json"
    tensor_path = output_dir / "embeddings.safetensors"
    if not args.overwrite and manifest_path.is_file() and tensor_path.is_file():
        raise FileExistsError(f"Text cache already exists at {output_dir}; pass --overwrite to replace it.")

    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    encoder = _load_registered_model(
        str(args.text_encoder_path.expanduser().resolve()),
        "wan_video_text_encoder",
        torch_dtype=dtype,
        device=args.device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=str(args.tokenizer_path.expanduser().resolve()),
        seq_len=args.context_length,
        clean="whitespace",
    )

    tensors: dict[str, torch.Tensor] = {}
    manifest_tasks = []
    with torch.inference_mode():
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start : start + args.batch_size]
            prompts = [WAN22_PROMPT_TEMPLATE.format(task=item["task"]) for item in batch]
            ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
            mask = mask.to(device=args.device, dtype=torch.bool)
            context = encoder(ids.to(args.device), mask)
            for offset, item in enumerate(batch):
                slot = start + offset
                tensors[f"context.{slot}"] = context[offset].to(
                    device="cpu", dtype=torch.bfloat16
                ).contiguous()
                tensors[f"mask.{slot}"] = mask[offset].to(device="cpu").contiguous()
                manifest_tasks.append({**item, "slot": slot})

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tensor_path))
    manifest = {
        "format_version": 1,
        "world_model": "wan22",
        "text_encoder": "umt5-xxl",
        "text_encoder_path": str(args.text_encoder_path.expanduser().resolve()),
        "text_encoder_model_hash": hash_model_file(
            str(args.text_encoder_path.expanduser().resolve())
        ),
        "context_length": args.context_length,
        "embedding_dim": int(
            next(value for key, value in tensors.items() if key.startswith("context.")).shape[1]
        ),
        "prompt_template": WAN22_PROMPT_TEMPLATE,
        "tasks": manifest_tasks,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved {len(tasks)} task embeddings to {output_dir}")


def _model_fingerprint(path_or_id: str) -> str:
    path = Path(path_or_id).expanduser()
    if not path.exists():
        return hashlib.sha256(path_or_id.encode()).hexdigest()
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix in {".json", ".model", ".txt"}
    )
    for item in files:
        stat = item.stat()
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        digest.update(str(stat.st_size).encode())
        with item.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def encode_dream_tac(args: argparse.Namespace, tasks: list[dict], output_dir: Path) -> None:
    from transformers import T5EncoderModel, T5TokenizerFast

    if not tasks:
        raise ValueError(f"Dataset contains no tasks: {args.dataset_root}")
    manifest_path = output_dir / "manifest.json"
    tensor_path = output_dir / "embeddings.safetensors"
    if not args.overwrite and manifest_path.is_file() and tensor_path.is_file():
        raise FileExistsError(f"Dream-Tac cache already exists at {output_dir}; pass --overwrite.")
    if not args.pretrained_path:
        raise ValueError("Dream-Tac text precomputation requires --pretrained-path.")
    context_length = 512
    encoder_path, tokenizer_path = resolve_cosmos_text_assets(str(args.pretrained_path))
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
    encoder = (
        T5EncoderModel.from_pretrained(encoder_path, local_files_only=True)
        .to(args.device)
        .eval()
    )
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    encoder.to(dtype=dtype)
    tensors: dict[str, torch.Tensor] = {}
    manifest_tasks = []
    with torch.inference_mode():
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start : start + args.batch_size]
            prompts = [str(item["task"]) for item in batch]
            tokens = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=context_length,
            ).to(args.device)
            context = encoder(
                input_ids=tokens.input_ids, attention_mask=tokens.attention_mask
            ).last_hidden_state
            context = context.masked_fill(~tokens.attention_mask.bool().unsqueeze(-1), 0)
            for offset, item in enumerate(batch):
                slot = start + offset
                tensors[f"context.{slot}"] = (
                    context[offset].to(device="cpu", dtype=torch.bfloat16).contiguous()
                )
                tensors[f"mask.{slot}"] = tokens.attention_mask[offset].bool().cpu().contiguous()
                manifest_tasks.append({**item, "slot": slot})
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tensor_path))
    manifest = {
        "format_version": 1,
        "world_model": "dream_tac",
        "text_encoder": "t5-11b",
        "pretrained_path": str(args.pretrained_path),
        "text_encoder_path": str(encoder_path),
        "tokenizer_path": str(tokenizer_path),
        "text_encoder_model_hash": _model_fingerprint(str(encoder_path)),
        "context_length": context_length,
        "embedding_dim": int(
            next(v for k, v in tensors.items() if k.startswith("context.")).shape[1]
        ),
        "prompt_template": "{task}",
        "tasks": manifest_tasks,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved {len(tasks)} Dream-Tac task embeddings to {output_dir}")


def main() -> None:
    args = parse_args()
    tasks = read_dataset_tasks(args.dataset_root)
    output_dir = args.dataset_root / "text_embeddings" / args.world_model
    if args.world_model == "wan22":
        encode_wan22(args, tasks, output_dir)
    elif args.world_model == "dream_tac":
        encode_dream_tac(args, tasks, output_dir)


if __name__ == "__main__":
    main()
