"""Build an explicitly synthetic smoke policy with a matching fine-tuned encoder."""

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoProcessor

from tacmind0.tacdream import TacDreamDataConfig, TacDreamModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-model", type=Path, required=True)
    parser.add_argument("--jsonl-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(19)
    cfg = TacDreamModelConfig(
        tactile_weights=str(args.world_model / "encoder.pt"),
        backbone_config=str(args.world_model / "backbone_config.json"),
        require_finetuned_tactile=True,
    )
    policy = cfg.build_model().cuda().eval()
    # This fixture trains only FiLM for one step; never label it a formal policy.
    policy.requires_grad_(False)
    policy.tactile_film.requires_grad_(True)
    processor = AutoProcessor.from_pretrained(
        cfg.model_name_or_path, local_files_only=True
    )
    dataset, collator = TacDreamDataConfig(
        jsonl_dir=args.jsonl_dir, image_dir=args.image_dir
    ).build_dataset(processor, cfg.chunk_size)
    batch = {k: v.cuda() for k, v in collator([dataset[0]]).items()}
    optimizer = torch.optim.AdamW(policy.tactile_film.parameters(), lr=1e-4)
    loss = policy(**batch).loss
    loss.backward()
    optimizer.step()
    policy.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    shutil.copyfile(
        Path(cfg.model_name_or_path) / "norm_stats.json",
        args.output_dir / "norm_stats.json",
    )
    (args.output_dir / "SMOKE_ONLY.txt").write_text(
        f"One FiLM step; not a formal trained policy. Loss {float(loss.detach())}\n"
    )
    print(float(loss.detach()), flush=True)


if __name__ == "__main__":
    main()
