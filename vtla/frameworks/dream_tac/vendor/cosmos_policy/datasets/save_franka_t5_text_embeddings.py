# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Precomputes T5 text embeddings for Franka/tactile task descriptions.

Reads unique task names from preprocessing_metadata.json and saves embeddings to t5_embeddings.pkl.

Usage:
    uv run -m cosmos_policy.datasets.save_franka_t5_text_embeddings --data_dir /path/to/preprocessed/task_folder

Example:
    uv run --extra cu128 --group franka -m cosmos_policy.datasets.save_franka_t5_text_embeddings --data_dir /path/to/preprocessed_data
"""

import argparse
import json
import os

from cosmos_policy.datasets.t5_embedding_utils import (
    generate_t5_embeddings,
    save_embeddings,
)


def main():
    parser = argparse.ArgumentParser(description="Precompute T5 text embeddings for Franka task descriptions")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Preprocessed dataset dir (contains preprocessing_metadata.json, train/, val/)",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.data_dir, "preprocessing_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"preprocessing_metadata.json not found in {args.data_dir}. "
            "Run preprocess_tactile_franka_data.py first."
        )

    with open(metadata_path) as f:
        metadata = json.load(f)
    task_name = metadata["dataset_info"]["task_name"]
    unique_commands = [task_name]
    print(f"Task name(s): {unique_commands}")

    t5_text_embeddings = generate_t5_embeddings(unique_commands)
    save_path = save_embeddings(t5_text_embeddings, args.data_dir)
    print(f"Done. Add to experiment config: t5_text_embeddings_path=\"{save_path}\"")


if __name__ == "__main__":
    main()
