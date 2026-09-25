#!/usr/bin/env bash
set -euo pipefail

dataset_mixture=${1:?"Usage: bash scripts/process_backbone_data.sh <registered_name|source/group/dataset_id> [extra args...]"}
extra_args=("${@:2}")

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
cd "${repo_root}"

exec python -m vtla.tac_encoder.process_backbone_data \
  --dataset_selection "${dataset_mixture}" \
  "${extra_args[@]}"
