#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/process_backbone_data.sh <dataset_id> [extra args...]" >&2
  exit 2
fi

dataset_id="$1"
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
cd "${repo_root}"

exec python -m vtla.tac_encoder.process_backbone_data \
  --dataset_id "${dataset_id}" \
  "$@"
