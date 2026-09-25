#!/usr/bin/env bash
# Merge concrete LeRobot datasets into a new source/division/dataset directory.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 5 ]]; then
  echo "Usage: bash scripts/merge_datasets.sh <out_source> <out_division> <out_id> <source/division/id> <source/division/id> [...]" >&2
  exit 2
fi

out_source=$1
out_division=$2
out_id=$3
shift 3
for component in "${out_source}" "${out_division}" "${out_id}"; do
  if [[ -z "${component}" || "${component}" == "." || "${component}" == ".." || "${component}" == */* ]]; then
    echo "Invalid output directory name: ${component}" >&2
    exit 2
  fi
done
catalog=playground/data
out="${catalog}/${out_source}/${out_division}/${out_id}"
if [[ -e "${out}" ]]; then
  echo "Output already exists: ${out}" >&2
  exit 1
fi

roots=()
for member in "$@"; do
  if [[ "${member}" != */*/* || "${member}" == */*/*/* ]]; then
    echo "Expected source/division/id, got: ${member}" >&2
    exit 2
  fi
  IFS=/ read -r member_source member_division member_id <<< "${member}"
  for component in "${member_source}" "${member_division}" "${member_id}"; do
    if [[ -z "${component}" || "${component}" == "." || "${component}" == ".." ]]; then
      echo "Invalid dataset reference: ${member}" >&2
      exit 2
    fi
  done
  root="${catalog}/${member}"
  if [[ ! -f "${root}/meta/info.json" ]]; then
    echo "Dataset metadata not found: ${root}/meta/info.json" >&2
    exit 1
  fi
  roots+=("${root}")
done

mkdir -p "${catalog}/${out_source}/${out_division}"
PYTHONPATH="$(pwd):${PYTHONPATH:-}" python tools/merge_datasets.py \
  --roots "${roots[@]}" --out "${out}" --repo-id "${out_id}"

echo "Merged dataset: ${out}"
