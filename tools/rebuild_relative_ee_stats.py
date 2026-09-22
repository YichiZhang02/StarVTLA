#!/usr/bin/env python3
"""Rebuild global/episode relative stats for an already migrated TCP dataset."""
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from vtla.datasets.tcp_contract import build_tcp_contract, validate_tcp_contract
from vtla.datasets.tcp_stats import collect_relative_stats, prepare_episode_stats, RELATIVE


def rebuild(root: Path, horizon: int, action_gap: int):
    info_path = root / 'meta/info.json'
    stats_path = root / 'meta/stats.json'
    info = json.loads(info_path.read_text())
    validate_tcp_contract(info.get('tcp_contract'), robot_type=info['robot_type'])
    contract = build_tcp_contract(info['robot_type'], horizon, action_gap)
    paths = sorted((root / 'data').rglob('*.parquet'))
    global_stats, episodes = collect_relative_stats(paths, contract['stats_offsets'])
    stats = json.loads(stats_path.read_text())
    stats[RELATIVE] = global_stats
    info['tcp_contract'] = contract
    prepared = prepare_episode_stats(root, paths, episodes)
    try:
        for path, value in ((stats_path, stats), (info_path, info)):
            temporary = path.with_name('.' + path.name + '.tcp-stats.tmp')
            if temporary.exists():
                raise FileExistsError(temporary)
            prepared.append((temporary, path))
            temporary.write_text(json.dumps(value, indent=4) + '\n')
        for temporary, path in prepared:
            temporary.replace(path)
    finally:
        for temporary, _ in prepared:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--horizon', type=int, required=True)
    offsets = parser.add_mutually_exclusive_group(required=True)
    offsets.add_argument('--action-gap', type=int, dest='offset_start', help='First statistics offset')
    offsets.add_argument('--offset-start', type=int, dest='offset_start', help='Signed first offset, e.g. -1 for Diffusion')
    args = parser.parse_args()
    rebuild(args.root, args.horizon, args.offset_start)


if __name__ == '__main__':
    main()
