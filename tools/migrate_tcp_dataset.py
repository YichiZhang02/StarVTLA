#!/usr/bin/env python3
"""Copy an existing dataset and rebuild TCP rot6d columns/statistics from raw data.

The source is never modified. --dry-run validates source metadata without copying.
The destination is published only after conversion and contract validation succeed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from vtla.datasets.tcp_contract import validate_tcp_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--dst', type=Path, required=True)
    parser.add_argument('--horizon', type=int, required=True)
    parser.add_argument('--action-gap', type=int, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    if src == dst or src in dst.parents or dst.exists():
        parser.error('Use a new destination outside the source dataset; existing paths are never overwritten.')
    if args.horizon < 1 or args.action_gap < 0:
        parser.error('horizon must be positive and action-gap nonnegative')
    info = json.loads((src / 'meta/info.json').read_text())
    robot_type = info.get('robot_type')
    if robot_type == 'umi':
        from tools.convert_umi_to_eepose import pose_indices
        for key in ('observation.state', 'action'):
            pose_indices(info['features'][key]['names'])
        converter = 'convert_umi_to_eepose.py'
    else:
        from deployment.robots import RobotConfig
        from vtla.engine.utils.ee_kinematics import joint_indices
        indices = joint_indices(info['features']['observation.state']['names'])
        RobotConfig.validate_kinematics_sides(robot_type, indices['sides'])
        for side in indices['sides']:
            RobotConfig.get_flange_tcp_calibration(robot_type, side)
        converter = 'convert_joints_to_eepose.py'
    if not list((src / 'data').rglob('*.parquet')):
        parser.error('No data parquet files found')
    print(f'{src} -> {dst}: {converter}, TCP rot6d, offsets={args.action_gap}..{args.action_gap + args.horizon - 1}')
    if args.dry_run:
        return
    stage = dst.with_name(dst.name + '.tcp-migration-partial')
    if stage.exists():
        parser.error(f'Partial output already exists: {stage}; inspect it before retrying.')
    shutil.copytree(src, stage)
    subprocess.run([sys.executable, str(REPO_ROOT / 'tools' / converter), '--root', str(stage),
                    '--horizon', str(args.horizon), '--action-gap', str(args.action_gap),
                    '--preserve-ee-grippers'], check=True)
    migrated = json.loads((stage / 'meta/info.json').read_text())
    validate_tcp_contract(migrated.get('tcp_contract'), robot_type=robot_type,
                          offsets=range(args.action_gap, args.action_gap + args.horizon))
    stage.rename(dst)
    print(f'Migrated dataset ready: {dst}')


if __name__ == '__main__':
    main()
