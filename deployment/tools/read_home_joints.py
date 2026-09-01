#!/usr/bin/env python
"""
读取双臂当前关节位置并输出 JSON，供硬件姿态诊断。

用法:
    python -m deployment.tools.read_home_joints
    python -m deployment.tools.read_home_joints --left-ip 192.168.1.201 --right-ip 192.168.1.200

输出示例:
    {"left_main_joint1": -0.091996, ..., "right_main_joint7": 0.111946}
"""

import argparse
import json
import sys
import time

import numpy as np

from deployment.hardware.follower_arms.realman_tcp import RealmanTcpFollower

JOINT_NAMES = [f"main_joint{i}" for i in range(1, 8)]


def read_side(ip: str, port: int, side: str) -> dict[str, float]:
    arm = RealmanTcpFollower(ip=ip, port=port, use_degrees=False, name=f"{side}_follower")
    try:
        arm.connect()
        # 等后台线程刷一帧
        time.sleep(0.15)
        joints = arm.read_joints_now()
        if joints is None:
            joints = arm.read_joints()
        return {f"{side}_{name}": round(float(v), 6) for name, v in zip(JOINT_NAMES, joints)}
    finally:
        arm.disconnect()


def main():
    ap = argparse.ArgumentParser(description="读取双臂当前关节位置 JSON")
    ap.add_argument("--left-ip",  default="192.168.1.201")
    ap.add_argument("--right-ip", default="192.168.1.200")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--side", choices=["left", "right", "both"], default="both")
    args = ap.parse_args()

    result: dict[str, float] = {}
    sides = (["left", "right"] if args.side == "both"
             else [args.side])
    ip_map = {"left": args.left_ip, "right": args.right_ip}

    for side in sides:
        print(f"[{side}] 连接 {ip_map[side]}:{args.port} ...", file=sys.stderr)
        result.update(read_side(ip_map[side], args.port, side))

    json_str = json.dumps(result, separators=(", ", ": "))
    print(json_str)

    # 同时在 stderr 打印每个关节方便核对
    print("\n当前关节 (弧度):", file=sys.stderr)
    for k, v in result.items():
        print(f"  {k}: {v:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
