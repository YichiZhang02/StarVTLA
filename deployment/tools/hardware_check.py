#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
睿尔曼 ugripper 硬件自检工具。分三个阶段, 从安全到危险:

  1. existence  存在性: ping 从臂/板子 IP, 检查主臂串口、顶相机 /dev/video* (不连任何硬件, 最安全)
  2. camera     图像:   实际连接顶相机/手腕鱼眼/触觉, 抓一帧, 报告 shape 和是否全黑 (--save 存图)
  3. teleop     主从同步: 连主臂+从臂, 读主臂位姿下发给从臂, 打印逐关节误差 (⚠️ 会让从臂运动!)

用法:
    python -m deployment.tools.hardware_check                      # 默认只跑 existence (安全)
    python -m deployment.tools.hardware_check --stage camera --save
    python -m deployment.tools.hardware_check --stage teleop --confirm-move --duration 5
    python -m deployment.tools.hardware_check --stage all --confirm-move

常用覆盖:
    --arms left,right         要检查的臂 (默认取 config)
    --left-port /dev/ttyRealmanBaseLeaderL  --right-port /dev/ttyRealmanBaseLeaderR
    --no-tactile              跳过触觉
"""

import argparse
import contextlib
import math
import os
import select
import subprocess
import sys
import termios
import time
import tty

from deployment.robots.rm_base_umi_dual import RmBaseUmiDualConfig
from deployment.robots.rm_isf_umi_left import RmIsfUmiLeftConfig
from deployment.robots.rm_isf_umi_right import RmIsfUmiRightConfig
from deployment.teleoperators.rm_leader_dual import RmLeaderDualConfig
from deployment.teleoperators.rm_leader_left import RmLeaderLeftConfig
from deployment.teleoperators.rm_leader_right import RmLeaderRightConfig

# ---------------- 终端着色 ----------------
_G, _R, _Y, _0 = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def ok(msg):
    print(f"  {_G}✓{_0} {msg}")


def fail(msg):
    print(f"  {_R}✗{_0} {msg}")


def warn(msg):
    print(f"  {_Y}!{_0} {msg}")


def header(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ---------------- 按臂取配置 ----------------
def follower_ip(cfg, side):
    if hasattr(cfg, "follower_ip"):
        return cfg.follower_ip
    return cfg.left_follower_ip if side == "left" else cfg.right_follower_ip


def board_ip(cfg, side):
    if hasattr(cfg, "board_ip"):
        return cfg.board_ip
    return cfg.left_board_ip if side == "left" else cfg.right_board_ip


def fisheye_udp_port(cfg, side):
    if hasattr(cfg, "fisheye_udp_port"):
        return cfg.fisheye_udp_port
    return cfg.left_fisheye_udp_port if side == "left" else cfg.right_fisheye_udp_port


def tactile_pc_ports(cfg, side):
    if hasattr(cfg, "tactile0_pc_port"):
        return cfg.tactile0_pc_port, cfg.tactile1_pc_port
    if side == "left":
        return cfg.left_tactile0_pc_port, cfg.left_tactile1_pc_port
    return cfg.right_tactile0_pc_port, cfg.right_tactile1_pc_port


def gripper_itinerary(cfg, side):
    if hasattr(cfg, "gripper_itinerary"):
        return cfg.gripper_itinerary
    return cfg.left_gripper_itinerary if side == "left" else cfg.right_gripper_itinerary


def leader_port(tcfg, side):
    if hasattr(tcfg, "port"):
        return tcfg.port
    return tcfg.left_port if side == "left" else tcfg.right_port


# ---------------- 通用探测 ----------------
def ping(ip: str, timeout_s: int = 1) -> bool:
    try:
        r = subprocess.run(
            ["ping", "-c", "1", "-W", str(timeout_s), ip],
            capture_output=True, timeout=timeout_s + 2,
        )
        return r.returncode == 0
    except Exception:
        return False


def frame_is_black(img) -> bool:
    import numpy as np
    return img is None or not np.any(img)


@contextlib.contextmanager
def _raw_stdin():
    """把终端切到 cbreak: 按键即时可读、无需回车; Ctrl-C 仍有效。非 tty 则空转。"""
    if not sys.stdin.isatty():
        yield False
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _poll_key():
    """非阻塞读一个键, 没有则返回 None。"""
    if not sys.stdin.isatty():
        return None
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _to_preview_uint8(frame):
    """Convert canonical uint16 tactile to uint8; keep processed uint8 unchanged."""
    import numpy as np

    if isinstance(frame, np.ndarray) and frame.dtype in (np.uint16, np.uint8):
        from tools.tactile_uint16_to_uint8 import tactile_uint16_to_uint8

        return tactile_uint16_to_uint8(frame)
    return frame


def _show_live(name, dev, save_name, out_dir):
    """实时显示某个相机, 直到按 q/ESC/n; 's' 存当前帧。帧为 RGB, 显示需转 BGR。"""
    import cv2

    win = f"{name}  (q/ESC/n=下一个, s=存图)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    saw_image = False
    preview = None
    while True:
        img = dev.async_read()
        if not frame_is_black(img):
            saw_image = True
            preview = _to_preview_uint8(img)
            cv2.imshow(win, preview[..., ::-1])  # RGB -> BGR
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27, ord("n")):  # q / ESC / n
            break
        if key == ord("s") and saw_image and preview is not None:
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{save_name}.png"), preview[..., ::-1])
            print(f"    已存 {save_name}.png")
    cv2.destroyWindow(win)
    (ok if saw_image else warn)(f"{name:18s} " + ("显示正常" if saw_image else "整段未拿到非空帧"))


# ==================================================================
# 阶段 1: 存在性
# ==================================================================
def stage_existence(cfg, tcfg, arms) -> bool:
    header("阶段 1 / 存在性 (ping IP + 串口 + /dev/video, 不连硬件)")
    all_ok = True

    print("\n[从臂 / 板子 IP]")
    for side in arms:
        for label, ip in ((f"{side} 从臂", follower_ip(cfg, side)),
                          (f"{side} 板子", board_ip(cfg, side))):
            if ping(ip):
                ok(f"{label:10s} {ip} 可 ping 通")
            else:
                fail(f"{label:10s} {ip} ping 不通")
                all_ok = False

    print(f"\n[本机 IP (pc_host, 触觉 UDP 回传用)]")
    if ping(cfg.pc_host):
        ok(f"pc_host {cfg.pc_host} 本机可达")
    else:
        warn(f"pc_host {cfg.pc_host} ping 不通 (若是本机网卡 IP 一般也没事, 但触觉回传需它正确)")

    print("\n[主臂串口]")
    for side in arms:
        p = leader_port(tcfg, side)
        if os.path.exists(p):
            ok(f"{side} 主臂 {p} 存在")
        else:
            fail(f"{side} 主臂 {p} 不存在 (检查 USB / udev 规则)")
            all_ok = False

    print("\n[顶部相机]")
    cam_top = cfg.cameras.get("cam_top")
    if cam_top is None:
        warn("config 未配置 cam_top (use 无 top 模式), 跳过")
    else:
        idx = cam_top.index_or_path
        dev = f"/dev/video{idx}" if isinstance(idx, int) else str(idx)
        if os.path.exists(dev):
            ok(f"cam_top {dev} 存在")
        else:
            fail(f"cam_top {dev} 不存在 (用 `ls /dev/video*` 看实际索引)")
            all_ok = False

    return all_ok


# ==================================================================
# 阶段 2: 摄像头 / 触觉图像
# ==================================================================
def stage_camera(cfg, arms, use_tactile, save: bool, show: bool) -> bool:
    header("阶段 2 / 图像 (实际连接并抓帧" + ("/实时显示)" if show else ")"))
    from deployment.hardware.wrist_cameras import FisheyeGrpcCamera
    from deployment.hardware.tactile_sensors import DmroboticsFlux
    from deployment.hardware.top_cameras import OpenCVTopCamera

    all_ok = True
    out_dir = os.path.join(os.path.dirname(__file__), "_check_output")
    if save:
        os.makedirs(out_dir, exist_ok=True)

    if show:
        print(f"\n{_Y}实时显示: 每个相机一个窗口, 按 q/ESC/n 看下一个, s 存当前帧。{_0}")

    def _grab(name, dev, save_name):
        nonlocal all_ok
        try:
            dev.connect()
            time.sleep(0.3)
            if show:
                _show_live(name, dev, save_name, out_dir)
            else:
                img = dev.async_read()
                shape = getattr(img, "shape", None)
                if frame_is_black(img):
                    warn(f"{name:18s} 连上了但首帧全黑/空 (shape={shape}) — 可能还没出帧或源异常")
                else:
                    ok(f"{name:18s} 拿到图像 shape={shape}")
                    preview = _to_preview_uint8(img)
                    if save:
                        import cv2

                        cv2.imwrite(
                            os.path.join(out_dir, f"{save_name}.png"), preview[..., ::-1]
                        )
        except Exception as e:
            fail(f"{name:18s} 失败: {e}")
            all_ok = False
        finally:
            try:
                dev.disconnect()
            except Exception:
                pass

    cam_top = cfg.cameras.get("cam_top")
    if cam_top is not None:
        print("\n[顶部相机]")
        _grab("cam_top", OpenCVTopCamera(cam_top, name="cam_top"), "cam_top")

    for side in arms:
        print(f"\n[{side} 臂 数据流]")
        _grab(f"{side}_cam_wrist", FisheyeGrpcCamera(
            ip=board_ip(cfg, side), grpc_port=cfg.fisheye_grpc_port,
            udp_port=fisheye_udp_port(cfg, side),
            width=cfg.fisheye_width, height=cfg.fisheye_height,
            max_datagram=cfg.fisheye_max_datagram, max_fps=cfg.stream_max_fps,
            first_frame_timeout=cfg.stream_first_frame_timeout, name=f"{side}_cam_wrist",
        ), f"{side}_cam_wrist")

        if use_tactile:
            p0, p1 = tactile_pc_ports(cfg, side)
            for ti, (dev_id, gport, pc_port) in enumerate((
                (cfg.tactile0_dev_id, cfg.tactile0_grpc_port, p0),
                (cfg.tactile1_dev_id, cfg.tactile1_grpc_port, p1),
            )):
                _grab(f"{side}_cam_finger{ti}", DmroboticsFlux(
                    dev_id=dev_id, remote_addr=f"{board_ip(cfg, side)}:{gport}",
                    pc_host=cfg.pc_host, pc_port=pc_port,
                    width=cfg.tactile_width, height=cfg.tactile_height,
                    max_fps=cfg.tactile_max_fps,
                    first_frame_timeout=cfg.stream_first_frame_timeout,
                    name=f"{side}_cam_finger{ti}",
                ), f"{side}_cam_finger{ti}")

    if save:
        print(f"\n图像已存到: {out_dir}")
    return all_ok


# ==================================================================
# 阶段 3: 主从同步 (会让从臂运动!)
# ==================================================================
def _normalize_leader_gripper(raw: float, tcfg) -> float:
    """主臂夹爪原始读数 -> [0,1] (1=张开), 与 bi leader 的归一化一致。"""
    lo, hi = tcfg.leader_gripper_min, tcfg.leader_gripper_max
    norm = (raw - lo) / max(hi - lo, 1e-6)
    norm = 0.5 + (norm - 0.5) * tcfg.gripper_gain  # 以中点放大
    return float(max(0.0, min(1.0, norm)))


def _limit_joint_velocity(previous, target, max_speed: float, dt: float):
    """Limit every commanded joint delta to max_speed * dt."""
    import numpy as np

    previous = np.asarray(previous, dtype=float)
    target = np.asarray(target, dtype=float)
    if previous.shape != target.shape:
        raise ValueError(f"joint shape mismatch: previous={previous.shape}, target={target.shape}")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(target)):
        raise ValueError("joint positions must be finite")
    if not np.isfinite(max_speed) or not np.isfinite(dt) or max_speed <= 0 or dt <= 0:
        raise ValueError("max_speed and dt must be positive")
    max_step = float(max_speed) * float(dt)
    return previous + np.clip(target - previous, -max_step, max_step)


def stage_teleop(
    cfg,
    tcfg,
    arms,
    duration: float,
    confirm_move: bool,
    use_gripper: bool,
    max_joint_speed_deg_s: float,
) -> bool:
    header("阶段 3 / 主从同步 (⚠️ 从臂会跟随主臂运动!)")
    import numpy as np
    from deployment.hardware.leader_arms import RealmanLeader
    from deployment.hardware.follower_arms import RealmanTcpFollower

    if not confirm_move:
        warn("未加 --confirm-move, 跳过 (该阶段会驱动从臂, 确认现场无障碍后再加该参数)")
        return True

    leaders, followers, grippers = {}, {}, {}
    all_ok = True
    try:
        # 连接
        for side in arms:
            lp = leader_port(tcfg, side)
            print(f"\n[{side}] 连接主臂 {lp} + 从臂 {follower_ip(cfg, side)} ...")
            ld = RealmanLeader(port=lp, baudrate=tcfg.baudrate, hex_data=tcfg.hex_data)
            ld.connect()
            leaders[side] = ld
            fo = RealmanTcpFollower(
                ip=follower_ip(cfg, side), port=cfg.follower_tcp_port,
                dof=7, use_degrees=cfg.use_degrees, name=f"{side}_follower",
            )
            fo.connect()
            followers[side] = fo

            # 夹爪 (非致命: 连不上就只测手臂)
            if use_gripper:
                from deployment.hardware.grippers import LingkongGripper
                try:
                    print(f"[{side}] 连接从臂夹爪 {board_ip(cfg, side)}:{cfg.gripper_grpc_port} (会自标定夹紧) ...")
                    gp = LingkongGripper(
                        server_address=f"{board_ip(cfg, side)}:{cfg.gripper_grpc_port}",
                        can_interface=cfg.gripper_can_interface,
                        can_bitrate=cfg.gripper_can_bitrate,
                        speed=cfg.gripper_speed, torque=cfg.gripper_torque,
                    )
                    if gp.connect() and gp.init_gripper(itinerary_override=gripper_itinerary(cfg, side)):
                        grippers[side] = gp
                        ok(f"{side} 夹爪就绪")
                    else:
                        warn(f"{side} 夹爪连接/初始化失败, 该臂只测手臂关节")
                except Exception as e:
                    warn(f"{side} 夹爪异常 ({e}), 该臂只测手臂关节")

        # 从实测从臂位置起步；后续目标每周期做逐关节速度限制。
        commanded = {}
        print("\n初始位姿对比 (主臂目标 vs 从臂当前):")
        for side in arms:
            lpos = np.asarray(leaders[side].read_position()[:7], dtype=float)
            if cfg.use_degrees:
                lpos = np.rad2deg(lpos)
            fpos = followers[side].read_joints_now()
            if fpos is None:
                fpos = followers[side].read_joints()
            fpos = np.asarray(fpos, dtype=float)
            if fpos.shape != (7,) or not np.all(np.isfinite(fpos)):
                raise RuntimeError(f"[{side}] 无法读取有效的 7 维从臂初始关节位置: {fpos}")
            commanded[side] = fpos.copy()
            gap = float(np.max(np.abs(lpos - fpos)))
            gap_deg = gap if cfg.use_degrees else float(np.rad2deg(gap))
            msg = f"{side}: 最大关节差 {gap_deg:.1f} deg"
            (warn if gap_deg > 15.0 else ok)(
                msg + ("；将从当前从臂姿态按限速平滑追赶" if gap_deg > 15.0 else "")
            )

        unit = "度" if cfg.use_degrees else "弧度"
        max_joint_speed = (
            max_joint_speed_deg_s
            if cfg.use_degrees
            else float(np.deg2rad(max_joint_speed_deg_s))
        )
        print(
            f"{_Y}逐关节目标限速: {max_joint_speed_deg_s:.1f} deg/s "
            f"(控制周期最多按 50ms 计算){_0}"
        )
        print(f"\n{_Y}3 秒后开始主从同步: 手动操作主臂, 从臂跟随。按 q / ESC 结束 (Ctrl-C 也可)。{_0}")
        if duration > 0:
            print(f"{_Y}(最长 {duration:.0f}s 后自动停止){_0}")
        for n in (3, 2, 1):
            print(f"  {n}...", flush=True); time.sleep(1)
        print("  同步中... (操作主臂, q/ESC 结束)")

        # 同步循环: 一直跑到按键 (或到达可选上限 duration)
        t0 = time.time()
        last_command_time = time.monotonic()
        max_gap = {side: 0.0 for side in arms}
        n_iter = 0
        with _raw_stdin():
            while True:
                k = _poll_key()
                if k in ("q", "\x1b"):  # q / ESC
                    print("\n  收到停止键")
                    break
                if duration > 0 and time.time() - t0 > duration:
                    print("\n  到达最长时长, 自动停止")
                    break
                now = time.monotonic()
                # Cap dt so a debugger pause or serial stall cannot authorize one large jump.
                dt = min(max(now - last_command_time, 1e-3), 0.05)
                last_command_time = now
                for side in arms:
                    pos = leaders[side].read_position()
                    target = np.asarray(pos[:7], dtype=float)
                    if cfg.use_degrees:
                        target = np.rad2deg(target)
                    limited = _limit_joint_velocity(
                        commanded[side], target, max_joint_speed, dt
                    )
                    if followers[side].send_joints(limited) == 0:
                        commanded[side] = limited
                    # 夹爪: 主臂第 8 个读数归一化后下发
                    if side in grippers:
                        grippers[side].move_norm(_normalize_leader_gripper(pos[7], tcfg))
                time.sleep(0.02)  # ~50Hz
                # 抽样误差
                for side in arms:
                    fnow = followers[side].read_joints()
                    tgt = np.asarray(leaders[side].read_position()[:7])
                    if cfg.use_degrees:
                        tgt = np.rad2deg(tgt)
                    max_gap[side] = max(max_gap[side], float(np.max(np.abs(tgt - fnow))))
                n_iter += 1

        print(f"\n同步结束 ({n_iter} 次下发):")
        for side in arms:
            g = max_gap[side]
            (ok if g < 0.2 else warn)(f"{side}: 跟随期间最大滞后 {g:.3f} {unit} "
                                      + ("(跟随良好)" if g < 0.2 else "(滞后偏大: 检查网络/限速/从臂负载)"))
    except KeyboardInterrupt:
        warn("用户中断")
    except Exception as e:
        fail(f"主从同步出错: {e}")
        all_ok = False
    finally:
        for d in list(grippers.values()) + list(followers.values()) + list(leaders.values()):
            try:
                d.disconnect()
            except Exception:
                pass
    return all_ok


# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="睿尔曼 ugripper 硬件自检")
    ap.add_argument(
        "--robot-type",
        choices=["rm_base_umi_dual", "rm_isf_umi_left", "rm_isf_umi_right"],
        default="rm_base_umi_dual",
    )
    ap.add_argument("--stage", choices=["existence", "camera", "teleop", "all"], default="existence")
    ap.add_argument("--arms", default=None, help="逗号分隔, 如 left,right (默认取 config)")
    ap.add_argument("--left-port", default=None, help="左主臂串口 (默认取 teleop config)")
    ap.add_argument("--right-port", default=None, help="右主臂串口")
    ap.add_argument("--no-tactile", action="store_true", help="跳过触觉")
    ap.add_argument("--save", action="store_true", help="阶段2 把抓到的帧存成 png")
    ap.add_argument("--show", action="store_true", help="阶段2 开窗实时显示 (而非存图; 需要 DISPLAY)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="阶段3 最长秒数; 0=一直跟随到按 q/ESC (默认)")
    ap.add_argument("--confirm-move", action="store_true", help="阶段3 必须显式确认 (会驱动从臂)")
    ap.add_argument("--no-gripper", action="store_true", help="阶段3 不连/不同步从臂夹爪")
    ap.add_argument(
        "--max-joint-speed-deg-s",
        type=float,
        default=2.0,
        help="阶段3每个关节的最大目标角速度 deg/s (默认 2)",
    )
    args = ap.parse_args()
    if not math.isfinite(args.max_joint_speed_deg_s) or args.max_joint_speed_deg_s <= 0:
        ap.error("--max-joint-speed-deg-s 必须是大于 0 的有限值")

    config_by_type = {
        "rm_base_umi_dual": (RmBaseUmiDualConfig, RmLeaderDualConfig),
        "rm_isf_umi_left": (RmIsfUmiLeftConfig, RmLeaderLeftConfig),
        "rm_isf_umi_right": (RmIsfUmiRightConfig, RmLeaderRightConfig),
    }
    robot_config_cls, teleop_config_cls = config_by_type[args.robot_type]
    cfg = robot_config_cls()
    tcfg = teleop_config_cls()
    if args.left_port:
        if args.robot_type == "rm_isf_umi_left":
            tcfg.port = args.left_port
        elif isinstance(tcfg, RmLeaderDualConfig):
            tcfg.left_port = args.left_port
        else:
            ap.error("rm_isf_umi_right 不接受 --left-port；请使用 --right-port")
    if args.right_port:
        if args.robot_type == "rm_isf_umi_right":
            tcfg.port = args.right_port
        elif isinstance(tcfg, RmLeaderDualConfig):
            tcfg.right_port = args.right_port
        else:
            ap.error("rm_isf_umi_left 不接受 --right-port；请使用 --left-port")
    default_arms = list(cfg.kinematics_sides) if not hasattr(cfg, "arms") else list(cfg.arms)
    arms = args.arms.split(",") if args.arms else default_arms
    if not hasattr(cfg, "arms") and tuple(arms) != cfg.kinematics_sides:
        ap.error(f"{args.robot_type} 只支持 --arms {','.join(cfg.kinematics_sides)}")
    use_tactile = cfg.use_tactile and not args.no_tactile

    print(f"自检目标: arms={arms}, use_tactile={use_tactile}, stage={args.stage}")

    results = {}
    if args.stage in ("existence", "all"):
        results["existence"] = stage_existence(cfg, tcfg, arms)
    if args.stage in ("camera", "all"):
        results["camera"] = stage_camera(cfg, arms, use_tactile, args.save, args.show)
    if args.stage in ("teleop", "all"):
        results["teleop"] = stage_teleop(
            cfg,
            tcfg,
            arms,
            args.duration,
            args.confirm_move,
            use_gripper=not args.no_gripper,
            max_joint_speed_deg_s=args.max_joint_speed_deg_s,
        )

    header("汇总")
    for k, v in results.items():
        (ok if v else fail)(f"{k}: {'通过' if v else '有问题, 见上'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
