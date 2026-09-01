#!/usr/bin/env python3
"""
双臂 统一实时可视化 (PC 端) —— 多进程版

每个臂一个窗口 (arm_left / arm_right), 每个窗口横排 3 个画面:
    [ 鱼眼(手腕相机) | 触觉0 | 触觉1 ]

数据来源 (全 gRPC, 板子上服务开机自启):
    - 鱼眼  : fish_camera gRPC (50088) -> OpenStream(MJPG) -> FCP1 分片 UDP -> 重组 + imdecode
    - 触觉  : dmrobotics Flux gRPC (50051 dev0 / 50052 dev2) -> getDepth + getDeformation2D

触觉可视化 (复用 main.py / 旧 pc_unified_client 的归一化):
    depth*50 灰度做底 -> GRAY2BGR -> 叠加 deformation 箭头 (put_arrows_on_image)

每个画面左上角显示该路实时 FPS。

架构: 每路数据源各一个进程 (dmrobotics 多传感器/多 gRPC 流在单进程线程里会互相饿死 GIL,
      必须用多进程, 与原版 main_mp.py 一致)。各进程把缩放后的小图经 Queue(maxsize=1)
      传回主进程合成显示。

运行: /home/dm/miniforge3/envs/lerobot_tactile/bin/python unified_client.py
      --arms left,right          # 选臂
      --headless 6               # 无显示器: 收6秒, 每臂存 unified_<arm>.png 并打印 fps
按 q / ESC 退出
"""
import argparse
import os
import socket
import sys
import time
import multiprocessing as mp
from queue import Empty

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # deployment/tools/
SDK_ROOT = os.path.join(os.path.dirname(HERE), "sdk")  # deployment/sdk/
sys.path.insert(0, os.path.join(SDK_ROOT, "SDK_Publish_1.2.3"))
sys.path.insert(0, os.path.join(SDK_ROOT, "fish_camera_client"))

PC_HOST = "192.168.1.120"
PANEL_H = 360  # 每个画面显示高度

ARMS = {
    "left": {
        "ip": "192.168.1.11",
        "fisheye": {"grpc_port": 50088, "udp_port": 50100},
        "tactiles": [
            {"name": "tac0", "remote": "192.168.1.11:50051", "dev_id": 0, "pc_port": 60000},
            {"name": "tac1", "remote": "192.168.1.11:50052", "dev_id": 2, "pc_port": 60001},
        ],
    },
    "right": {
        "ip": "192.168.1.10",
        "fisheye": {"grpc_port": 50088, "udp_port": 50101},
        "tactiles": [
            {"name": "tac0", "remote": "192.168.1.10:50051", "dev_id": 0, "pc_port": 60002},
            {"name": "tac1", "remote": "192.168.1.10:50052", "dev_id": 2, "pc_port": 60003},
        ],
    },
}


def _resize_h(img, h=PANEL_H):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def _put_latest(q, item):
    """maxsize=1 队列: 丢旧留新。"""
    try:
        q.get_nowait()
    except Empty:
        pass
    try:
        q.put_nowait(item)
    except Exception:
        pass


# ==================== 触觉进程 ====================
def tactile_proc(desc, q, stop):
    from dmrobotics import Sensor, SensorOptions, Mode
    from dmrobotics.utils import put_arrows_on_image

    t = []
    def fps_now():
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    while not stop.is_set():
        sensor = None
        try:
            opt = SensorOptions(
                dev_id=desc["dev_id"], backend="Flux", mode=Mode.STANDARD, show_fps=False,
                enable_raw=False, enable_deformation=True, enable_depth=True,
                enable_shear=False, enable_force=False,
                remote_addr=desc["remote"], pc_host=PC_HOST, pc_port=desc["pc_port"],
            )
            sensor = Sensor(opt)
            last = -1
            while not stop.is_set():
                if not sensor.wait_for_new(last, timeout_ms=500):
                    continue
                fid, depth = sensor.getDepth()
                _, deform = sensor.getDeformation2D()
                last = fid if fid is not None else last
                if depth is None:
                    continue
                depth = np.asarray(depth)
                depth_u8 = (depth * 50).clip(0, 255).astype(np.uint8)
                canvas = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
                if deform is not None:
                    deform = np.asarray(deform)
                    if deform.ndim == 3 and deform.shape[-1] >= 2:
                        canvas = put_arrows_on_image(canvas, deform, step=16, scale=20.0)
                now = time.time(); t.append(now)
                if len(t) > 30:
                    t.pop(0)
                _put_latest(q, (_resize_h(canvas), fps_now()))
        except Exception as e:
            if not stop.is_set():
                print(f"[{desc['name']}] 触觉出错, 2s后重连: {e}", flush=True)
                time.sleep(2)
        finally:
            try:
                if sensor is not None:
                    sensor.disconnect()
            except Exception:
                pass


# ==================== 鱼眼进程 ====================
def fisheye_proc(ip, grpc_port, udp_port, q, stop, want_w=1920, want_h=1080):
    import grpc
    import camera_proxy_pb2 as pb
    import camera_proxy_pb2_grpc as pbg
    from udp_frame import FrameReassembler

    t = []
    def fps_now():
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    while not stop.is_set():
        sock = None; channel = None; call = None
        try:
            channel = grpc.insecure_channel(f"{ip}:{grpc_port}")
            grpc.channel_ready_future(channel).result(timeout=8)
            stub = pbg.CameraProxyStub(channel)
            caps = stub.ListCapabilities(pb.CapabilityRequest(device=""), timeout=5)
            device = w = h = fps = None
            for cam in caps.cameras:
                for codec in cam.codecs:
                    if codec.codec == "MJPG" and len(codec.modes):
                        modes = list(codec.modes)
                        m = next((x for x in modes if x.width == want_w and x.height == want_h),
                                 max(modes, key=lambda x: x.width * x.height))
                        device, w, h, fps = cam.device, m.width, m.height, (max(m.fps) if m.fps else 30)
                        break
                if device:
                    break
            if device is None:
                raise RuntimeError("无 MJPG 模式")

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
            sock.bind(("0.0.0.0", udp_port))
            sock.settimeout(1.0)

            req = pb.StreamRequest(codec="MJPG", width=w, height=h, fps=int(fps),
                                   udp_port=udp_port, client_ip="", device=device, max_datagram=1200)
            call = stub.OpenStream(req)
            import threading
            def _drain():
                try:
                    for _ in call:
                        pass
                except Exception:
                    pass  # 关闭时 call.cancel() 会抛 CANCELLED, 忽略
            threading.Thread(target=_drain, daemon=True).start()

            reasm = FrameReassembler(timeout_sec=0.5)
            while not stop.is_set():
                try:
                    datagram, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                done = reasm.push(datagram)
                if done is None:
                    continue
                img = cv2.imdecode(np.frombuffer(done.payload, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                now = time.time(); t.append(now)
                if len(t) > 30:
                    t.pop(0)
                _put_latest(q, (_resize_h(img), fps_now()))
        except Exception as e:
            if not stop.is_set():
                print(f"[fisheye {ip}] 出错, 2s后重连: {e}", flush=True)
                time.sleep(2)
        finally:
            for c in (call,):
                try:
                    c.cancel()
                except Exception:
                    pass
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
            try:
                if channel is not None:
                    channel.close()
            except Exception:
                pass


# ==================== 主进程显示 ====================
def overlay(panel, label, fps):
    txt = f"{label}  {fps:4.1f} fps"
    cv2.putText(panel, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return panel


def placeholder(label):
    p = np.full((PANEL_H, int(PANEL_H * 4 / 3), 3), 40, np.uint8)
    cv2.putText(p, f"{label} connecting...", (10, PANEL_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2, cv2.LINE_AA)
    return p


def hstack_pad(panels):
    H = max(p.shape[0] for p in panels)
    out = []
    for p in panels:
        if p.shape[0] != H:
            c = np.zeros((H, p.shape[1], 3), np.uint8)
            c[:p.shape[0]] = p
            p = c
        out.append(p)
        out.append(np.zeros((H, 4, 3), np.uint8))
    return np.hstack(out[:-1])


def main():
    ap = argparse.ArgumentParser(description="双臂 鱼眼+触觉 统一可视化 (多进程)")
    ap.add_argument("--arms", default="left,right")
    ap.add_argument("--headless", type=float, default=0.0)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    stop = mp.Event()
    procs = []
    # arm -> {"labels":[...], "queues":[...], "last":[(panel,fps)|None]}
    view = {}

    for arm in arms:
        cfg = ARMS[arm]
        labels, queues = [], []
        # 鱼眼
        qf = mp.Queue(maxsize=1)
        procs.append(mp.Process(target=fisheye_proc,
                                args=(cfg["ip"], cfg["fisheye"]["grpc_port"],
                                      cfg["fisheye"]["udp_port"], qf, stop), daemon=True))
        labels.append("fisheye"); queues.append(qf)
        # 触觉
        for td in cfg["tactiles"]:
            qt = mp.Queue(maxsize=1)
            procs.append(mp.Process(target=tactile_proc, args=(td, qt, stop), daemon=True))
            labels.append(td["name"]); queues.append(qt)
        view[arm] = {"labels": labels, "queues": queues, "last": [None] * len(queues)}

    for p in procs:
        p.start()

    headless = args.headless > 0
    if not headless:
        for arm in arms:
            cv2.namedWindow(f"arm_{arm}", cv2.WINDOW_NORMAL)
    print(f"已启动: {arms}  {'[headless %gs]' % args.headless if headless else '按 q/ESC 退出'}", flush=True)

    def build(arm):
        v = view[arm]
        panels = []
        for i, (label, q) in enumerate(zip(v["labels"], v["queues"])):
            # 取队列最新
            try:
                while True:
                    v["last"][i] = q.get_nowait()
            except Empty:
                pass
            item = v["last"][i]
            if item is None:
                panels.append(placeholder(label))
            else:
                panel, fps = item
                panels.append(overlay(panel.copy(), label, fps))
        return hstack_pad(panels)

    t0 = time.time()
    try:
        while True:
            if headless:
                if time.time() - t0 >= args.headless:
                    break
                time.sleep(0.05)
                continue
            for arm in arms:
                cv2.imshow(f"arm_{arm}", build(arm))
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if headless:
            for arm in arms:
                canvas = build(arm)
                cv2.imwrite(f"unified_{arm}.png", canvas)
                v = view[arm]
                stats = "  ".join(f"{lb}={(it[1] if it else 0):.1f}" for lb, it in zip(v["labels"], v["last"]))
                print(f"[{arm}] {stats}  -> unified_{arm}.png {canvas.shape[1]}x{canvas.shape[0]}", flush=True)
        stop.set()
        time.sleep(0.6)
        for p in procs:
            if p.is_alive():
                p.terminate()
        if not headless:
            cv2.destroyAllWindows()
    print("退出")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
