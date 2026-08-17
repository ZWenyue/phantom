#!/usr/bin/env python3
"""Continuously run YOLO-World forward passes to keep GPU utilization high.

Stop with:  kill $(cat /tmp/keep_gpu_busy.pid)
"""
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import torch

STOP = False


def _handle_stop(signum, _frame):
    global STOP
    STOP = True
    print(f"\n[keep_gpu] caught signal {signum}, stopping...", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="/home/a26160/SRC/phantom/yolov8x-worldv2.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--reserve-gb", type=float, default=48.0, help="extra VRAM to pin")
    p.add_argument("--log-every", type=int, default=30)
    p.add_argument("--pid-file", default="/tmp/keep_gpu_busy.pid")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    Path(args.pid_file).write_text(str(os.getpid()))
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    from ultralytics import YOLOWorld

    print(f"[keep_gpu] pid={os.getpid()} loading {args.weights} on {device}", flush=True)
    model = YOLOWorld(args.weights)
    model.to(device)
    model.model.eval()
    model.set_classes(["object", "hand", "book", "stapler"])
    core = model.model
    if hasattr(core, "fuse"):
        try:
            core.fuse()
        except Exception as exc:
            print(f"[keep_gpu] fuse skipped: {exc}", flush=True)
    core.half()

    x = torch.randn(
        args.batch, 3, args.imgsz, args.imgsz, device=device, dtype=torch.float16
    )
    reserve = None
    if args.reserve_gb > 0:
        n = int(args.reserve_gb * (1024 ** 3) / 2)  # float16 bytes
        reserve = torch.empty(n, device=device, dtype=torch.float16)
        print(f"[keep_gpu] reserved ~{args.reserve_gb:.1f} GiB extra VRAM", flush=True)

    with torch.inference_mode():
        # warmup
        for _ in range(3):
            core(x)
        torch.cuda.synchronize()

    step = 0
    t0 = time.time()
    last_log = t0
    print(
        f"[keep_gpu] looping batch={args.batch} imgsz={args.imgsz} "
        f"mem={torch.cuda.memory_allocated(device)/1024**3:.1f} GiB allocated",
        flush=True,
    )
    with torch.inference_mode():
        while not STOP:
            core(x)
            if reserve is not None:
                reserve.mul_(1.0000001)  # tiny op so the reserved buffer stays live
            step += 1
            now = time.time()
            if now - last_log >= args.log_every:
                torch.cuda.synchronize()
                dt = now - t0
                fps = step * args.batch / max(dt, 1e-6)
                mem = torch.cuda.memory_allocated(device) / 1024 ** 3
                print(
                    f"[keep_gpu] t={dt:.0f}s steps={step} ~{fps:.1f} img/s "
                    f"alloc={mem:.1f}GiB",
                    flush=True,
                )
                last_log = now

    print("[keep_gpu] exited", flush=True)
    try:
        os.remove(args.pid_file)
    except OSError:
        pass


if __name__ == "__main__":
    main()
