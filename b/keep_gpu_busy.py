#!/usr/bin/env python3
"""Continuously run YOLO-World forward passes to keep GPU utilization high.

One process per GPU. Default: all visible CUDA devices.

Stop with:  kill $(cat /tmp/keep_gpu_busy.pid)
"""
from __future__ import annotations

import argparse
import os
import signal
import time
from multiprocessing import Process, set_start_method
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
    p.add_argument(
        "--device",
        default=None,
        help="single device, e.g. cuda:0 (overrides --devices)",
    )
    p.add_argument(
        "--devices",
        default="all",
        help="comma-separated GPU ids (0,1,2) or 'all' (default: all visible GPUs)",
    )
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--reserve-gb", type=float, default=48.0, help="extra VRAM to pin per GPU")
    p.add_argument("--log-every", type=int, default=30)
    p.add_argument("--pid-file", default="/tmp/keep_gpu_busy.pid")
    return p.parse_args()


def resolve_devices(args: argparse.Namespace) -> list[str]:
    if args.device:
        d = args.device.strip()
        return [d if d.startswith("cuda:") else f"cuda:{int(d)}"]

    raw = (args.devices or "all").strip()
    if raw.lower() in ("all", "*"):
        n = torch.cuda.device_count()
        if n == 0:
            raise SystemExit("[keep_gpu] no CUDA devices visible")
        return [f"cuda:{i}" for i in range(n)]

    out: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("cuda:"):
            out.append(part)
        else:
            out.append(f"cuda:{int(part)}")
    if not out:
        raise SystemExit("[keep_gpu] --devices parsed to empty list")
    return out


def run_loop(device_str: str, args: argparse.Namespace) -> None:
    global STOP
    STOP = False
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    tag = f"[keep_gpu {device_str}]"
    torch.backends.cudnn.benchmark = True
    device = torch.device(device_str)
    torch.cuda.set_device(device)

    from ultralytics import YOLOWorld

    print(f"{tag} pid={os.getpid()} loading {args.weights}", flush=True)
    model = YOLOWorld(args.weights)
    model.to(device)
    model.model.eval()
    model.set_classes(["object", "hand", "book", "stapler"])
    core = model.model
    if hasattr(core, "fuse"):
        try:
            core.fuse()
        except Exception as exc:
            print(f"{tag} fuse skipped: {exc}", flush=True)
    core.half()

    x = torch.randn(
        args.batch, 3, args.imgsz, args.imgsz, device=device, dtype=torch.float16
    )
    reserve = None
    if args.reserve_gb > 0:
        n = int(args.reserve_gb * (1024 ** 3) / 2)  # float16 bytes
        reserve = torch.empty(n, device=device, dtype=torch.float16)
        print(f"{tag} reserved ~{args.reserve_gb:.1f} GiB extra VRAM", flush=True)

    with torch.inference_mode():
        for _ in range(3):
            core(x)
        torch.cuda.synchronize(device)

    step = 0
    t0 = time.time()
    last_log = t0
    print(
        f"{tag} looping batch={args.batch} imgsz={args.imgsz} "
        f"mem={torch.cuda.memory_allocated(device)/1024**3:.1f} GiB allocated",
        flush=True,
    )
    with torch.inference_mode():
        while not STOP:
            core(x)
            if reserve is not None:
                reserve.mul_(1.0000001)
            step += 1
            now = time.time()
            if now - last_log >= args.log_every:
                torch.cuda.synchronize(device)
                dt = now - t0
                fps = step * args.batch / max(dt, 1e-6)
                mem = torch.cuda.memory_allocated(device) / 1024 ** 3
                print(
                    f"{tag} t={dt:.0f}s steps={step} ~{fps:.1f} img/s "
                    f"alloc={mem:.1f}GiB",
                    flush=True,
                )
                last_log = now

    print(f"{tag} exited", flush=True)


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    Path(args.pid_file).write_text(str(os.getpid()))
    devices = resolve_devices(args)
    print(
        f"[keep_gpu] parent pid={os.getpid()} devices={','.join(devices)}",
        flush=True,
    )

    if len(devices) == 1:
        run_loop(devices[0], args)
    else:
        set_start_method("spawn", force=True)
        procs = [
            Process(target=run_loop, args=(d, args), name=f"keep_gpu_{d}")
            for d in devices
        ]
        for proc in procs:
            proc.start()
        try:
            while not STOP:
                alive = [proc for proc in procs if proc.is_alive()]
                if not alive:
                    break
                time.sleep(0.4)
        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            for proc in procs:
                proc.join(timeout=15)

    print("[keep_gpu] exited", flush=True)
    try:
        os.remove(args.pid_file)
    except OSError:
        pass


if __name__ == "__main__":
    main()
