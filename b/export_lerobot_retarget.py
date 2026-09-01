#!/usr/bin/env python3
"""Export contact-grounded retarget demos to LeRobot v2.1.

Unlike ``export_lerobot.py`` (legacy ``inpaint_processor/training_data_*.npz`` +
root ``video_overlay_*_*.mkv``), this reads Stage C outputs:

  {demo}/retarget_processor/quality_report.npz   (must have accept=True)
  {demo}/retarget_processor/training_data.npz
      (joints, gripper, EE pos xyz + quat xyzw in robot frame)
  {demo}/retarget_processor/video_overlay.mkv
  {demo}/narration.csv

Parquet columns: observation.state / action (16-D joints+gripper) and
observation.ee / action.ee (14-D: left then right, pos+xyzw). action* is the
next-frame value with the last frame repeated.

Demos that failed the trajectory quality gate are skipped even if a leftover
``training_data.npz`` / overlay from a previous run is still on disk.

Run from ``phantom/`` in the ``lerobot`` env::

    python b/export_lerobot_retarget.py \\
        --src /home/a26160/DATA/test_phantom_processed/egodex_basic_pick_place \\
        --dst /home/a26160/DATA/test_phantom_processed_lerobot/egodex_basic_pick_place_retarget
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

FPS = 15
CHUNK_SIZE = 1000
CAM_KEY = "observation.images.cam_ego"
N_JOINTS = 7
STATE_NAMES = (
    [f"left_joint_{i}" for i in range(1, N_JOINTS + 1)] + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(1, N_JOINTS + 1)] + ["right_gripper"]
)
# Robot-frame EE from training_data.npz: pos (xyz) + scipy xyzw quaternion.
EE_NAMES = (
    [f"left_ee_{k}" for k in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    + [f"right_ee_{k}" for k in ("x", "y", "z", "qx", "qy", "qz", "qw")]
)
EE_KEY = "observation.ee"
EE_ACTION_KEY = "action.ee"


def _export_helpers():
    """Lazy-import OpenCV / pandas helpers so ``--dry-run`` only needs numpy."""
    _b = Path(__file__).resolve().parent
    if str(_b) not in sys.path:
        sys.path.insert(0, str(_b))
    from export_lerobot import encode_video_and_sample, image_stats, vec_stats
    import pandas as pd
    return encode_video_and_sample, image_stats, vec_stats, pd


def read_narration(path: Path) -> str:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    data = {}
    for row in rows[1:]:
        if not row:
            continue
        data[row[0]] = ",".join(row[1:])
    return data.get("narration", "").strip()


def _next_frame(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[1:], x[-1:]], axis=0)


def _ee_pose(d, valid: np.ndarray, side: str) -> np.ndarray:
    pos = np.asarray(d[f"action_pos_{side}"], dtype=np.float32)[valid]
    quat = np.asarray(d[f"action_orixyzw_{side}"], dtype=np.float32)[valid]
    return np.concatenate([pos, quat], axis=1)


def load_state_action(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    valid = d["valid"]
    jl = d["joint_pos_left"][valid]
    jr = d["joint_pos_right"][valid]
    gl = d["gripper_width_left"][valid]
    gr = d["gripper_width_right"][valid]
    state = np.concatenate([jl, gl[:, None], jr, gr[:, None]], axis=1).astype(np.float32)
    action = _next_frame(state)
    ee = np.concatenate([_ee_pose(d, valid, "left"), _ee_pose(d, valid, "right")], axis=1)
    ee_action = _next_frame(ee)
    return state, action, ee, ee_action, valid


def _bool(x) -> bool:
    return bool(np.asarray(x).reshape(-1)[0])


def _reasons(report) -> list[str]:
    if "reasons" not in report.files:
        return []
    raw = np.asarray(report["reasons"], dtype=object).reshape(-1)
    return [str(x) for x in raw.tolist() if str(x)]


def _has_grasp_and_release(demo_dir: Path) -> bool:
    stageb = demo_dir / "stageb_processor"
    names = ["q_trajectory_right.npz", "q_trajectory_left.npz", "q_trajectory.npz"]
    for name in names:
        qt = stageb / name
        if not qt.exists():
            continue
        data = np.load(qt, allow_pickle=True)
        if "parked" in data.files and _bool(data["parked"]):
            continue
        phase = np.asarray(data["phase"])
        if bool(np.any(phase == 1) and np.any(phase == 3)):
            return True
    return False


def discover_episodes(
    src: Path,
    require_grasp_release: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Return (accepted episodes, skip records)."""
    episodes: list[dict] = []
    skipped: list[dict] = []
    demo_dirs = sorted(
        (d for d in src.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not demo_dirs:
        raise SystemExit(f"no integer demo dirs under {src}")

    for d in demo_dirs:
        retarget = d / "retarget_processor"
        qr_path = retarget / "quality_report.npz"
        npz_path = retarget / "training_data.npz"
        video_path = retarget / "video_overlay.mkv"
        narration_path = d / "narration.csv"

        if not qr_path.exists():
            skipped.append(dict(demo=d.name, reason="no quality_report.npz"))
            continue
        report = np.load(qr_path, allow_pickle=True)
        if not _bool(report["accept"]):
            why = "; ".join(_reasons(report)) or "quality gate reject"
            skipped.append(dict(demo=d.name, reason=f"quality_gate: {why}"))
            continue
        if not npz_path.exists():
            skipped.append(dict(demo=d.name, reason="accept=True but no training_data.npz"))
            continue
        if not video_path.exists():
            skipped.append(dict(demo=d.name, reason="accept=True but no video_overlay.mkv"))
            continue
        if not narration_path.exists():
            skipped.append(dict(demo=d.name, reason="no narration.csv"))
            continue
        if require_grasp_release and not _has_grasp_and_release(d):
            skipped.append(dict(demo=d.name, reason="missing grasp/release phase"))
            continue

        episodes.append(dict(
            demo_dir=d,
            npz_path=npz_path,
            video_path=video_path,
            narration_path=narration_path,
            quality_path=qr_path,
        ))
    return episodes, skipped


def build_info(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    width: int,
    height: int,
    robot_type: str,
) -> dict:
    total_chunks = max(1, -(-total_episodes // CHUNK_SIZE))
    return {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes,
        "total_chunks": total_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [len(STATE_NAMES)],
                "names": [STATE_NAMES],
            },
            "action": {
                "dtype": "float32",
                "shape": [len(STATE_NAMES)],
                "names": [STATE_NAMES],
            },
            EE_KEY: {
                "dtype": "float32",
                "shape": [len(EE_NAMES)],
                "names": [list(EE_NAMES)],
            },
            EE_ACTION_KEY: {
                "dtype": "float32",
                "shape": [len(EE_NAMES)],
                "names": [list(EE_NAMES)],
            },
            CAM_KEY: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export quality-gated retarget_processor demos to LeRobot v2.1",
    )
    parser.add_argument("--src", type=Path, required=True,
                        help="processed task dir, e.g. .../egodex_basic_pick_place")
    parser.add_argument("--dst", type=Path, required=True,
                        help="LeRobot dataset output dir (will be created)")
    parser.add_argument(
        "--allow-partial-frames",
        action="store_true",
        help="keep episodes that contain invalid frames (default: skip them)",
    )
    parser.add_argument(
        "--require-grasp-release",
        action="store_true",
        help="also skip demos whose Stage B phase has no grasp (1) and release (3)",
    )
    parser.add_argument(
        "--robot-type",
        default="panda_single_arm",
        help="written to meta/info.json (default: panda_single_arm)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list accepted/skipped demos and exit without writing --dst",
    )
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"src is not a directory: {src}")

    episodes, skipped = discover_episodes(src, require_grasp_release=args.require_grasp_release)
    print(f"discovered {len(episodes)} accepted demo(s), skipped {len(skipped)} under {src}")
    for rec in skipped:
        print(f"  skip demo {rec['demo']}: {rec['reason']}")
    if args.dry_run:
        print("dry-run: not writing", dst)
        print(f"would export: {' '.join(ep['demo_dir'].name for ep in episodes) or '(none)'}")
        return
    if not episodes:
        raise SystemExit(f"no quality-gated retarget episodes found under {src}")

    encode_video_and_sample, image_stats, vec_stats, pd = _export_helpers()

    data_dir = dst / "data"
    videos_dir = dst / "videos"
    meta_dir = dst / "meta"
    for d in (data_dir, videos_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    task_to_index: dict[str, int] = {}
    episodes_meta = []
    episodes_stats = []
    global_index = 0
    total_frames = 0
    video_width = video_height = None
    episode_index = 0
    export_failed: list[dict] = []

    for ep in episodes:
        demo_id = ep["demo_dir"].name
        state, action, ee, ee_action, valid = load_state_action(ep["npz_path"])
        n_frames = state.shape[0]
        if n_frames == 0:
            export_failed.append(dict(demo=demo_id, reason="no valid frames"))
            print(f"skip demo {demo_id}: no valid frames")
            continue
        if not args.allow_partial_frames and not valid.all():
            n_invalid = int((~valid).sum())
            export_failed.append(dict(demo=demo_id, reason=f"{n_invalid} invalid frame(s)"))
            print(f"skip demo {demo_id}: {n_invalid} invalid frame(s)")
            continue

        chunk = episode_index // CHUNK_SIZE
        chunk_dir = f"chunk-{chunk:03d}"
        video_out = videos_dir / chunk_dir / CAM_KEY / f"episode_{episode_index:06d}.mp4"
        samples, width, height = encode_video_and_sample(ep["video_path"], valid, video_out)
        if video_width is None:
            video_width, video_height = width, height
        elif (width, height) != (video_width, video_height):
            print(f"warning: demo {demo_id} resolution {width}x{height} != {video_width}x{video_height}")

        narration = read_narration(ep["narration_path"])
        if narration not in task_to_index:
            task_to_index[narration] = len(task_to_index)
        task_index = task_to_index[narration]

        df = pd.DataFrame({
            "observation.state": [row.tolist() for row in state],
            "action": [row.tolist() for row in action],
            EE_KEY: [row.tolist() for row in ee],
            EE_ACTION_KEY: [row.tolist() for row in ee_action],
            "timestamp": (np.arange(n_frames, dtype=np.float32) / FPS),
            "frame_index": np.arange(n_frames, dtype=np.int64),
            "episode_index": np.full(n_frames, episode_index, dtype=np.int64),
            "index": np.arange(global_index, global_index + n_frames, dtype=np.int64),
            "task_index": np.full(n_frames, task_index, dtype=np.int64),
        })
        parquet_out = data_dir / chunk_dir / f"episode_{episode_index:06d}.parquet"
        parquet_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_out, index=False)

        episodes_meta.append({
            "episode_index": episode_index,
            "demo_id": demo_id,
            "tasks": [narration],
            "length": n_frames,
        })

        stats = {
            "observation.state": vec_stats(state),
            "action": vec_stats(action),
            EE_KEY: vec_stats(ee),
            EE_ACTION_KEY: vec_stats(ee_action),
            "timestamp": vec_stats(df["timestamp"].to_numpy()),
            "frame_index": vec_stats(df["frame_index"].to_numpy()),
            "episode_index": vec_stats(df["episode_index"].to_numpy()),
            "index": vec_stats(df["index"].to_numpy()),
            "task_index": vec_stats(df["task_index"].to_numpy()),
        }
        if samples:
            stats[CAM_KEY] = image_stats(samples)
        episodes_stats.append({"episode_index": episode_index, "stats": stats})

        global_index += n_frames
        total_frames += n_frames
        print(f"episode {episode_index}: demo {demo_id} -> {n_frames} frames, task={narration!r}")
        episode_index += 1

    if not episodes_meta:
        raise SystemExit("all accepted demos were skipped at export time")

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task, idx in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")

    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for e in episodes_stats:
            f.write(json.dumps(e) + "\n")

    with open(meta_dir / "skipped.jsonl", "w") as f:
        for rec in skipped + export_failed:
            f.write(json.dumps(rec) + "\n")

    info = build_info(
        len(episodes_meta), total_frames, len(task_to_index),
        video_width, video_height, args.robot_type,
    )
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(
        f"done: {len(episodes_meta)} episodes, {total_frames} frames -> {dst} "
        f"(skipped {len(skipped) + len(export_failed)})"
    )


if __name__ == "__main__":
    main()
