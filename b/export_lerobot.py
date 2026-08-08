"""
Export PHANTOM-processed EPIC-KITCHENS demos to LeRobot v2.1 format.

Reads, per episode directory under --src:
  - inpaint_processor/training_data_<bimanual_setup>.npz  (joint positions, gripper widths, valid mask)
  - video_overlay_<robot>_<bimanual_setup>.mkv             (robot-inpainted egocentric video)
  - narration.csv                                          (language instruction)

and writes a LeRobot v2.1 dataset directory (meta/, data/, videos/) under --dst.

Run in the `lerobot` conda env (needs ffmpeg, pandas, pyarrow, opencv, numpy):
    conda run -n lerobot python phantom/export_lerobot.py --src data/processed/epic --dst data/processed/epic_lerobot/epic
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FPS = 15
CHUNK_SIZE = 1000
CAM_KEY = "observation.images.cam_ego"
N_JOINTS = 7
MAX_IMAGE_SAMPLES = 100

STATE_NAMES = (
    [f"left_joint_{i}" for i in range(1, N_JOINTS + 1)] + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(1, N_JOINTS + 1)] + ["right_gripper"]
)


def discover_episodes(src: Path) -> list[dict]:
    episodes = []
    demo_dirs = sorted((d for d in src.iterdir() if d.is_dir()), key=lambda p: int(p.name))
    for d in demo_dirs:
        npz_matches = sorted((d / "inpaint_processor").glob("training_data_*.npz"))
        if not npz_matches:
            print(f"skip {d}: no training_data_*.npz found")
            continue
        npz_path = npz_matches[0]
        bimanual_setup = npz_path.stem[len("training_data_"):]
        video_matches = list(d.glob(f"video_overlay_*_{bimanual_setup}.mkv"))
        if not video_matches:
            print(f"skip {d}: no video_overlay_*_{bimanual_setup}.mkv found")
            continue
        narration_path = d / "narration.csv"
        if not narration_path.exists():
            print(f"skip {d}: no narration.csv found")
            continue
        episodes.append(dict(
            demo_dir=d,
            npz_path=npz_path,
            video_path=video_matches[0],
            narration_path=narration_path,
            bimanual_setup=bimanual_setup,
        ))
    return episodes


def read_narration(path: Path) -> str:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    data = {}
    for row in rows[1:]:
        if not row:
            continue
        data[row[0]] = ",".join(row[1:])
    return data.get("narration", "").strip()


def load_state_action(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    valid = d["valid"]
    jl = d["joint_pos_left"][valid]
    jr = d["joint_pos_right"][valid]
    gl = d["gripper_width_left"][valid]
    gr = d["gripper_width_right"][valid]
    state = np.concatenate([jl, gl[:, None], jr, gr[:, None]], axis=1).astype(np.float32)
    action = np.concatenate([state[1:], state[-1:]], axis=0)
    return state, action, valid


def encode_video_and_sample(video_path: Path, valid: np.ndarray, out_path: Path) -> tuple[list[np.ndarray], int, int]:
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    if n_valid > 0:
        step = max(1, n_valid // MAX_IMAGE_SAMPLES)
        sample_positions = set(range(0, n_valid, step)[:MAX_IMAGE_SAMPLES])
    else:
        sample_positions = set()

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(FPS),
        "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    samples = []
    valid_i = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(valid) and valid[frame_idx]:
            proc.stdin.write(frame.tobytes())
            if valid_i in sample_positions:
                samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            valid_i += 1
        frame_idx += 1
    cap.release()
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed encoding {out_path}")
    if valid_i != n_valid:
        print(f"warning: {video_path} had {valid_i} readable frames but valid mask expects {n_valid}")

    return samples, width, height


def vec_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim > 1:
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.shape[0])],
    }


def image_stats(samples: list[np.ndarray]) -> dict:
    arr = np.stack(samples).transpose(0, 3, 1, 2)  # (S, C, H, W)
    c = arr.shape[1]
    flat = arr.reshape(c, -1)

    def wrap(v):
        return [[[float(x)]] for x in v]

    return {
        "min": wrap(flat.min(axis=1)),
        "max": wrap(flat.max(axis=1)),
        "mean": wrap(flat.mean(axis=1)),
        "std": wrap(flat.std(axis=1)),
        "count": [arr.shape[0]],
    }


def build_info(total_episodes: int, total_frames: int, total_tasks: int, width: int, height: int) -> dict:
    total_chunks = max(1, -(-total_episodes // CHUNK_SIZE))
    return {
        "codebase_version": "v2.1",
        "robot_type": "kinova3_bimanual_shoulders",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("data/processed/epic"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed/epic_lerobot/epic"))
    args = parser.parse_args()

    episodes = discover_episodes(args.src)
    if not episodes:
        raise SystemExit(f"no episodes found under {args.src}")

    data_dir = args.dst / "data"
    videos_dir = args.dst / "videos"
    meta_dir = args.dst / "meta"
    for d in (data_dir, videos_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    task_to_index: dict[str, int] = {}
    episodes_meta = []
    episodes_stats = []
    global_index = 0
    total_frames = 0
    video_width = video_height = None

    for episode_index, ep in enumerate(episodes):
        chunk = episode_index // CHUNK_SIZE
        chunk_dir = f"chunk-{chunk:03d}"

        state, action, valid = load_state_action(ep["npz_path"])
        n_frames = state.shape[0]
        if n_frames == 0:
            print(f"skip {ep['demo_dir']}: no valid frames")
            continue

        video_out = videos_dir / chunk_dir / CAM_KEY / f"episode_{episode_index:06d}.mp4"
        samples, width, height = encode_video_and_sample(ep["video_path"], valid, video_out)
        if video_width is None:
            video_width, video_height = width, height
        elif (width, height) != (video_width, video_height):
            print(f"warning: {ep['video_path']} resolution {width}x{height} != {video_width}x{video_height}")

        narration = read_narration(ep["narration_path"])
        if narration not in task_to_index:
            task_to_index[narration] = len(task_to_index)
        task_index = task_to_index[narration]

        df = pd.DataFrame({
            "observation.state": [row.tolist() for row in state],
            "action": [row.tolist() for row in action],
            "timestamp": (np.arange(n_frames, dtype=np.float32) / FPS),
            "frame_index": np.arange(n_frames, dtype=np.int64),
            "episode_index": np.full(n_frames, episode_index, dtype=np.int64),
            "index": np.arange(global_index, global_index + n_frames, dtype=np.int64),
            "task_index": np.full(n_frames, task_index, dtype=np.int64),
        })
        parquet_out = data_dir / chunk_dir / f"episode_{episode_index:06d}.parquet"
        parquet_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_out, index=False)

        episodes_meta.append({"episode_index": episode_index, "tasks": [narration], "length": n_frames})

        stats = {
            "observation.state": vec_stats(state),
            "action": vec_stats(action),
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
        print(f"episode {episode_index}: {ep['demo_dir'].name} -> {n_frames} frames, task={narration!r}")

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task, idx in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")

    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for e in episodes_stats:
            f.write(json.dumps(e) + "\n")

    info = build_info(len(episodes_meta), total_frames, len(task_to_index), video_width, video_height)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"done: {len(episodes_meta)} episodes, {total_frames} frames -> {args.dst}")


if __name__ == "__main__":
    main()
