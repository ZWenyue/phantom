#!/usr/bin/env python3
"""Convert LeRobot / data-juicer hand-action exports into Phantom robot_inpaint inputs.

Typical source layout (data-juicer ego_hand_action_annotation):

    output/
      frames/{demo_id}/frame_*.jpg          # optional, used for naming
      lerobot_dataset/staging/
        data/{uuid}.parquet                 # observation.state / frame_index
        videos/{uuid}.mp4                   # full-length RGB video
        meta/{uuid}.jsonl

Writes Phantom-ready demos:

    {processed_root}/{demo_name}/{i}/
      inpaint_processor/video_human_inpaint.mkv
      smoothing_processor/smoothed_actions_{hand}_single_arm.npz
      action_processor/actions_{hand}_single_arm.npz
      segmentation_processor/masks_arm.npy
      depth.npy                             # optional
    {raw_root}/{demo_name}/{i}/video_L.mp4  # stub so process_data.py can discover demos

Example:

    python b/lerobot_to_phantom_adapter.py \\
      --staging-dir /path/to/output/lerobot_dataset/staging \\
      --frames-dir /path/to/output/frames \\
      --demo-name ego_test \\
      --processed-root b/data/processed \\
      --raw-root b/data/raw \\
      --hand left \\
      --pose-frame camera

Then run Phantom:

    cd phantom
    python process_data.py demo_name=ego_test \\
      data_root_dir=../b/data/raw \\
      processed_data_root_dir=../b/data/processed \\
      mode=robot_inpaint epic=false depth_for_overlay=false \\
      square=false bimanual_setup=single_arm target_hand=left \\
      robot=Panda debug=true
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pandas is required. Install with: pip install pandas pyarrow\n"
        f"Original import error: {exc}"
    ) from exc

try:
    from scipy.spatial.transform import Rotation
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scipy is required. Install with: pip install scipy\n"
        f"Original import error: {exc}"
    ) from exc


DEFAULT_EXTRINSICS = Path(__file__).resolve().parents[1] / "phantom" / "camera" / "camera_extrinsics.json"


@dataclass
class EpisodeConvertResult:
    demo_idx: int
    uuid: str
    demo_label: str
    num_states: int
    num_video_frames: int
    out_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapter: LeRobot staging hand states -> Phantom robot_inpaint inputs"
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        required=True,
        help="Path to lerobot_dataset/staging (contains data/, videos/, meta/)",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional frames/ dir used to name demos (e.g. 1018, 1034)",
    )
    parser.add_argument("--demo-name", type=str, default="ego_test", help="Phantom demo_name")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "processed",
        help="Root for processed Phantom demos",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "raw",
        help="Root for raw stubs (numbered folders + video_L.mp4)",
    )
    parser.add_argument(
        "--hand",
        choices=["left", "right"],
        default="left",
        help="Which hand trajectory to export for single-arm overlay",
    )
    parser.add_argument(
        "--pose-frame",
        choices=["camera", "robot", "world"],
        default="camera",
        help="Coordinate frame of observation.state xyz/rpy. "
        "'camera' applies T_cam2robot from --extrinsics; "
        "'robot'/'world' keep poses as-is.",
    )
    parser.add_argument(
        "--extrinsics",
        type=Path,
        default=DEFAULT_EXTRINSICS,
        help="Phantom-style camera_extrinsics.json used when --pose-frame=camera",
    )
    parser.add_argument(
        "--euler-order",
        type=str,
        default="xyz",
        help="scipy Euler order for state roll/pitch/yaw (default: xyz)",
    )
    parser.add_argument(
        "--max-gripper-width",
        type=float,
        default=0.08,
        help="Map LeRobot gripper in [-1,1] to meters: width=(g+1)/2*max",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Optional dir of per-frame depth_*.npy or a single depth video npy. "
        "If unset, depth.npy is skipped (use depth_for_overlay=false).",
    )
    parser.add_argument(
        "--uuids",
        nargs="*",
        default=None,
        help="Optional subset of episode uuids to convert",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing demo output folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without writing files",
    )
    return parser.parse_args()


def load_extrinsics_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Extrinsics file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        camera_extrinsics = json.load(f)
    cam_base_pos = np.asarray(camera_extrinsics[0]["camera_base_pos"], dtype=np.float64)
    cam_base_ori = np.asarray(camera_extrinsics[0]["camera_base_ori"], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = cam_base_pos
    T[:3, :3] = cam_base_ori.reshape(3, 3)
    return T


def gripper_to_width(gripper: np.ndarray, max_width: float) -> np.ndarray:
    """Map LeRobot gripper [-1 closed, +1 open] to meters."""
    g = np.clip(np.asarray(gripper, dtype=np.float64), -1.0, 1.0)
    return ((g + 1.0) / 2.0) * float(max_width)


def states_to_phantom_actions(
    states: np.ndarray,
    *,
    pose_frame: str,
    T_cam2robot: Optional[np.ndarray],
    euler_order: str,
    max_gripper_width: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (T,8) states -> ee_pts (T,3), ee_oris (T,3,3), ee_widths (T,)."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 8:
        raise ValueError(f"Expected states shape (T,8+), got {states.shape}")

    xyz = states[:, 0:3]
    rpy = states[:, 3:6]
    gripper = states[:, 7]
    ee_oris = Rotation.from_euler(euler_order, rpy).as_matrix()

    if pose_frame == "camera":
        if T_cam2robot is None:
            raise ValueError("T_cam2robot is required when pose_frame=camera")
        ones = np.ones((len(xyz), 1), dtype=np.float64)
        xyz_h = np.concatenate([xyz, ones], axis=1)  # (T,4)
        ee_pts = (T_cam2robot @ xyz_h.T).T[:, :3]
        R_c2r = T_cam2robot[:3, :3]
        ee_oris = np.einsum("ij,tjk->tik", R_c2r, ee_oris)
    else:
        # robot / world: already expressed in the frame Phantom should track
        ee_pts = xyz.copy()

    ee_widths = gripper_to_width(gripper, max_gripper_width)
    return ee_pts.astype(np.float64), ee_oris.astype(np.float64), ee_widths.astype(np.float64)


def read_episode_parquet(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to read {path}. Install pyarrow (pip install pyarrow). Detail: {exc}"
        ) from exc

    if "observation.state" not in df.columns:
        raise KeyError(f"{path} missing observation.state column")
    if "frame_index" not in df.columns:
        raise KeyError(f"{path} missing frame_index column")

    states = np.stack([np.asarray(x, dtype=np.float64) for x in df["observation.state"].to_list()])
    frame_index = np.asarray(df["frame_index"].to_list(), dtype=np.int64)
    return states, frame_index


def count_video_frames(video_path: Path) -> Tuple[int, int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n <= 0 or w <= 0 or h <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}: n={n}, w={w}, h={h}")
    return n, h, w


def read_first_rgb_frame(video_path: Path) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    return frame[:, :, ::-1]  # BGR -> RGB


def ensure_mkv(src_mp4: Path, dst_mkv: Path) -> None:
    """Remux/copy mp4 into mkv. Falls back to plain copy if ffmpeg is unavailable."""
    dst_mkv.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.copy2(src_mp4, dst_mkv)
        print(f"  [warn] ffmpeg not found; copied {src_mp4.name} -> {dst_mkv.name}")
        return

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src_mp4),
        "-c",
        "copy",
        str(dst_mkv),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not dst_mkv.exists():
        # Fallback: re-encode with a widely available codec
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(src_mp4),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dst_mkv),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0 or not dst_mkv.exists():
            raise RuntimeError(
                f"ffmpeg failed to write {dst_mkv}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )


def match_uuid_to_frame_demo(
    video_path: Path,
    frames_dir: Optional[Path],
) -> str:
    """Best-effort map episode video -> frames/<demo_id> via first-frame similarity."""
    if frames_dir is None or not frames_dir.exists():
        return video_path.stem

    from PIL import Image

    ref = read_first_rgb_frame(video_path).astype(np.float32)
    ref_small = _resize_rgb(ref, 64, 64)
    best_name = video_path.stem
    best_score = float("inf")

    for demo_dir in sorted([p for p in frames_dir.iterdir() if p.is_dir()]):
        frame0 = demo_dir / "frame_0.jpg"
        if not frame0.exists():
            # try sorted first jpg
            jpgs = sorted(demo_dir.glob("*.jpg"))
            if not jpgs:
                continue
            frame0 = jpgs[0]
        img = np.asarray(Image.open(frame0).convert("RGB"), dtype=np.float32)
        img_small = _resize_rgb(img, 64, 64)
        score = float(np.mean(np.abs(ref_small - img_small)))
        if score < best_score:
            best_score = score
            best_name = demo_dir.name

    return best_name


def _resize_rgb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def discover_episodes(staging_dir: Path, uuids: Optional[Sequence[str]]) -> List[Tuple[str, Path, Path]]:
    data_dir = staging_dir / "data"
    video_dir = staging_dir / "videos"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Missing videos dir: {video_dir}")

    episodes = []
    for parquet_path in sorted(data_dir.glob("*.parquet")):
        uuid = parquet_path.stem
        if uuids is not None and uuid not in set(uuids):
            continue
        video_path = video_dir / f"{uuid}.mp4"
        if not video_path.exists():
            print(f"[skip] no video for uuid={uuid}")
            continue
        episodes.append((uuid, parquet_path, video_path))
    if not episodes:
        raise FileNotFoundError(f"No convertible episodes found under {staging_dir}")
    return episodes


def write_dummy_masks(path: Path, num_frames: int, height: int, width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = np.zeros((num_frames, height, width), dtype=np.bool_)
    np.save(path, masks)


def maybe_write_depth(
    depth_dir: Optional[Path],
    demo_label: str,
    out_path: Path,
    num_frames: int,
    height: int,
    width: int,
) -> bool:
    if depth_dir is None:
        return False
    if not depth_dir.exists():
        print(f"  [warn] depth dir not found: {depth_dir}")
        return False

    # Prefer a single stacked array named after demo, else per-frame files.
    candidates = [
        depth_dir / f"{demo_label}.npy",
        depth_dir / f"depth_{demo_label}.npy",
        depth_dir / "depth.npy",
    ]
    for cand in candidates:
        if cand.exists():
            depth = np.load(cand)
            if depth.ndim == 2:
                depth = np.repeat(depth[None, ...], num_frames, axis=0)
            if depth.shape[0] < num_frames:
                raise ValueError(f"Depth {cand} has {depth.shape[0]} frames, need >= {num_frames}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, depth[:num_frames].astype(np.float32))
            return True

    # Per-frame depth_000.npy style
    frame_files = sorted(depth_dir.glob("*.npy"))
    if not frame_files:
        print(f"  [warn] no depth npy files in {depth_dir}")
        return False

    # If directory contains many unrelated hashes (moge_arrays), skip auto use
    if len(frame_files) > num_frames * 2 and demo_label not in "".join(p.name for p in frame_files[:5]):
        print(
            f"  [warn] depth dir looks like a shared hash dump ({len(frame_files)} files); "
            "skipping auto depth. Pass a per-demo depth array if needed."
        )
        return False

    depths = []
    for i in range(num_frames):
        exact = depth_dir / f"depth_{i:04d}.npy"
        alt = depth_dir / f"{i}.npy"
        path = exact if exact.exists() else alt
        if not path.exists():
            print(f"  [warn] missing depth frame {i}, skipping depth export")
            return False
        depths.append(np.load(path))
    stacked = np.stack(depths).astype(np.float32)
    if stacked.shape[1:] != (height, width):
        print(
            f"  [warn] depth shape {stacked.shape[1:]} != video {(height, width)}; "
            "still writing depth.npy"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, stacked)
    return True


def convert_one_episode(
    *,
    demo_idx: int,
    uuid: str,
    parquet_path: Path,
    video_path: Path,
    demo_label: str,
    args: argparse.Namespace,
    T_cam2robot: Optional[np.ndarray],
) -> EpisodeConvertResult:
    states, frame_index = read_episode_parquet(parquet_path)
    num_video_frames, height, width = count_video_frames(video_path)

    if frame_index.min() < 0 or frame_index.max() >= num_video_frames:
        raise ValueError(
            f"uuid={uuid}: frame_index out of range for video "
            f"(max={frame_index.max()}, video_frames={num_video_frames})"
        )
    if len(states) != len(frame_index):
        raise ValueError(f"uuid={uuid}: states/frame_index length mismatch")

    ee_pts, ee_oris, ee_widths = states_to_phantom_actions(
        states,
        pose_frame=args.pose_frame,
        T_cam2robot=T_cam2robot,
        euler_order=args.euler_order,
        max_gripper_width=args.max_gripper_width,
    )
    union_indices = frame_index.astype(np.int64)

    processed_demo = args.processed_root / args.demo_name / str(demo_idx)
    raw_demo = args.raw_root / args.demo_name / str(demo_idx)

    print(
        f"[{demo_idx}] uuid={uuid} label={demo_label} "
        f"states={len(states)} video_frames={num_video_frames} -> {processed_demo}"
    )

    if args.dry_run:
        return EpisodeConvertResult(
            demo_idx=demo_idx,
            uuid=uuid,
            demo_label=demo_label,
            num_states=len(states),
            num_video_frames=num_video_frames,
            out_dir=processed_demo,
        )

    if processed_demo.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: {processed_demo} (pass --overwrite to replace)"
            )
        shutil.rmtree(processed_demo)
    if raw_demo.exists() and args.overwrite:
        shutil.rmtree(raw_demo)

    # Raw stub for process_data.py discovery
    raw_demo.mkdir(parents=True, exist_ok=True)
    raw_video = raw_demo / "video_L.mp4"
    if not raw_video.exists():
        shutil.copy2(video_path, raw_video)

    # Processed tree
    inpaint_dir = processed_demo / "inpaint_processor"
    smooth_dir = processed_demo / "smoothing_processor"
    action_dir = processed_demo / "action_processor"
    seg_dir = processed_demo / "segmentation_processor"
    for d in (inpaint_dir, smooth_dir, action_dir, seg_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Keep a copy of source video at demo root as well (handy for debugging)
    shutil.copy2(video_path, processed_demo / "video_L.mp4")

    human_inpaint = inpaint_dir / "video_human_inpaint.mkv"
    ensure_mkv(video_path, human_inpaint)

    hand = args.hand
    smooth_path = smooth_dir / f"smoothed_actions_{hand}_single_arm.npz"
    actions_path = action_dir / f"actions_{hand}_single_arm.npz"
    np.savez(smooth_path, ee_pts=ee_pts, ee_oris=ee_oris, ee_widths=ee_widths)
    np.savez(
        actions_path,
        union_indices=union_indices,
        ee_pts=ee_pts,
        ee_oris=ee_oris,
        ee_widths=ee_widths,
    )

    write_dummy_masks(seg_dir / "masks_arm.npy", num_video_frames, height, width)
    maybe_write_depth(
        args.depth_dir,
        demo_label,
        processed_demo / "depth.npy",
        num_video_frames,
        height,
        width,
    )

    # Sidecar metadata for traceability
    meta = {
        "uuid": uuid,
        "demo_label": demo_label,
        "hand": hand,
        "pose_frame": args.pose_frame,
        "euler_order": args.euler_order,
        "max_gripper_width": args.max_gripper_width,
        "num_states": int(len(states)),
        "num_video_frames": int(num_video_frames),
        "union_indices": union_indices.tolist(),
        "source_parquet": str(parquet_path),
        "source_video": str(video_path),
        "extrinsics": str(args.extrinsics) if args.pose_frame == "camera" else None,
    }
    with open(processed_demo / "adapter_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return EpisodeConvertResult(
        demo_idx=demo_idx,
        uuid=uuid,
        demo_label=demo_label,
        num_states=len(states),
        num_video_frames=num_video_frames,
        out_dir=processed_demo,
    )


def main() -> None:
    args = parse_args()
    staging_dir = args.staging_dir.resolve()
    args.processed_root = args.processed_root.resolve()
    args.raw_root = args.raw_root.resolve()

    T_cam2robot = None
    if args.pose_frame == "camera":
        T_cam2robot = load_extrinsics_matrix(args.extrinsics.resolve())
        print(f"Using T_cam2robot from {args.extrinsics}")
    else:
        print(f"Keeping poses in {args.pose_frame} frame (no extrinsics applied)")

    episodes = discover_episodes(staging_dir, args.uuids)
    results: List[EpisodeConvertResult] = []
    mapping: Dict[str, Dict[str, object]] = {}

    for demo_idx, (uuid, parquet_path, video_path) in enumerate(episodes):
        demo_label = match_uuid_to_frame_demo(video_path, args.frames_dir)
        result = convert_one_episode(
            demo_idx=demo_idx,
            uuid=uuid,
            parquet_path=parquet_path,
            video_path=video_path,
            demo_label=demo_label,
            args=args,
            T_cam2robot=T_cam2robot,
        )
        results.append(result)
        mapping[str(demo_idx)] = {
            "uuid": uuid,
            "demo_label": demo_label,
            "num_states": result.num_states,
            "out_dir": str(result.out_dir),
        }

    if not args.dry_run:
        map_path = args.processed_root / args.demo_name / "adapter_index.json"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        print(f"\nWrote index: {map_path}")

    print("\nDone. Next (from phantom/):")
    print(
        "  python process_data.py"
        f" demo_name={args.demo_name}"
        f" data_root_dir={args.raw_root}"
        f" processed_data_root_dir={args.processed_root}"
        " mode=robot_inpaint epic=false depth_for_overlay=false"
        f" square=false bimanual_setup=single_arm target_hand={args.hand}"
        " robot=Panda debug=true"
    )
    print(
        "\nNotes:\n"
        "  - If the robot appears in the wrong place, fix --extrinsics / --pose-frame.\n"
        "  - Gripper widths are mapped from [-1,1] -> [0, max_gripper_width].\n"
        "  - masks_arm.npy is all-zeros (fine when depth_for_overlay=false)."
    )


if __name__ == "__main__":
    main()
