#!/usr/bin/env python3
"""Convert EgoDex HDF5 hand-action data into Phantom pipeline inputs.

EgoDex source layout (per task):
    /mnt/r/DATA/EgoDex/test/<task>/
        0.hdf5   0.mp4
        1.hdf5   1.mp4
        ...

Each HDF5 contains:
    camera/intrinsic         (3,3)  - camera intrinsic matrix
    transforms/camera        (N,4,4) - per-frame camera-to-world
    transforms/leftHand      (N,4,4) - left hand world transform
    transforms/rightHand     (N,4,4) - right hand world transform
    transforms/<finger_joints> (N,4,4) - all finger joint transforms
    confidences/<joint>      (N,) - per-joint confidence

Writes Phantom-ready demos:
    {output_root}/egodex_{task}/
        convert_meta.json     -> scale + hydra camera overrides (if scaled)
        0/
            video_L.mp4       -> symlink, or re-encoded if --scale/--height
            hand_det.pkl      -> generated from 3D projections
        1/
            ...

Example:
    python b/convert_egodex.py \
        --task basic_pick_place \
        --egodex-root /mnt/r/DATA/EgoDex/test \
        --output-root b/data/raw \
        --max-episodes 10

Then run Phantom:
    cd phantom
    python process_data.py \
        --config-path=../b/configs --config-name=egodex \
        demo_name=egodex_basic_pick_place mode=all
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise SystemExit(f"h5py is required: {exc}") from exc

from epic_kitchens.hoa.types import HandDetection, BBox, HandSide, HandState, FloatVector


# ---------------------------------------------------------------------------
# Joint names per hand side
# ---------------------------------------------------------------------------

FINGER_JOINTS = {
    "left": [
        "leftHand",
        "leftLittleFingerMetacarpal", "leftLittleFingerKnuckle",
        "leftLittleFingerIntermediateBase", "leftLittleFingerIntermediateTip", "leftLittleFingerTip",
        "leftRingFingerMetacarpal", "leftRingFingerKnuckle",
        "leftRingFingerIntermediateBase", "leftRingFingerIntermediateTip", "leftRingFingerTip",
        "leftMiddleFingerMetacarpal", "leftMiddleFingerKnuckle",
        "leftMiddleFingerIntermediateBase", "leftMiddleFingerIntermediateTip", "leftMiddleFingerTip",
        "leftIndexFingerMetacarpal", "leftIndexFingerKnuckle",
        "leftIndexFingerIntermediateBase", "leftIndexFingerIntermediateTip", "leftIndexFingerTip",
        "leftThumbKnuckle", "leftThumbIntermediateBase", "leftThumbIntermediateTip", "leftThumbTip",
    ],
    "right": [
        "rightHand",
        "rightLittleFingerMetacarpal", "rightLittleFingerKnuckle",
        "rightLittleFingerIntermediateBase", "rightLittleFingerIntermediateTip", "rightLittleFingerTip",
        "rightRingFingerMetacarpal", "rightRingFingerKnuckle",
        "rightRingFingerIntermediateBase", "rightRingFingerIntermediateTip", "rightRingFingerTip",
        "rightMiddleFingerMetacarpal", "rightMiddleFingerKnuckle",
        "rightMiddleFingerIntermediateBase", "rightMiddleFingerIntermediateTip", "rightMiddleFingerTip",
        "rightIndexFingerMetacarpal", "rightIndexFingerKnuckle",
        "rightIndexFingerIntermediateBase", "rightIndexFingerIntermediateTip", "rightIndexFingerTip",
        "rightThumbKnuckle", "rightThumbIntermediateBase", "rightThumbIntermediateTip", "rightThumbTip",
    ],
}

BBOX_PADDING_RATIO = 0.15
NATIVE_W = 1920
NATIVE_H = 1080
CAMERA_JSON_SRC = (
    Path(__file__).resolve().parent.parent / "phantom" / "camera" / "camera_intrinsics_egodex.json"
)


def even(n: int) -> int:
    n = max(2, int(n))
    return n - (n % 2)


def probe_video_wh(path: Path) -> Tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        w_s, h_s = out.split("x")
        return int(w_s), int(h_s)
    except Exception:
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return NATIVE_W, NATIVE_H


def resolve_target_wh(
    src_w: int,
    src_h: int,
    scale: Optional[float],
    height: Optional[int],
) -> Tuple[int, int, float, float]:
    if scale is not None and height is not None:
        raise ValueError("pass only one of --scale or --height")
    if height is not None:
        if height <= 0:
            raise ValueError("--height must be positive")
        dst_h = even(height)
        sy = dst_h / float(src_h)
        dst_w = even(round(src_w * sy))
        sx = dst_w / float(src_w)
        return dst_w, dst_h, sx, sy
    if scale is None or abs(scale - 1.0) < 1e-9:
        return src_w, src_h, 1.0, 1.0
    if scale <= 0:
        raise ValueError("--scale must be positive")
    dst_w = even(round(src_w * scale))
    dst_h = even(round(src_h * scale))
    return dst_w, dst_h, dst_w / float(src_w), dst_h / float(src_h)


def scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    Ks = np.asarray(K, dtype=np.float64).copy()
    Ks[0, 0] *= sx
    Ks[1, 1] *= sy
    Ks[0, 2] *= sx
    Ks[1, 2] *= sy
    return Ks


def write_scaled_camera_json(
    src: Path,
    dst: Path,
    sx: float,
    sy: float,
) -> None:
    with open(src, "r") as f:
        data = json.load(f)
    for side in data:
        if not isinstance(data[side], dict):
            continue
        for key in ("fx", "fy", "cx", "cy"):
            if key in data[side]:
                s = sx if key in ("fx", "cx") else sy
                data[side][key] = float(data[side][key]) * s
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def encode_scaled_video(src: Path, dst: Path, width: int, height: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def project_world_to_image(
    points_world: np.ndarray,
    T_c2w: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Project 3D world points to 2D image coordinates.

    Args:
        points_world: (N, 3) world coordinates
        T_c2w: (4, 4) camera-to-world transform
        K: (3, 3) camera intrinsic matrix

    Returns:
        (N, 2) pixel coordinates [u, v]
    """
    T_w2c = np.linalg.inv(T_c2w)
    ones = np.ones((len(points_world), 1))
    pts_h = np.concatenate([points_world, ones], axis=1)  # (N, 4)
    pts_cam = (T_w2c @ pts_h.T).T[:, :3]  # (N, 3)

    # Filter out points behind camera
    valid = pts_cam[:, 2] > 0.01
    uv = np.zeros((len(points_world), 2))
    if valid.any():
        pts_valid = pts_cam[valid]
        projected = (K @ pts_valid.T).T  # (M, 3)
        uv[valid, 0] = projected[:, 0] / projected[:, 2]
        uv[valid, 1] = projected[:, 1] / projected[:, 2]
    return uv, valid


def compute_bbox_from_joints(
    uv: np.ndarray,
    valid: np.ndarray,
    img_w: int,
    img_h: int,
    padding_ratio: float = BBOX_PADDING_RATIO,
) -> Optional[Tuple[float, float, float, float]]:
    """Compute normalized bounding box from projected joint positions.

    Returns:
        (left, top, right, bottom) in [0, 1] range, or None if no valid points.
    """
    if not valid.any():
        return None

    uv_valid = uv[valid]
    x_min, y_min = uv_valid.min(axis=0)
    x_max, y_max = uv_valid.max(axis=0)

    w = x_max - x_min
    h = y_max - y_min
    size = max(w, h, 30.0)  # minimum 30px bbox

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    half = size / 2 * (1 + padding_ratio)

    left = max(0, cx - half) / img_w
    top = max(0, cy - half) / img_h
    right = min(img_w, cx + half) / img_w
    bottom = min(img_h, cy + half) / img_h

    if right <= left or bottom <= top:
        return None

    return (left, top, right, bottom)


def generate_hand_det_pkl(
    hdf5_path: str,
    img_w: int = 1920,
    img_h: int = 1080,
    sx: float = 1.0,
    sy: float = 1.0,
) -> dict:
    """Generate hand_det.pkl data from EgoDex HDF5 file.

    Returns:
        defaultdict keyed by frame index (as str), values are lists of HandDetection.
    """
    hand_det = defaultdict(list)

    with h5py.File(hdf5_path, "r") as f:
        K = scale_K(f["camera/intrinsic"][:], sx, sy)
        T_c2w_all = f["transforms/camera"][:]
        n_frames = T_c2w_all.shape[0]

        # Pre-load all joint transforms
        joint_transforms = {}
        for side in ["left", "right"]:
            for joint_name in FINGER_JOINTS[side]:
                key = f"transforms/{joint_name}"
                if key in f:
                    joint_transforms[joint_name] = f[key][:]

        # Pre-load confidences for hand joints
        hand_confidences = {}
        for side_name in ["leftHand", "rightHand"]:
            conf_key = f"confidences/{side_name}"
            if conf_key in f:
                hand_confidences[side_name] = f[conf_key][:]

    for frame_idx in range(n_frames):
        T_c2w = T_c2w_all[frame_idx]

        for side in ["left", "right"]:
            # Collect 3D positions of all joints for this hand
            points_3d = []
            for joint_name in FINGER_JOINTS[side]:
                if joint_name in joint_transforms:
                    pos = joint_transforms[joint_name][frame_idx, :3, 3]
                    points_3d.append(pos)

            if len(points_3d) < 5:
                continue

            points_3d = np.array(points_3d)

            # Check confidence
            conf_key = f"{side}Hand"
            if conf_key in hand_confidences:
                conf = hand_confidences[conf_key][frame_idx]
                if conf < 0.3:
                    continue

            # Project to image
            uv, valid = project_world_to_image(points_3d, T_c2w, K)
            bbox = compute_bbox_from_joints(uv, valid, img_w, img_h)

            if bbox is None:
                continue

            left, top, right, bottom = bbox

            det = HandDetection(
                bbox=BBox(left=left, top=top, right=right, bottom=bottom),
                score=np.float32(1.0),
                state=HandState.NO_CONTACT,
                side=HandSide.LEFT if side == "left" else HandSide.RIGHT,
                object_offset=FloatVector(np.float32(0.0), np.float32(0.0)),
            )
            hand_det[str(frame_idx)].append(det)

    return hand_det


def convert_one_episode(
    hdf5_path: Path,
    video_path: Path,
    output_dir: Path,
    img_w: int = 1920,
    img_h: int = 1080,
    sx: float = 1.0,
    sy: float = 1.0,
    scale_video: bool = False,
) -> bool:
    """Convert a single EgoDex episode to Phantom format.

    Returns:
        True if successful, False if skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    video_link = output_dir / "video_L.mp4"
    if video_link.exists() or video_link.is_symlink():
        video_link.unlink()
    if scale_video:
        encode_scaled_video(video_path, video_link, img_w, img_h)
    else:
        video_link.symlink_to(video_path.resolve())

    hand_det = generate_hand_det_pkl(str(hdf5_path), img_w, img_h, sx=sx, sy=sy)
    pkl_path = output_dir / "hand_det.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(hand_det, f)

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert EgoDex HDF5 data to Phantom pipeline format"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="EgoDex task name (e.g., basic_pick_place)",
    )
    parser.add_argument(
        "--egodex-root",
        type=Path,
        default=Path("/mnt/r/DATA/EgoDex/test"),
        help="Root directory of EgoDex test data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "raw",
        help="Output root for Phantom raw data",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to convert (default: all)",
    )
    parser.add_argument(
        "--demo",
        type=int,
        default=None,
        help="Convert a single episode id (HDF5 stem / output folder name)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted episodes",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Uniform scale of source video (e.g. 0.5 → half resolution). "
             "Mutually exclusive with --height. Default: keep native, symlink.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target video height in pixels (width follows aspect, even). "
             "Mutually exclusive with --scale. Also sets Phantom input_resolution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_dir = args.egodex_root / args.task
    if not task_dir.exists():
        print(f"Error: task directory not found: {task_dir}")
        sys.exit(1)

    # Discover episodes
    hdf5_files = sorted(
        [f for f in task_dir.iterdir() if f.suffix == ".hdf5"],
        key=lambda x: int(x.stem),
    )

    if args.demo is not None:
        hdf5_files = [task_dir / f"{args.demo}.hdf5"]
        if not hdf5_files[0].is_file():
            print(f"Error: missing {hdf5_files[0]}")
            sys.exit(1)

    if not hdf5_files:
        print(f"Error: no HDF5 files found in {task_dir}")
        sys.exit(1)

    if args.max_episodes is not None:
        hdf5_files = hdf5_files[: args.max_episodes]

    demo_name = f"egodex_{args.task}"
    output_base = args.output_root / demo_name
    output_base.mkdir(parents=True, exist_ok=True)

    first_video = task_dir / f"{hdf5_files[0].stem}.mp4"
    src_w, src_h = probe_video_wh(first_video) if first_video.exists() else (NATIVE_W, NATIVE_H)
    dst_w, dst_h, sx, sy = resolve_target_wh(src_w, src_h, args.scale, args.height)
    scale_video = not (abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9)

    print(f"Converting {len(hdf5_files)} episodes from {task_dir}")
    print(f"Output: {output_base}")
    if scale_video:
        print(f"Video:  {src_w}x{src_h} → {dst_w}x{dst_h}  (sx={sx:.6f}, sy={sy:.6f})")
        cam_name = f"camera_intrinsics_egodex_{dst_h}p.json"
        cam_rel = f"camera/{cam_name}"
        if not CAMERA_JSON_SRC.is_file():
            print(f"Error: missing source intrinsics {CAMERA_JSON_SRC}")
            sys.exit(1)
        write_scaled_camera_json(CAMERA_JSON_SRC, CAMERA_JSON_SRC.parent / cam_name, sx, sy)
        write_scaled_camera_json(CAMERA_JSON_SRC, output_base / "camera_intrinsics.json", sx, sy)
        meta = {
            "scale_x": sx,
            "scale_y": sy,
            "src_wh": [src_w, src_h],
            "dst_wh": [dst_w, dst_h],
            "input_resolution": dst_h,
            "output_resolution": dst_h,
            "camera_intrinsics": cam_rel,
        }
        with open(output_base / "convert_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        print(f"Intrinsics: {CAMERA_JSON_SRC.parent / cam_name}")
    else:
        print(f"Video:  {src_w}x{src_h} (native, symlink)")
        meta_path = output_base / "convert_meta.json"
        if meta_path.exists():
            meta_path.unlink()

    converted = 0
    for idx, hdf5_path in enumerate(hdf5_files):
        episode_id = hdf5_path.stem
        video_path = task_dir / f"{episode_id}.mp4"

        if not video_path.exists():
            print(f"  [{idx}] Skipping {episode_id}: no video file")
            continue

        # --demo writes to the HDF5 stem; bulk convert keeps enumeration index
        # (legacy: 0.hdf5, 1.hdf5, … → folders 0, 1, …).
        output_dir = output_base / episode_id if args.demo is not None else output_base / str(idx)

        if output_dir.exists() and not args.overwrite:
            print(f"  [{idx}] Skipping {episode_id}: already exists")
            converted += 1
            continue

        try:
            convert_one_episode(
                hdf5_path,
                video_path,
                output_dir,
                img_w=dst_w,
                img_h=dst_h,
                sx=sx,
                sy=sy,
                scale_video=scale_video,
            )
            converted += 1

            # Print brief info
            with h5py.File(hdf5_path, "r") as f:
                n_frames = f["transforms/camera"].shape[0]
            with open(output_dir / "hand_det.pkl", "rb") as f:
                det = pickle.load(f)
            n_det_frames = len(det)
            print(f"  [{idx}] {episode_id}: {n_frames} frames, {n_det_frames} frames with hand detections")

        except Exception as e:
            print(f"  [{idx}] Error converting {episode_id}: {e}")
            continue

    print(f"\nConverted {converted}/{len(hdf5_files)} episodes")
    print(f"\nNext steps (from phantom/):")
    if scale_video:
        print(f"  Hydra overrides (also written to {output_base / 'convert_meta.json'}):")
        print(f"    input_resolution={dst_h} output_resolution={dst_h} \\")
        print(f"    camera_intrinsics=camera/camera_intrinsics_egodex_{dst_h}p.json")
        print(f"  run_hand_inpaint.sh / run_contact_retarget.sh pick these up automatically.")
    print(f"  python process_data.py \\")
    print(f"    --config-path=../b/configs --config-name=egodex \\")
    print(f"    demo_name={demo_name} mode=all")


if __name__ == "__main__":
    main()
