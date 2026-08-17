#!/usr/bin/env python3
"""Write EgoDex GT 3D hands into Phantom ``hand_data_{left,right}.npz``.

IntentProcessor reads camera-frame ``kpts_3d``. EgoDex stores world-frame joint
transforms; we convert with OpenCV ``T_w2c`` (same convention as the DA3 depth
script). Existing HaMeR files are copied to ``hand_data_hamer_{side}.npz``.

Example:
    python b/export_egodex_hand_gt.py --task basic_pick_place --demo 0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ARKIT_SUFFIXES = [
    "Hand",
    "ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip",
    "IndexFingerKnuckle", "IndexFingerIntermediateBase", "IndexFingerIntermediateTip", "IndexFingerTip",
    "MiddleFingerKnuckle", "MiddleFingerIntermediateBase", "MiddleFingerIntermediateTip", "MiddleFingerTip",
    "RingFingerKnuckle", "RingFingerIntermediateBase", "RingFingerIntermediateTip", "RingFingerTip",
    "LittleFingerKnuckle", "LittleFingerIntermediateBase", "LittleFingerIntermediateTip", "LittleFingerTip",
]
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="basic_pick_place")
    p.add_argument("--demo", type=int, default=0)
    p.add_argument("--egodex-root", type=Path, default=Path("/home/a26160/DATA/test"))
    p.add_argument("--processed-root", type=Path, default=Path("/home/a26160/DATA/test_phantom_processed"))
    p.add_argument("--gt-conf-thresh", type=float, default=0.5)
    p.add_argument("--camera-convention", choices=("opencv", "opengl"), default="opencv")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def arkit_names(side: str) -> List[str]:
    return [f"{side}{s}" for s in ARKIT_SUFFIXES]


def load_hdf5(path: Path, conf_thresh: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    import h5py

    with h5py.File(path, "r") as f:
        K = np.asarray(f["camera/intrinsic"][:], dtype=np.float64)
        T_c2w = np.asarray(f["transforms/camera"][:], dtype=np.float64)
        n = len(T_c2w)
        gt_world, gt_valid = {}, {}
        for side in ("left", "right"):
            names = arkit_names(side)
            joints = np.stack([f[f"transforms/{nm}"][:, :3, 3] for nm in names], axis=1)
            gt_world[side] = np.asarray(joints, dtype=np.float64)
            key = f"confidences/{side}Hand"
            if key in f:
                gt_valid[side] = np.asarray(f[key][:], dtype=np.float64) >= conf_thresh
            else:
                gt_valid[side] = np.ones(n, dtype=bool)
    return K, T_c2w, gt_world, gt_valid


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nkj->nki", T[:, :3, :3], pts) + T[:, None, :3, 3]


def project(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    uvw = pts_cam @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[..., :2] / uvw[..., 2:3]
    uv[~np.isfinite(uv)] = 0.0
    return uv


def write_hand_npz(path: Path, kpts_cam: np.ndarray, kpts_2d: np.ndarray, detected: np.ndarray) -> None:
    n = len(detected)
    np.savez_compressed(
        path,
        hand_detected=detected.astype(bool),
        kpts_2d=kpts_2d.astype(np.float64),
        kpts_3d=kpts_cam.astype(np.float64),
        frame_indices=np.arange(n, dtype=np.int64),
        source=np.array("egodex_gt"),
    )


def main() -> None:
    args = parse_args()
    hdf5_path = args.egodex_root / args.task / f"{args.demo}.hdf5"
    demo_dir = args.processed_root / f"egodex_{args.task}" / str(args.demo)
    hand_dir = demo_dir / "hand_processor"
    if not hdf5_path.is_file():
        raise SystemExit(f"missing {hdf5_path}")
    hand_dir.mkdir(parents=True, exist_ok=True)

    K, T_c2w, gt_world, gt_valid = load_hdf5(hdf5_path, args.gt_conf_thresh)
    if args.camera_convention == "opengl":
        T_c2w = T_c2w @ GL_TO_CV[None]
    T_w2c = np.linalg.inv(T_c2w)

    for side in ("left", "right"):
        out = hand_dir / f"hand_data_{side}.npz"
        backup = hand_dir / f"hand_data_hamer_{side}.npz"
        if out.exists() and not args.overwrite:
            raise SystemExit(f"{out} exists (pass --overwrite)")
        if out.exists() and not backup.exists():
            shutil.copy2(out, backup)
            print(f"backed up {out.name} -> {backup.name}")

        kpts_cam = transform_points(T_w2c, gt_world[side])
        kpts_2d = project(kpts_cam, K)
        detected = gt_valid[side] & (kpts_cam[:, 0, 2] > 0.01)
        write_hand_npz(out, kpts_cam, kpts_2d, detected)
        print(f"wrote {out}  detected={int(detected.sum())}/{len(detected)}")

    cam_out = demo_dir / "T_camera_c2w.npy"
    np.save(cam_out, T_c2w)
    print(f"wrote {cam_out}  T={len(T_c2w)}")


if __name__ == "__main__":
    main()
