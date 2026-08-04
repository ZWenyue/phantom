#!/usr/bin/env python3
"""Verify ActionProcessor's per-frame camera extrinsics.

Checks the behaviour that matters:
  * a static camera reproduces the old fixed-extrinsic result exactly (no regression
    for tripod recordings such as the original Phantom Zed2 data),
  * a moving camera now yields a trajectory that is a single rigid transform of the
    true world-frame trajectory, which the fixed extrinsic could not do,
  * every malformed / missing / partial input falls back safely instead of producing
    silently wrong actions.

Run inside the phantom environment (imports torch via phantom.hand):

    python b/test_per_frame_extrinsics.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from phantom.processors.action_processor import ActionProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_FRAMES = 60
N_JOINTS = 21


def make_cam2robot_init() -> np.ndarray:
    """A plausible calibrated camera-to-robot transform."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.3, 0.1]).as_matrix()
    T[:3, 3] = [0.1, -0.2, 0.4]
    return T


def make_camera_track(n: int, moving: bool) -> np.ndarray:
    """(n, 4, 4) camera-to-world poses. If not moving, every pose is identical."""
    t = np.arange(n) / 30.0
    poses = np.zeros((n, 4, 4))
    for i in range(n):
        T = np.eye(4)
        if moving:
            # Wobble in place with a slow drift, like a seated Vision Pro recording.
            T[:3, :3] = Rotation.from_euler(
                "xyz", [0.05 * np.sin(2 * np.pi * 0.5 * t[i]),
                        0.15 * np.sin(2 * np.pi * 0.2 * t[i]),
                        0.02 * np.sin(2 * np.pi * 1.5 * t[i])]).as_matrix()
            T[:3, 3] = [0.05 * np.sin(2 * np.pi * 0.3 * t[i]),
                        0.03 * np.sin(2 * np.pi * 0.7 * t[i]),
                        0.02 * np.sin(2 * np.pi * 0.4 * t[i])]
        else:
            T[:3, :3] = Rotation.from_euler("xyz", [0.1, 0.2, -0.05]).as_matrix()
            T[:3, 3] = [0.3, 0.1, -0.2]
        poses[i] = T
    return poses


def make_hand_world(n: int) -> np.ndarray:
    """(n, 21, 3) smooth world-frame hand trajectory."""
    t = np.arange(n) / 30.0
    tpl = np.random.default_rng(0).normal(0.0, 0.03, size=(N_JOINTS, 3))
    centre = np.stack([0.15 * np.sin(2 * np.pi * 0.15 * t),
                       0.10 * np.sin(2 * np.pi * 0.23 * t),
                       0.60 + 0.05 * np.sin(2 * np.pi * 0.11 * t)], axis=1)
    return tpl[None] + centre[:, None, :]


def to_camera_frame(hand_world: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """World-frame points -> per-frame camera frame, i.e. what HaMeR would observe."""
    T_w2c = np.linalg.inv(T_world_cam)
    return np.einsum("nij,nkj->nki", T_w2c[:, :3, :3], hand_world) + T_w2c[:, None, :3, 3]


def apply_fixed(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply one (4, 4) transform to (n, k, 3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]


def bare_processor(T_cam2robot: np.ndarray, use_per_frame: bool = True,
                   data_folder: str = "/nonexistent") -> ActionProcessor:
    """An ActionProcessor with only the attributes _get_cam2robot touches.

    Bypasses __init__ so the test needs no config, dataset or camera calibration files.
    """
    ap = object.__new__(ActionProcessor)
    ap.T_cam2robot = T_cam2robot
    ap.use_per_frame_extrinsics = use_per_frame
    ap.data_folder = data_folder
    return ap


def write_poses(tmp: Path, T_world_cam: np.ndarray, key: str = "T_world_cam") -> SimpleNamespace:
    """Write camera_poses.npz and return a stub Paths exposing it."""
    path = tmp / "camera_poses.npz"
    np.savez_compressed(path, **{key: T_world_cam})
    return SimpleNamespace(camera_poses=path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def main() -> int:
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    T_init = make_cam2robot_init()
    hand_world = make_hand_world(N_FRAMES)
    missing_paths = SimpleNamespace(camera_poses=Path("/nonexistent/camera_poses.npz"))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # --- fallbacks -----------------------------------------------------
        ap_off = bare_processor(T_init, use_per_frame=False)
        moving = make_camera_track(N_FRAMES, moving=True)
        paths_moving = write_poses(tmp, moving)
        T = ap_off._get_cam2robot(paths_moving, N_FRAMES, "0")
        check("flag off ignores camera_poses.npz", T.ndim == 2 and np.allclose(T, T_init))

        ap = bare_processor(T_init)
        T = ap._get_cam2robot(missing_paths, N_FRAMES, "0")
        check("missing file falls back to the fixed extrinsic", T.ndim == 2 and np.allclose(T, T_init))

        bad_shape = write_poses(tmp, np.zeros((N_FRAMES, 3, 3)))
        T = ap._get_cam2robot(bad_shape, N_FRAMES, "0")
        check("wrong array shape falls back", T.ndim == 2 and np.allclose(T, T_init))

        wrong_key = write_poses(tmp, moving, key="poses")
        T = ap._get_cam2robot(wrong_key, N_FRAMES, "0")
        check("missing T_world_cam key falls back", T.ndim == 2 and np.allclose(T, T_init))

        nonfinite_first = moving.copy()
        nonfinite_first[0, 0, 3] = np.nan
        T = ap._get_cam2robot(write_poses(tmp, nonfinite_first), N_FRAMES, "0")
        check("non-finite first pose falls back (cannot anchor)",
              T.ndim == 2 and np.allclose(T, T_init))

        # --- anchoring ------------------------------------------------------
        paths_moving = write_poses(tmp, moving)
        T_seq = ap._get_cam2robot(paths_moving, N_FRAMES, "0")
        check("per-frame stack has one transform per frame",
              T_seq.ndim == 3 and T_seq.shape == (N_FRAMES, 4, 4), f"shape={T_seq.shape}")
        check("first frame equals the calibrated extrinsic",
              np.allclose(T_seq[0], T_init, atol=1e-12),
              f"max|diff|={np.abs(T_seq[0] - T_init).max():.2e}")
        check("per-frame transforms stay rigid",
              all(np.allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-9) for t in T_seq)
              and np.allclose(T_seq[:, 3, :], [0, 0, 0, 1]))

        # --- static camera must reproduce the old behaviour exactly ---------
        static = make_camera_track(N_FRAMES, moving=False)
        T_static = ap._get_cam2robot(write_poses(tmp, static), N_FRAMES, "0")
        check("static camera reproduces the fixed extrinsic on every frame",
              np.allclose(T_static, T_init[None], atol=1e-12),
              f"max|diff|={np.abs(T_static - T_init[None]).max():.2e}")

        hand_cam_static = to_camera_frame(hand_world, static)
        rf_static_per = ActionProcessor._convert_pts_to_robot_frame(hand_cam_static, T_static)
        rf_static_fix = ActionProcessor._convert_pts_to_robot_frame(hand_cam_static, T_init)
        check("static camera: per-frame and fixed paths agree to machine precision",
              np.allclose(rf_static_per, rf_static_fix, atol=1e-12),
              f"max|diff|={np.abs(rf_static_per - rf_static_fix).max():.2e}")

        # --- the actual bug -------------------------------------------------
        # With per-frame extrinsics the robot-frame trajectory must be exactly one rigid
        # transform of the true world trajectory. With a fixed extrinsic it cannot be.
        hand_cam = to_camera_frame(hand_world, moving)
        T_robot_world = T_init @ np.linalg.inv(moving[0])
        expected = apply_fixed(T_robot_world, hand_world)

        rf_per = ActionProcessor._convert_pts_to_robot_frame(hand_cam, T_seq)
        err_per = np.linalg.norm(rf_per - expected, axis=-1)
        check("per-frame: robot frame is a rigid transform of the world trajectory",
              err_per.max() < 1e-9, f"max err={err_per.max() * 1000:.2e} mm")

        rf_fix = ActionProcessor._convert_pts_to_robot_frame(hand_cam, T_init)
        err_fix = np.linalg.norm(rf_fix - expected, axis=-1)
        check("fixed extrinsic still shows the camera-motion error it always had",
              err_fix.mean() > 0.010,
              f"mean err={err_fix.mean() * 1000:.1f} mm, max={err_fix.max() * 1000:.1f} mm")

        # An idle hand is the worst case: all of its apparent motion is camera motion.
        idle_world = np.repeat(hand_world[:1], N_FRAMES, axis=0)
        idle_cam = to_camera_frame(idle_world, moving)
        idle_per = ActionProcessor._convert_pts_to_robot_frame(idle_cam, T_seq)
        idle_fix = ActionProcessor._convert_pts_to_robot_frame(idle_cam, T_init)
        span_per = np.linalg.norm(idle_per[:, 0].max(0) - idle_per[:, 0].min(0))
        span_fix = np.linalg.norm(idle_fix[:, 0].max(0) - idle_fix[:, 0].min(0))
        check("idle hand stays still with per-frame extrinsics but drifts with a fixed one",
              span_per < 1e-9 < span_fix,
              f"wrist travel: per-frame={span_per * 1000:.3f} mm, fixed={span_fix * 1000:.1f} mm")

        # --- length handling -------------------------------------------------
        T_trunc = ap._get_cam2robot(write_poses(tmp, moving), N_FRAMES - 10, "0")
        check("extra poses are truncated", len(T_trunc) == N_FRAMES - 10, f"n={len(T_trunc)}")

        T_pad = ap._get_cam2robot(write_poses(tmp, moving[:N_FRAMES - 10]), N_FRAMES, "0")
        check("missing poses are carried forward",
              len(T_pad) == N_FRAMES and np.allclose(T_pad[-1], T_pad[N_FRAMES - 11]),
              f"n={len(T_pad)}")

        nonfinite_mid = moving.copy()
        nonfinite_mid[20] = np.nan
        T_gap = ap._get_cam2robot(write_poses(tmp, nonfinite_mid), N_FRAMES, "0")
        check("a non-finite pose mid-sequence is carried forward, not propagated",
              np.isfinite(T_gap).all() and np.allclose(T_gap[20], T_gap[19]))

        # --- _convert_pts_to_robot_frame contract ---------------------------
        try:
            ActionProcessor._convert_pts_to_robot_frame(hand_cam, T_seq[:-5])
            check("length mismatch raises ValueError", False, "no exception raised")
        except ValueError:
            check("length mismatch raises ValueError", True)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'=' * 78}")
    print(f"{len(results) - n_fail}/{len(results)} passed")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
