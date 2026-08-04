#!/usr/bin/env python3
"""Diagnose how much ego camera motion corrupts Phantom's hand-pose pipeline.

Purpose
-------
Before building a SLAM + multi-frame optimization stack (see doc/slam_hand_pose.md),
answer two questions with EgoDex ground truth (Vision Pro head pose + 3D hand joints,
already parsed by convert_egodex.py):

  Q_motive : Does ignoring camera motion actually break the trajectory?
             EgoDex ships GT camera poses, so we can measure the induced error with a
             *perfect* hand estimator. If this number is small, the premise is dead.

  Q_headroom : Of HaMeR's actual world-frame error, how much is high-frequency jitter
               (removable by temporal smoothing) vs. drift/bias (not removable)?
               This is the ceiling on what the proposed optimizer can win.

Everything is measured against GT, so no metric here is self-referential (unlike jerk
or bone-length variance, which the proposed objective directly minimizes).

Blocks
------
  [0] Convention check    Are transforms/camera and kpts_3d in the frame we assume?
  [1] Camera motion       How much does the head move, and can monocular SLAM work on it?
  [2] Apparent vs true    Camera-frame vs world-frame hand motion, stratified by hand
                          speed and by idle/active hand.                        (GT only)
  [3] Fixed-extrinsics    Oracle hand pose + fixed T_cam2robot -> trajectory error. (GT only)
  [4] HaMeR cam-frame     MPJPE / root-relative / PA / XY-vs-Z split.           (needs HaMeR)
  [5] World-frame split   const bias | low-freq drift | high-freq jitter.       (needs HaMeR)
  [6] Smoothing ceiling   smooth in cam frame vs world frame, scored in world.  (needs HaMeR)
  [7] Bone lengths        GT noise floor vs HaMeR bone-length variability.

Blocks 1-3 need only the EgoDex HDF5, so they run in seconds on hundreds of episodes
without any pipeline output. Blocks 4-7 need hand_processor/hand_data_{side}.npz.

Usage
-----
    # GT-only (no pipeline run needed) -- gives the motivation numbers
    python b/diagnose_ego_camera_motion.py \
        --egodex-root /mnt/r/DATA/EgoDex/test \
        --task basic_pick_place --max-episodes 20 \
        --out b/out/diag

    # Full, against an already-processed demo tree
    python b/diagnose_ego_camera_motion.py \
        --egodex-root /mnt/r/DATA/EgoDex/test \
        --task basic_pick_place \
        --processed-root /mnt/r/DATA/EgoDex/test_phantom_processed/egodex_basic_pick_place \
        --out b/out/diag --plot

Notes
-----
* EgoDex uses the ARKit hand skeleton (25 joints/hand); HaMeR uses MANO's 21. The
  mapping in ARKIT_TO_MANO21 drops the 5 *Metacarpal joints. MANO fingertips are
  regressed onto the mesh surface, so expect a systematic per-joint offset of order
  1 cm against ARKit's anatomical joints. That is why block 5 separates the constant
  component out, and why block 6 reports offset-corrected numbers alongside raw ones.
* Block 3 is convention-invariant and offset-invariant (it is a best-fit rigid
  residual), so it is the most trustworthy single number in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Joint conventions
# ---------------------------------------------------------------------------

# Repo/HaMeR 21-joint order, confirmed by phantom/hand.py bone topology:
#   0 wrist | 1-4 thumb (mcp,pip,dip,tip) | 5-8 index | 9-12 middle
#   13-16 ring | 17-20 pinky
MANO21_NAMES = [
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# 20 bones: wrist->MCP x5, MCP->PIP x5, PIP->DIP x5, DIP->TIP x5
MANO21_BONES: List[Tuple[int, int]] = (
    [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
    + [(i, i + 1) for i in (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)]
)

# ARKit / visionOS joint suffixes matching the MANO21 order above.
# Knuckle==MCP, IntermediateBase==PIP, IntermediateTip==DIP, Tip==fingertip.
_ARKIT_SUFFIXES = [
    "Hand",
    "ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip",
    "IndexFingerKnuckle", "IndexFingerIntermediateBase", "IndexFingerIntermediateTip", "IndexFingerTip",
    "MiddleFingerKnuckle", "MiddleFingerIntermediateBase", "MiddleFingerIntermediateTip", "MiddleFingerTip",
    "RingFingerKnuckle", "RingFingerIntermediateBase", "RingFingerIntermediateTip", "RingFingerTip",
    "LittleFingerKnuckle", "LittleFingerIntermediateBase", "LittleFingerIntermediateTip", "LittleFingerTip",
]


def arkit_joint_names(side: str) -> List[str]:
    """HDF5 dataset names under transforms/ for one hand, in MANO21 order."""
    return [f"{side}{suffix}" for suffix in _ARKIT_SUFFIXES]


# OpenGL/ARKit (x right, y up, z backward) -> OpenCV (x right, y down, z forward)
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------
# Small linear-algebra / signal helpers
# ---------------------------------------------------------------------------

def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to points.

    Args:
        T: (4, 4) or (T, 4, 4) rigid transform(s).
        pts: (..., 3) or (T, J, 3) points.

    Returns:
        Transformed points, same shape as pts.
    """
    if T.ndim == 2:
        return pts @ T[:3, :3].T + T[:3, 3]
    # per-frame: T (N,4,4), pts (N,J,3)
    return np.einsum("nij,nkj->nki", T[:, :3, :3], pts) + T[:, None, :3, 3]


def rigid_align(src: np.ndarray, dst: np.ndarray, with_scale: bool = False) -> np.ndarray:
    """Least-squares rigid (optionally similarity) alignment of src onto dst.

    Args:
        src: (N, 3) source points.
        dst: (N, 3) target points.
        with_scale: also solve for a uniform scale.

    Returns:
        (N, 3) src mapped into dst's frame.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(s0.T @ d0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    scale = (S * [1.0, 1.0, d]).sum() / (s0 ** 2).sum() if with_scale else 1.0
    return scale * (s0 @ R.T) + mu_d


def gaussian_lowpass(x: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian low-pass along axis 0, with edge-replicating padding.

    Args:
        x: (T, ...) signal.
        sigma: kernel std in frames. sigma <= 0 returns x unchanged.
    """
    if sigma <= 0:
        return x.copy()
    radius = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    pad = [(radius, radius)] + [(0, 0)] * (x.ndim - 1)
    xp = np.pad(x, pad, mode="edge")
    out = np.zeros_like(x, dtype=float)
    for i, w in enumerate(k):
        out += w * xp[i : i + x.shape[0]]
    return out


def contiguous_runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Half-open [start, end) index ranges of True runs at least min_len long."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask)))
    return runs


def mm(x: float) -> float:
    """Meters -> millimeters, rounded for reporting."""
    return float(np.round(x * 1000.0, 2))


def rot_angle_deg(R: np.ndarray) -> np.ndarray:
    """Geodesic angle of (..., 3, 3) rotation matrices, in degrees."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def hand_speed(world: np.ndarray, fps: float) -> np.ndarray:
    """(T,) world-frame hand speed in m/s, averaged over joints, first frame replicated.

    Hand speed turns out to be the covariate that everything depends on: camera-motion
    contamination is negligible while the hand is transporting and dominant while it is
    hovering, grasping or resting. Never aggregate across speeds without stratifying.
    """
    v = np.linalg.norm(np.diff(world, axis=0), axis=-1).mean(1) * fps
    return np.concatenate([v[:1], v]) if len(v) else np.zeros(len(world))


# Speed strata in m/s. The slowest bin is where manipulation precision actually matters
# (approach, grasp, release, fine alignment) and where the camera dominates.
SPEED_BINS: List[Tuple[float, float]] = [(0.0, 0.05), (0.05, 0.15), (0.15, np.inf)]

# A hand whose world-frame wrist travels less than this over the episode is idle.
IDLE_EXTENT_M = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """One EgoDex episode's GT plus (optionally) the pipeline's HaMeR output."""
    name: str
    K: np.ndarray                                   # (3,3)
    T_c2w: np.ndarray                               # (N,4,4) camera-to-world, OpenCV convention
    gt_world: Dict[str, np.ndarray]                 # side -> (N,21,3) world-frame GT joints
    gt_valid: Dict[str, np.ndarray]                 # side -> (N,) bool
    hamer_cam: Dict[str, np.ndarray] = field(default_factory=dict)   # side -> (N,21,3)
    hamer_2d: Dict[str, np.ndarray] = field(default_factory=dict)    # side -> (N,21,2)
    hamer_valid: Dict[str, np.ndarray] = field(default_factory=dict) # side -> (N,) bool
    img_wh: Optional[Tuple[int, int]] = None

    @property
    def n_frames(self) -> int:
        return len(self.T_c2w)


def load_gt(hdf5_path: Path, conf_thresh: float, to_opencv: bool) -> Episode:
    """Read camera trajectory + 21-joint world-frame hand GT from an EgoDex HDF5."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"h5py is required to read EgoDex HDF5: {exc}") from exc

    with h5py.File(hdf5_path, "r") as f:
        K = np.asarray(f["camera/intrinsic"][:], dtype=float)
        T_c2w = np.asarray(f["transforms/camera"][:], dtype=float)
        n = len(T_c2w)

        gt_world, gt_valid = {}, {}
        for side in ("left", "right"):
            names = arkit_joint_names(side)
            missing = [nm for nm in names if f"transforms/{nm}" not in f]
            if missing:
                raise KeyError(f"{hdf5_path.name}: missing transforms/{missing[0]} (+{len(missing)-1} more)")
            joints = np.stack([f[f"transforms/{nm}"][:, :3, 3] for nm in names], axis=1)  # (N,21,3)
            gt_world[side] = np.asarray(joints, dtype=float)

            conf_key = f"confidences/{side}Hand"
            if conf_key in f:
                gt_valid[side] = np.asarray(f[conf_key][:], dtype=float) >= conf_thresh
            else:
                gt_valid[side] = np.ones(n, dtype=bool)

    if to_opencv:
        T_c2w = T_c2w @ GL_TO_CV[None]

    return Episode(name=hdf5_path.stem, K=K, T_c2w=T_c2w, gt_world=gt_world, gt_valid=gt_valid)


def attach_hamer(ep: Episode, demo_dir: Path, prefer_3d: bool) -> bool:
    """Load hand_data_{side}.npz into the episode. Returns True if anything loaded."""
    hp = demo_dir / "hand_processor"
    got = False
    for side in ("left", "right"):
        candidates = []
        if prefer_3d:
            candidates.append(hp / f"hand_data_3d_{side}.npz")
        candidates.append(hp / f"hand_data_{side}.npz")
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        d = np.load(path, allow_pickle=True)
        k3 = np.asarray(d["kpts_3d"], dtype=float)
        det = np.asarray(d["hand_detected"]).astype(bool)
        n = min(len(k3), ep.n_frames)
        if len(k3) != ep.n_frames:
            print(f"    ! {ep.name}/{side}: {len(k3)} HaMeR frames vs {ep.n_frames} GT frames; "
                  f"truncating to {n}")
        ep.hamer_cam[side] = k3[:n]
        ep.hamer_valid[side] = det[:n]
        if "kpts_2d" in d:
            ep.hamer_2d[side] = np.asarray(d["kpts_2d"], dtype=float)[:n]
        got = True
    return got


def probe_video_size(demo_dir: Path) -> Optional[Tuple[int, int]]:
    """Read (W, H) from the demo's video, if OpenCV and the file are available."""
    try:
        import cv2
    except ImportError:
        return None
    for name in ("video_L.mp4", "video_rgb_imgs.mkv"):
        p = demo_dir / name
        if not p.exists():
            continue
        cap = cv2.VideoCapture(str(p))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return (w, h)
    return None


# ---------------------------------------------------------------------------
# Block 0: convention / correspondence sanity check
# ---------------------------------------------------------------------------

def block0_convention(ep: Episode) -> dict:
    """Verify our camera convention and ARKit->MANO21 mapping before trusting anything.

    Two independent tests:
      * in_image_frac -- GT joints projected with GT camera should land in frame.
      * reproj_px     -- projected GT joints should sit near HaMeR's 2D keypoints.
                         A near-constant scale factor between the two means the
                         pipeline resized frames; a large residual after fitting
                         scale+shift means the joint mapping or convention is wrong.
    Both are reported for the assumed convention and for the flipped one.
    """
    out: dict = {}
    W, H = ep.img_wh if ep.img_wh else (None, None)

    for label, T_c2w in (("assumed", ep.T_c2w), ("flipped", ep.T_c2w @ GL_TO_CV[None])):
        T_w2c = np.linalg.inv(T_c2w)
        in_frac, reproj, scales = [], [], []
        for side in ("left", "right"):
            valid = ep.gt_valid[side]
            if not valid.any():
                continue
            cam = transform_points(T_w2c, ep.gt_world[side])          # (N,21,3)
            front = cam[..., 2] > 1e-3
            uvw = cam @ ep.K.T
            with np.errstate(divide="ignore", invalid="ignore"):
                uv = uvw[..., :2] / uvw[..., 2:3]
            if W:
                inside = front & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
                in_frac.append(float(inside[valid].mean()))

            if side in ep.hamer_2d:
                n = len(ep.hamer_2d[side])
                m = valid[:n] & ep.hamer_valid[side]
                if m.sum() >= 5:
                    a = uv[:n][m].reshape(-1, 2)
                    b = ep.hamer_2d[side][m].reshape(-1, 2)
                    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
                    a, b = a[ok], b[ok]
                    if len(a) >= 5:
                        # fit b ~= s*a + t (uniform scale + shift) to absorb any resize
                        s = np.std(b, axis=0).mean() / max(np.std(a, axis=0).mean(), 1e-9)
                        t = b.mean(0) - s * a.mean(0)
                        reproj.append(float(np.median(np.linalg.norm(s * a + t - b, axis=1))))
                        scales.append(float(s))

        out[label] = {
            "in_image_frac": float(np.mean(in_frac)) if in_frac else None,
            "reproj_median_px_after_similarity_fit": float(np.mean(reproj)) if reproj else None,
            "fitted_pixel_scale": float(np.mean(scales)) if scales else None,
        }
    return out


# ---------------------------------------------------------------------------
# Blocks 1-3: GT-only
# ---------------------------------------------------------------------------

def block1_camera_motion(ep: Episode, fps: float) -> dict:
    """How much does the head actually move, and can monocular SLAM work on it?

    The SLAM conditioning numbers matter as much as the motion magnitudes. Monocular
    structure-from-motion needs translation parallax that competes with rotation-induced
    flow; a head that wobbles in place and returns to where it started provides almost
    none, which is the classic degenerate configuration:

      net_over_path            -- ~0 means the camera oscillates rather than travels.
      max_baseline_over_depth  -- triangulation parallax. Below ~0.05 is ill-conditioned,
                                  and metric scale recovery has essentially no signal.
      rot_over_trans_flow      -- rotation-induced image flow divided by translation-induced
                                  flow. Above ~3 means rotation dominates and depth is
                                  poorly observable.
    """
    t = ep.T_c2w[:, :3, 3]
    R = ep.T_c2w[:, :3, :3]
    dt = np.diff(t, axis=0)
    dR = np.einsum("nij,nkj->nik", R[1:], R[:-1])
    lin = np.linalg.norm(dt, axis=1) * fps           # m/s
    ang = rot_angle_deg(dR) * fps                    # deg/s
    path = float(np.linalg.norm(dt, axis=1).sum())
    net = float(np.linalg.norm(t[-1] - t[0]))

    # Scene depth proxy: median GT wrist depth in the camera frame.
    T_w2c = np.linalg.inv(ep.T_c2w)
    depths = [float(np.median(transform_points(T_w2c, ep.gt_world[s])[ep.gt_valid[s], 0, 2]))
              for s in ("left", "right") if ep.gt_valid[s].any()]
    Z = float(np.median(depths)) if depths else float("nan")
    max_baseline = float(np.linalg.norm(t[:, None, :] - t[None, :, :], axis=-1).max())

    d_theta = np.radians(rot_angle_deg(dR))                      # rad/frame
    trans_flow = np.linalg.norm(dt, axis=1) / max(Z, 1e-6)       # rad/frame at depth Z
    rot_over_trans = float(np.median(d_theta / np.maximum(trans_flow, 1e-9)))

    return {
        "n_frames": ep.n_frames,
        "duration_s": round(ep.n_frames / fps, 2),
        "path_length_m": round(path, 4),
        "net_displacement_m": round(net, 4),
        "net_over_path": round(net / max(path, 1e-9), 4),
        "translation_extent_m": round(float(np.linalg.norm(t.max(0) - t.min(0))), 4),
        "lin_speed_mps": {"mean": round(float(lin.mean()), 4), "p95": round(float(np.percentile(lin, 95)), 4)},
        "ang_speed_dps": {"mean": round(float(ang.mean()), 3), "p95": round(float(np.percentile(ang, 95)), 3)},
        "total_rotation_deg": round(float(rot_angle_deg(R[0].T @ R[-1])), 2),
        "slam_conditioning": {
            "scene_depth_m": round(Z, 4),
            "max_baseline_m": round(max_baseline, 4),
            "max_baseline_over_depth": round(max_baseline / max(Z, 1e-6), 4),
            "rot_over_trans_flow_median": round(rot_over_trans, 2),
        },
    }


def block2_apparent_vs_true(ep: Episode, fps: float) -> dict:
    """Camera-frame apparent hand motion vs. true world-frame hand motion (GT only).

    The camera-frame signal is what HaMeR sees and what the current pipeline treats as
    hand motion. If |d hand_cam| >> |d hand_world|, most of that signal is the camera.

    Stratified by true hand speed, because the episode-level ratio is bimodal and its
    median is meaningless: a transporting hand swamps the camera while a hovering or
    resting hand is swamped by it. Idle hands are flagged separately for the same reason
    -- in a bimanual setup their action labels can be almost pure camera motion.
    """
    T_w2c = np.linalg.inv(ep.T_c2w)
    out = {}
    for side in ("left", "right"):
        valid = ep.gt_valid[side]
        pair = valid[:-1] & valid[1:]
        if pair.sum() < 10:
            out[side] = None
            continue
        world = ep.gt_world[side]
        cam = transform_points(T_w2c, world)
        v_world = np.linalg.norm(np.diff(world, axis=0), axis=-1)[pair] * fps   # (M,21)
        v_cam = np.linalg.norm(np.diff(cam, axis=0), axis=-1)[pair] * fps
        # Per-frame ratio of apparent to true speed, aggregated over joints.
        tv, av = v_world.mean(1), v_cam.mean(1)
        ratio = av / np.maximum(tv, 1e-6)

        strata = {}
        for lo, hi in SPEED_BINS:
            m = (tv >= lo) & (tv < hi)
            if m.sum() < 5:
                continue
            strata[f"speed_{lo:g}_{hi:g}"] = {
                "n_frames": int(m.sum()),
                "true_mps": round(float(tv[m].mean()), 4),
                "apparent_mps": round(float(av[m].mean()), 4),
                "ratio_median": round(float(np.median(ratio[m])), 3),
                "frac_camera_dominates": round(float((ratio[m] > 2.0).mean()), 3),
            }

        wrist = world[valid][:, 0]
        extent = float(np.linalg.norm(wrist.max(0) - wrist.min(0)))
        out[side] = {
            "n_valid_frames": int(valid.sum()),
            "hand_extent_m": round(extent, 4),
            "is_idle_hand": bool(extent < IDLE_EXTENT_M),
            "true_speed_mps": round(float(v_world.mean()), 4),
            "apparent_speed_mps": round(float(v_cam.mean()), 4),
            "apparent_over_true": {
                "median": round(float(np.median(ratio)), 3),
                "p90": round(float(np.percentile(ratio, 90)), 3),
            },
            "frac_frames_camera_dominates": round(float((ratio > 2.0).mean()), 3),
            "by_speed": strata,
        }
    return out


def block3_fixed_extrinsics(ep: Episode, fps: float, window_s: Sequence[float]) -> dict:
    """Error induced by a fixed T_cam2robot, assuming a PERFECT hand estimator.

    The pipeline maps camera-frame points to the robot frame with one constant
    transform, so its output is hand_cam_gt[t] up to a fixed rigid transform. We
    therefore report the residual of the best-fit rigid alignment between
    hand_cam_gt and hand_world_gt over the whole sequence -- the irreducible error
    of the fixed-extrinsics assumption. A static camera would give exactly 0.

    Convention-invariant and offset-invariant: the most trustworthy number here.
    """
    T_w2c = np.linalg.inv(ep.T_c2w)
    out = {}
    for side in ("left", "right"):
        valid = ep.gt_valid[side]
        if valid.sum() < 30:
            out[side] = None
            continue
        world = ep.gt_world[side][valid]
        cam = transform_points(T_w2c, ep.gt_world[side])[valid]

        A, B = cam.reshape(-1, 3), world.reshape(-1, 3)
        res = np.linalg.norm(rigid_align(A, B) - B, axis=1)

        # Wrist-only version: the trajectory the robot end-effector actually follows.
        wa, wb = cam[:, 0], world[:, 0]
        res_wrist = np.linalg.norm(rigid_align(wa, wb) - wb, axis=1)

        # How local does the assumption have to be before it becomes acceptable?
        per_window = {}
        idx = np.flatnonzero(valid)
        for ws in window_s:
            wlen = max(4, int(round(ws * fps)))
            errs = []
            for s in range(0, len(idx) - wlen + 1, max(1, wlen // 2)):
                sl = slice(s, s + wlen)
                a = transform_points(T_w2c[idx[sl]], ep.gt_world[side][idx[sl]]).reshape(-1, 3)
                b = ep.gt_world[side][idx[sl]].reshape(-1, 3)
                errs.append(np.linalg.norm(rigid_align(a, b) - b, axis=1).mean())
            per_window[f"{ws}s"] = mm(float(np.mean(errs))) if errs else None

        out[side] = {
            "residual_all_joints_mm": {
                "mean": mm(float(res.mean())),
                "rms": mm(float(np.sqrt((res ** 2).mean()))),
                "p95": mm(float(np.percentile(res, 95))),
            },
            "residual_wrist_mm": {
                "mean": mm(float(res_wrist.mean())),
                "p95": mm(float(np.percentile(res_wrist, 95))),
            },
            "residual_by_window_mm": per_window,
            "hand_world_extent_m": round(float(np.linalg.norm(world[:, 0].max(0) - world[:, 0].min(0))), 4),
        }
    return out


# ---------------------------------------------------------------------------
# Blocks 4-7: need HaMeR output
# ---------------------------------------------------------------------------

def _paired(ep: Episode, side: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Frames where both GT and HaMeR are valid. Returns (mask, gt_world, hamer_cam, T_c2w)."""
    if side not in ep.hamer_cam:
        return None
    n = len(ep.hamer_cam[side])
    mask = ep.gt_valid[side][:n] & ep.hamer_valid[side]
    if mask.sum() < 30:
        return None
    return mask, ep.gt_world[side][:n], ep.hamer_cam[side], ep.T_c2w[:n]


def block4_hamer_camera_frame(ep: Episode) -> dict:
    """HaMeR error in the camera frame, split into the parts that matter.

    root_mm       -- wrist position error: the weak-perspective global translation,
                     which monocular methods estimate badly.
    root_rel_mm   -- articulation error after subtracting the wrist: what HaMeR is good at.
    pa_mm         -- Procrustes-aligned: shape/pose error with rotation+scale removed.
    xy_mm / z_mm  -- lateral vs. depth error. If z >> xy, an isotropic 3D data term
                     anchors the optimizer to the worst channel (use 2D reprojection).
    """
    out = {}
    for side in ("left", "right"):
        p = _paired(ep, side)
        if p is None:
            out[side] = None
            continue
        mask, gt_world, ham_cam, T_c2w = p
        gt_cam = transform_points(np.linalg.inv(T_c2w), gt_world)[mask]
        hm = ham_cam[mask]

        err = hm - gt_cam
        root_err = np.linalg.norm(err[:, 0], axis=-1)
        rel = (hm - hm[:, :1]) - (gt_cam - gt_cam[:, :1])
        pa = np.concatenate([
            np.linalg.norm(rigid_align(h, g, with_scale=True) - g, axis=1)
            for h, g in zip(hm, gt_cam)
        ])
        d_gt = np.linalg.norm(gt_cam[:, 0], axis=-1)
        d_hm = np.linalg.norm(hm[:, 0], axis=-1)

        out[side] = {
            "n_frames": int(mask.sum()),
            "detection_rate": round(float(ep.hamer_valid[side].mean()), 3),
            "mpjpe_mm": mm(float(np.linalg.norm(err, axis=-1).mean())),
            "root_mm": mm(float(root_err.mean())),
            "root_rel_mpjpe_mm": mm(float(np.linalg.norm(rel, axis=-1).mean())),
            "pa_mpjpe_mm": mm(float(pa.mean())),
            "xy_rms_mm": mm(float(np.sqrt((err[..., :2] ** 2).sum(-1).mean()))),
            "z_rms_mm": mm(float(np.sqrt((err[..., 2] ** 2).mean()))),
            "wrist_depth_ratio_hamer_over_gt": round(float(np.median(d_hm / np.maximum(d_gt, 1e-6))), 4),
            "wrist_depth_gt_m": round(float(np.median(d_gt)), 4),
        }
    return out


def block5_world_error_split(ep: Episode, sigmas: Sequence[float]) -> dict:
    """Split world-frame HaMeR error into constant / drift / jitter energy.

    Uses GT camera poses, so this is the SLAM-is-perfect upper bound.
      const  -- per-joint time-mean of the error. Part convention offset (MANO vs
                ARKit), part calibration bias. Temporal smoothing cannot touch it.
      drift  -- low-frequency residual. Smoothing cannot fix this either.
      jitter -- high-frequency residual. This is the *only* part a temporal
                smoothness prior can remove, i.e. the method's headroom.
    """
    out = {}
    for side in ("left", "right"):
        p = _paired(ep, side)
        if p is None:
            out[side] = None
            continue
        mask, gt_world, ham_cam, T_c2w = p
        ham_world = transform_points(T_c2w, ham_cam)

        per_sigma = {}
        for sg in sigmas:
            e_sq = c_sq = d_sq = j_sq = 0.0
            npts = 0
            for a, b in contiguous_runs(mask, min_len=max(9, int(6 * sg))):
                e = ham_world[a:b] - gt_world[a:b]           # (L,21,3)
                const = e.mean(0, keepdims=True)
                resid = e - const
                drift = gaussian_lowpass(resid, sg)
                jitter = resid - drift
                e_sq += float((e ** 2).sum())
                c_sq += float((np.broadcast_to(const, e.shape) ** 2).sum())
                d_sq += float((drift ** 2).sum())
                j_sq += float((jitter ** 2).sum())
                npts += e.shape[0] * e.shape[1]
            if npts == 0:
                continue
            per_sigma[f"sigma_{sg:g}f"] = {
                "const_frac": round(c_sq / max(e_sq, 1e-12), 3),
                "drift_frac": round(d_sq / max(e_sq, 1e-12), 3),
                "jitter_frac": round(j_sq / max(e_sq, 1e-12), 3),
                "jitter_rms_mm": mm(float(np.sqrt(j_sq / (3 * npts)))),
            }

        e_all = ham_world[mask] - gt_world[mask]
        w_mpjpe = float(np.linalg.norm(e_all, axis=-1).mean())
        e_centered = e_all - e_all.mean(0, keepdims=True)
        out[side] = {
            "w_mpjpe_mm": mm(w_mpjpe),
            "w_mpjpe_offset_corrected_mm": mm(float(np.linalg.norm(e_centered, axis=-1).mean())),
            "energy_split": per_sigma,
        }
    return out


def block6_smoothing_ceiling(ep: Episode, sigmas: Sequence[float], fps: float) -> dict:
    """Does smoothing in the WORLD frame beat smoothing in the CAMERA frame?

    This is the core hypothesis of doc/slam_hand_pose.md, tested with GT camera poses
    and a plain Gaussian filter as a stand-in for the proposed optimizer:

      cam_frame   -- filter hand_cam, then transform to world. Approximates the current
                     pipeline (GP smoothing applied after a FIXED extrinsic, so it is
                     effectively camera-frame smoothing).
      world_frame -- transform to world first, then filter. The proposed method.

    Both are scored as world-frame MPJPE against GT. If world_frame does not win, the
    premise of the whole project fails and no amount of optimizer engineering saves it.

    Scored overall AND on slow frames only. The two frames are nearly equivalent while
    the hand is transporting, so a pooled number dilutes the comparison; the slow frames
    are both where the frames diverge most and where manipulation precision matters.
    """
    out = {}
    for side in ("left", "right"):
        p = _paired(ep, side)
        if p is None:
            out[side] = None
            continue
        mask, gt_world, ham_cam, T_c2w = p
        ham_world = transform_points(T_c2w, ham_cam)
        speed = hand_speed(gt_world, fps)
        slow_thresh = SPEED_BINS[0][1]

        def score(pred: np.ndarray, sl: slice) -> Dict[str, Tuple[float, int]]:
            """Weighted (error, count) for all frames and for slow frames only."""
            e = pred - gt_world[sl]
            cor = e - e.mean(0, keepdims=True)
            slow = speed[sl] < slow_thresh
            res = {
                "raw": (float(np.linalg.norm(e, axis=-1).mean()) * (sl.stop - sl.start), sl.stop - sl.start),
                "cor": (float(np.linalg.norm(cor, axis=-1).mean()) * (sl.stop - sl.start), sl.stop - sl.start),
            }
            n_slow = int(slow.sum())
            res["cor_slow"] = ((float(np.linalg.norm(cor[slow], axis=-1).mean()) * n_slow, n_slow)
                               if n_slow else (0.0, 0))
            return res

        runs = contiguous_runs(mask, min_len=15)
        if not runs:
            out[side] = None
            continue

        curves = {"cam_frame": {}, "world_frame": {}}
        for sg in sigmas:
            acc = {k: {m: [0.0, 0] for m in ("raw", "cor", "cor_slow")}
                   for k in ("cam_frame", "world_frame")}
            for a, b in runs:
                sl = slice(a, b)
                cam_s = transform_points(T_c2w[sl], gaussian_lowpass(ham_cam[sl], sg))
                wld_s = gaussian_lowpass(ham_world[sl], sg)
                for key, pred in (("cam_frame", cam_s), ("world_frame", wld_s)):
                    for metric, (tot, cnt) in score(pred, sl).items():
                        acc[key][metric][0] += tot
                        acc[key][metric][1] += cnt
            for key in curves:
                a_k = acc[key]
                curves[key][f"sigma_{sg:g}f"] = {
                    "w_mpjpe_mm": mm(a_k["raw"][0] / max(a_k["raw"][1], 1)),
                    "w_mpjpe_offset_corrected_mm": mm(a_k["cor"][0] / max(a_k["cor"][1], 1)),
                    "w_mpjpe_slow_frames_mm": (mm(a_k["cor_slow"][0] / a_k["cor_slow"][1])
                                               if a_k["cor_slow"][1] else None),
                    "n_slow_frames": a_k["cor_slow"][1],
                }

        def best(key: str, metric: str) -> dict:
            items = {k: v for k, v in curves[key].items() if v[metric] is not None}
            if not items:
                return {}
            k = min(items, key=lambda s: items[s][metric])
            return {"sigma": k, **items[k]}

        b_cam, b_wld = best("cam_frame", "w_mpjpe_offset_corrected_mm"), best("world_frame", "w_mpjpe_offset_corrected_mm")
        s_cam = best("cam_frame", "w_mpjpe_slow_frames_mm")
        s_wld = best("world_frame", "w_mpjpe_slow_frames_mm")
        base = curves["world_frame"][f"sigma_{sigmas[0]:g}f"] if sigmas[0] == 0 else None

        def delta(w: dict, c: dict, metric: str) -> Optional[float]:
            if not w or not c or w.get(metric) is None or c.get(metric) is None:
                return None
            return round(w[metric] - c[metric], 2)

        out[side] = {
            "curves": curves,
            "best_cam_frame": b_cam,
            "best_world_frame": b_wld,
            "world_minus_cam_mm": delta(b_wld, b_cam, "w_mpjpe_offset_corrected_mm"),
            "world_minus_cam_slow_frames_mm": delta(s_wld, s_cam, "w_mpjpe_slow_frames_mm"),
            "unsmoothed_mm": base["w_mpjpe_offset_corrected_mm"] if base else None,
        }
    return out


def block7_bone_lengths(ep: Episode) -> dict:
    """Bone-length variability: GT noise floor vs. HaMeR.

    The proposed lambda_3 penalizes exactly this quantity, so its value here is only
    a diagnostic -- it tells you whether there is anything for the constraint to fix,
    and what GT's own floor is (i.e. how low a CV is even meaningful).
    """
    def cv_of(joints: np.ndarray) -> Tuple[float, float]:
        lens = np.stack([np.linalg.norm(joints[:, j] - joints[:, k], axis=-1)
                         for j, k in MANO21_BONES], axis=1)          # (T,20)
        cv = lens.std(0) / np.maximum(lens.mean(0), 1e-9)
        return float(cv.mean()), float(lens.mean(0).sum())

    out = {}
    for side in ("left", "right"):
        valid = ep.gt_valid[side]
        entry: dict = {}
        if valid.sum() >= 30:
            cv, total = cv_of(ep.gt_world[side][valid])
            entry["gt_bone_cv"] = round(cv, 4)
            entry["gt_total_bone_len_m"] = round(total, 4)
        p = _paired(ep, side)
        if p is not None:
            mask, _, ham_cam, _ = p
            cv, total = cv_of(ham_cam[mask])
            entry["hamer_bone_cv"] = round(cv, 4)
            entry["hamer_total_bone_len_m"] = round(total, 4)
        out[side] = entry or None
    return out


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def _collect(per_ep: List[dict], path: Sequence[str], only: str = "all") -> List[float]:
    """Pull a numeric leaf from every episode/side, skipping missing entries.

    Args:
        only: "all", "idle" or "active" -- restrict to hands classified by block 2.
              Aggregating idle and active hands together produces a bimodal mixture
              whose median means nothing, so most callers should pick a side.
    """
    vals = []
    for rec in per_ep:
        for side in ("left", "right"):
            if only != "all":
                av = rec.get("apparent_vs_true", {})
                av = av.get(side) if isinstance(av, dict) else None
                if not isinstance(av, dict):
                    continue
                if av.get("is_idle_hand", False) != (only == "idle"):
                    continue
            node = rec.get(path[0], {})
            node = node.get(side) if isinstance(node, dict) else None
            for key in path[1:]:
                if not isinstance(node, dict):
                    node = None
                    break
                node = node.get(key)
            if isinstance(node, (int, float)) and not isinstance(node, bool):
                vals.append(float(node))
    return vals


def summarize(per_ep: List[dict], sigmas: Sequence[float]) -> dict:
    """Aggregate the numbers that decide whether the project is viable."""
    def stat(vals: List[float]) -> Optional[dict]:
        if not vals:
            return None
        return {"mean": round(float(np.mean(vals)), 3),
                "median": round(float(np.median(vals)), 3),
                "p90": round(float(np.percentile(vals, 90)), 3),
                "n": len(vals)}

    cam = [r["camera_motion"] for r in per_ep if r.get("camera_motion")]
    mid = sigmas[len(sigmas) // 2]
    return {
        "n_episodes": len(per_ep),
        "camera": {
            "lin_speed_mps_mean": stat([c["lin_speed_mps"]["mean"] for c in cam]),
            "ang_speed_dps_mean": stat([c["ang_speed_dps"]["mean"] for c in cam]),
            "path_length_m": stat([c["path_length_m"] for c in cam]),
        },
        "slam_conditioning": {
            "net_over_path": stat([c["net_over_path"] for c in cam]),
            "max_baseline_over_depth": stat([c["slam_conditioning"]["max_baseline_over_depth"] for c in cam]),
            "rot_over_trans_flow": stat([c["slam_conditioning"]["rot_over_trans_flow_median"] for c in cam]),
            "scene_depth_m": stat([c["slam_conditioning"]["scene_depth_m"] for c in cam]),
        },
        # Idle and active hands behave completely differently; never pool them.
        "apparent_over_true_speed": {
            "idle_hands": stat(_collect(per_ep, ["apparent_vs_true", "apparent_over_true", "median"], "idle")),
            "active_hands": stat(_collect(per_ep, ["apparent_vs_true", "apparent_over_true", "median"], "active")),
        },
        "apparent_over_true_by_speed": {
            f"speed_{lo:g}_{hi:g}": {
                "ratio_median": stat(_collect(
                    per_ep, ["apparent_vs_true", "by_speed", f"speed_{lo:g}_{hi:g}", "ratio_median"])),
                "frac_camera_dominates": stat(_collect(
                    per_ep, ["apparent_vs_true", "by_speed", f"speed_{lo:g}_{hi:g}", "frac_camera_dominates"])),
            }
            for lo, hi in SPEED_BINS
        },
        "n_idle_hands": len(_collect(per_ep, ["apparent_vs_true", "n_valid_frames"], "idle")),
        "n_active_hands": len(_collect(per_ep, ["apparent_vs_true", "n_valid_frames"], "active")),
        "fixed_extrinsics_residual_mm": {
            "all_joints_mean": stat(_collect(per_ep, ["fixed_extrinsics", "residual_all_joints_mm", "mean"])),
            "wrist_mean": stat(_collect(per_ep, ["fixed_extrinsics", "residual_wrist_mm", "mean"])),
            "idle_hands": stat(_collect(per_ep, ["fixed_extrinsics", "residual_all_joints_mm", "mean"], "idle")),
            "active_hands": stat(_collect(per_ep, ["fixed_extrinsics", "residual_all_joints_mm", "mean"], "active")),
            "by_window": {w: stat(_collect(per_ep, ["fixed_extrinsics", "residual_by_window_mm", w]))
                          for w in ("0.5s", "1.0s", "2.0s", "5.0s")},
        },
        "hamer_camera_frame_mm": {
            "mpjpe": stat(_collect(per_ep, ["hamer_camera_frame", "mpjpe_mm"])),
            "root": stat(_collect(per_ep, ["hamer_camera_frame", "root_mm"])),
            "root_rel_mpjpe": stat(_collect(per_ep, ["hamer_camera_frame", "root_rel_mpjpe_mm"])),
            "pa_mpjpe": stat(_collect(per_ep, ["hamer_camera_frame", "pa_mpjpe_mm"])),
            "xy_rms": stat(_collect(per_ep, ["hamer_camera_frame", "xy_rms_mm"])),
            "z_rms": stat(_collect(per_ep, ["hamer_camera_frame", "z_rms_mm"])),
        },
        "world_error_split": {
            f"at_sigma_{mid:g}f": {
                "const_frac": stat(_collect(per_ep, ["world_error_split", "energy_split", f"sigma_{mid:g}f", "const_frac"])),
                "drift_frac": stat(_collect(per_ep, ["world_error_split", "energy_split", f"sigma_{mid:g}f", "drift_frac"])),
                "jitter_frac": stat(_collect(per_ep, ["world_error_split", "energy_split", f"sigma_{mid:g}f", "jitter_frac"])),
            },
            "w_mpjpe_mm": stat(_collect(per_ep, ["world_error_split", "w_mpjpe_mm"])),
            "w_mpjpe_offset_corrected_mm": stat(
                _collect(per_ep, ["world_error_split", "w_mpjpe_offset_corrected_mm"])),
        },
        "smoothing_ceiling_mm": {
            "best_cam_frame": stat(_collect(per_ep, ["smoothing_ceiling", "best_cam_frame", "w_mpjpe_offset_corrected_mm"])),
            "best_world_frame": stat(_collect(per_ep, ["smoothing_ceiling", "best_world_frame", "w_mpjpe_offset_corrected_mm"])),
            "world_minus_cam": stat(_collect(per_ep, ["smoothing_ceiling", "world_minus_cam_mm"])),
            "world_minus_cam_slow_frames": stat(
                _collect(per_ep, ["smoothing_ceiling", "world_minus_cam_slow_frames_mm"])),
            "unsmoothed": stat(_collect(per_ep, ["smoothing_ceiling", "unsmoothed_mm"])),
        },
        "bone_cv": {
            "gt": stat(_collect(per_ep, ["bone_lengths", "gt_bone_cv"])),
            "hamer": stat(_collect(per_ep, ["bone_lengths", "hamer_bone_cv"])),
        },
    }


def print_report(summary: dict, per_ep: List[dict], sigmas: Sequence[float]) -> None:
    """Print the decision-relevant numbers with the verdict spelled out."""
    def g(d, *keys):
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d

    line = "=" * 78
    print(f"\n{line}\nDIAGNOSTIC SUMMARY  ({summary['n_episodes']} episodes)\n{line}")

    conv = per_ep[0].get("convention", {}) if per_ep else {}
    if conv:
        print("\n[0] Convention check (first episode)")
        for label in ("assumed", "flipped"):
            c = conv.get(label, {})
            print(f"    {label:<8} in-image={c.get('in_image_frac')}  "
                  f"reproj={c.get('reproj_median_px_after_similarity_fit')} px  "
                  f"pixel_scale={c.get('fitted_pixel_scale')}")
        print("    -> 'assumed' must win clearly. If 'flipped' wins, rerun with "
              "--camera-convention opengl.")
        print("    -> reproj >> 10 px means the ARKit->MANO21 mapping or the units are wrong;"
              " stop and fix that before reading anything below.")

    print("\n[1] Camera motion")
    print(f"    lin speed  {g(summary,'camera','lin_speed_mps_mean','mean')} m/s   "
          f"ang speed {g(summary,'camera','ang_speed_dps_mean','mean')} deg/s   "
          f"path {g(summary,'camera','path_length_m','mean')} m")
    sl = summary.get("slam_conditioning", {})
    print("    SLAM conditioning:")
    print(f"      net/path displacement  {g(sl,'net_over_path','median')}   "
          "(~0 = camera wobbles in place instead of travelling)")
    print(f"      max baseline / depth   {g(sl,'max_baseline_over_depth','median')}   "
          "(<0.05 = triangulation ill-conditioned, metric scale has no signal)")
    print(f"      rotation/translation flow  {g(sl,'rot_over_trans_flow','median')}   "
          "(>3 = rotation-dominated, depth poorly observable)")
    print("    -> Both bad => monocular SLAM is in its degenerate regime on this data;")
    print("       sec 3.1-3.2 of the plan would be building on sand.")

    print("\n[2] Apparent (camera-frame) vs true (world-frame) hand speed")
    ao = summary.get("apparent_over_true_speed", {})
    print(f"    idle hands   ratio {g(ao,'idle_hands','median')}  (n={summary.get('n_idle_hands')})")
    print(f"    active hands ratio {g(ao,'active_hands','median')}  (n={summary.get('n_active_hands')})")
    byspeed = summary.get("apparent_over_true_by_speed", {})
    for key, val in byspeed.items():
        print(f"    {key:<18} ratio {g(val,'ratio_median','median')}   "
              f"camera dominates {g(val,'frac_camera_dominates','median')} of frames")
    print("    -> The effect is speed-dependent, so a pooled ratio is meaningless. Slow")
    print("       frames are the grasp/release/alignment moments that decide manipulation.")

    print("\n[3] Fixed-extrinsics residual with a PERFECT hand estimator  <-- key motivation")
    fe = summary.get("fixed_extrinsics_residual_mm", {})
    print(f"    all joints  {g(fe,'all_joints_mean','median')} mm (median ep)   "
          f"wrist {g(fe,'wrist_mean','median')} mm")
    print(f"    idle hands  {g(fe,'idle_hands','median')} mm   "
          f"active hands {g(fe,'active_hands','median')} mm")
    bw = fe.get("by_window", {})
    print("    by window:  " + "   ".join(
        f"{w}={g(bw, w, 'median')}mm" for w in ("0.5s", "1.0s", "2.0s", "5.0s")))
    print("    -> This is error the current pipeline CANNOT avoid, no matter how good HaMeR is.")
    print("    -> Growth with window length = cumulative drift. Compare against your policy's")
    print("       action-chunk horizon and against grasp tolerance (~5-10 mm).")

    if g(summary, "hamer_camera_frame_mm", "mpjpe"):
        print("\n[4] HaMeR camera-frame error")
        for k in ("mpjpe", "root", "root_rel_mpjpe", "pa_mpjpe", "xy_rms", "z_rms"):
            print(f"    {k:<16} {g(summary,'hamer_camera_frame_mm',k,'median')} mm")
        print("    -> root >> root_rel and z_rms >> xy_rms both argue for a 2D-reprojection")
        print("       data term plus a weak depth prior, not an isotropic 3D L2 term.")

    mid = sigmas[len(sigmas) // 2]
    split = g(summary, "world_error_split", f"at_sigma_{mid:g}f")
    if split and split.get("jitter_frac"):
        print(f"\n[5] World-frame error decomposition (sigma={mid:g} frames)  <-- headroom")
        print(f"    const  {g(split,'const_frac','median')}   "
              f"drift  {g(split,'drift_frac','median')}   "
              f"jitter {g(split,'jitter_frac','median')}")
        print(f"    W-MPJPE {g(summary,'world_error_split','w_mpjpe_mm','median')} mm "
              f"(offset-corrected {g(summary,'world_error_split','w_mpjpe_offset_corrected_mm','median')} mm)")
        print("    -> jitter_frac is the ONLY part a temporal smoothness prior removes.")
        print("       < ~0.15 => the optimizer buys you almost nothing; reposition the paper.")

    sc = summary.get("smoothing_ceiling_mm", {})
    if sc.get("best_world_frame"):
        print("\n[6] Smoothing ceiling, scored as world-frame MPJPE  <-- core hypothesis")
        print(f"    unsmoothed        {g(sc,'unsmoothed','median')} mm")
        print(f"    best cam-frame    {g(sc,'best_cam_frame','median')} mm   (~ current pipeline)")
        print(f"    best world-frame  {g(sc,'best_world_frame','median')} mm   (~ proposed)")
        print(f"    world - cam       {g(sc,'world_minus_cam','median')} mm   (negative = world wins)")
        print(f"    world - cam, slow frames only  {g(sc,'world_minus_cam_slow_frames','median')} mm")
        print("    -> If neither is clearly negative, the central claim does not hold and")
        print("       no optimizer engineering will rescue it. Expect the slow-frame margin")
        print("       to be the larger of the two; that is the one worth reporting.")

    bc = summary.get("bone_cv", {})
    if bc.get("gt") or bc.get("hamer"):
        print("\n[7] Bone-length CV")
        print(f"    GT {g(bc,'gt','median')}   HaMeR {g(bc,'hamer','median')}   "
              "(GT is the floor; lambda_3 can only close the gap between them)")

    print(f"\n{line}\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(ep: Episode, side: str, fps: float, out_dir: Path) -> None:
    """Per-episode diagnostic figures for the one episode we inspect closely."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    ! matplotlib unavailable; skipping plots")
        return

    valid = ep.gt_valid[side]
    if valid.sum() < 30:
        return
    T_w2c = np.linalg.inv(ep.T_c2w)
    gt_w = ep.gt_world[side]
    gt_c = transform_points(T_w2c, gt_w)
    t = np.arange(ep.n_frames) / fps
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fig A: index fingertip (joint 8) in world vs camera frame -- the motivation figure.
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, np.where(valid, gt_w[:, 8, i], np.nan), label="GT world", lw=1.8)
        ax.plot(t, np.where(valid, gt_c[:, 8, i], np.nan), label="GT camera frame", lw=1.2, alpha=0.8)
        if side in ep.hamer_cam:
            n = len(ep.hamer_cam[side])
            hv = ep.hamer_valid[side]
            hw = transform_points(ep.T_c2w[:n], ep.hamer_cam[side])
            ax.plot(t[:n], np.where(hv, hw[:, 8, i], np.nan), label="HaMeR->world", lw=1.0, alpha=0.8)
        ax.set_ylabel("XYZ"[i] + " (m)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{ep.name}/{side}: index fingertip, world vs camera frame")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_traj_{ep.name}_{side}.png", dpi=140)
    plt.close(fig)

    # Fig B: camera speed alongside apparent/true hand speed ratio.
    dt = np.linalg.norm(np.diff(ep.T_c2w[:, :3, 3], axis=0), axis=1) * fps
    v_w = np.linalg.norm(np.diff(gt_w, axis=0), axis=-1).mean(1) * fps
    v_c = np.linalg.norm(np.diff(gt_c, axis=0), axis=-1).mean(1) * fps
    pair = valid[:-1] & valid[1:]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(t[1:], dt, lw=1.0)
    axes[0].set_ylabel("cam speed (m/s)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t[1:], np.where(pair, v_w, np.nan), label="true (world)", lw=1.2)
    axes[1].plot(t[1:], np.where(pair, v_c, np.nan), label="apparent (camera)", lw=1.2)
    axes[1].set_ylabel("hand speed (m/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[0].set_title(f"{ep.name}/{side}: camera motion contaminates the camera-frame signal")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_speed_{ep.name}_{side}.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test on synthetic data with known answers
# ---------------------------------------------------------------------------

def _synth_hand_template() -> np.ndarray:
    """A rigid 21-joint hand in MANO21 order. Bone lengths are exactly constant."""
    tpl = np.zeros((21, 3))
    for finger in range(5):
        base_x = -0.040 + 0.020 * finger
        for j in range(4):
            tpl[1 + finger * 4 + j] = [base_x, 0.030 + 0.024 * j, 0.0]
    return tpl


def _look_at_c2w(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world in OpenCV convention (x right, y down, z forward)."""
    up = np.array([0.0, 1.0, 0.0])
    z = target - eye
    z /= max(np.linalg.norm(z), 1e-9)
    x = np.cross(z, up)
    x /= max(np.linalg.norm(x), 1e-9)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def _synth_episode(n: int, moving_camera: bool, with_hamer: bool, seed: int,
                   fps: float = 30.0, static_hand: bool = False) -> Episode:
    """Build an Episode with exactly known GT (and optionally noised 'HaMeR')."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    tpl = _synth_hand_template()

    # Smooth hand motion in the world frame.
    ang = np.zeros(n) if static_hand else 0.5 * np.sin(2 * np.pi * 0.20 * t)
    ca, sa = np.cos(ang), np.sin(ang)
    R_hand = np.zeros((n, 3, 3))
    R_hand[:, 0, 0], R_hand[:, 0, 1] = ca, -sa
    R_hand[:, 1, 0], R_hand[:, 1, 1] = sa, ca
    R_hand[:, 2, 2] = 1.0
    p_hand = np.stack([
        0.15 * np.sin(2 * np.pi * 0.15 * t),
        0.10 * np.sin(2 * np.pi * 0.23 * t),
        0.60 + 0.05 * np.sin(2 * np.pi * 0.11 * t),
    ], axis=1)
    if static_hand:
        p_hand = np.repeat([[0.0, 0.0, 0.60]], n, axis=0)
    world = np.einsum("nij,kj->nki", R_hand, tpl) + p_hand[:, None, :]

    # Camera: slow drift plus a 2 Hz head bob, gaze locked on the wrist.
    if moving_camera:
        eyes = np.stack([
            0.25 * np.sin(2 * np.pi * 0.10 * t) + 0.030 * np.sin(2 * np.pi * 2.0 * t),
            0.10 * np.sin(2 * np.pi * 0.13 * t) + 0.030 * np.sin(2 * np.pi * 1.7 * t),
            0.020 * np.sin(2 * np.pi * 0.17 * t),
        ], axis=1)
    else:
        eyes = np.zeros((n, 3))
    T_c2w = np.stack([_look_at_c2w(eyes[i], p_hand[i] if moving_camera else np.array([0, 0, 0.6]))
                      for i in range(n)])

    K = np.array([[700.0, 0.0, 960.0], [0.0, 700.0, 540.0], [0.0, 0.0, 1.0]])
    ep = Episode(
        name=f"synth_{'moving' if moving_camera else 'static'}",
        K=K, T_c2w=T_c2w,
        gt_world={s: world.copy() for s in ("left", "right")},
        gt_valid={s: np.ones(n, dtype=bool) for s in ("left", "right")},
        img_wh=(1920, 1080),
    )

    if with_hamer:
        T_w2c = np.linalg.inv(T_c2w)
        gt_cam = transform_points(T_w2c, world)
        e_const = rng.normal(0.0, 0.008, size=(1, 21, 3))
        e_drift = gaussian_lowpass(rng.normal(0.0, 1.0, size=(n, 21, 3)), 25.0)
        e_drift *= 0.010 / max(float(np.sqrt((e_drift ** 2).mean())), 1e-9)
        e_jitter = rng.normal(0.0, 0.012, size=(n, 21, 3))
        ham_cam = gt_cam + e_const + e_drift + e_jitter

        uvw = gt_cam @ K.T
        uv = uvw[..., :2] / uvw[..., 2:3] + rng.normal(0.0, 1.0, size=(n, 21, 2))
        valid = np.ones(n, dtype=bool)
        valid[rng.choice(n, size=max(1, n // 30), replace=False)] = False
        for s in ("left", "right"):
            ep.hamer_cam[s] = ham_cam.copy()
            ep.hamer_2d[s] = uv.copy()
            ep.hamer_valid[s] = valid.copy()
    return ep


def run_selftest(fps: float = 30.0) -> int:
    """Validate every block against synthetic data whose answers we know a priori.

    Run this on the server before pointing the script at real data: it exercises the
    same code paths and catches convention/units/plumbing errors immediately.
    """
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("Self-test on synthetic data (200 frames @ 30 fps)\n")

    # --- static camera: the fixed-extrinsics assumption is exactly correct ---
    ep_s = _synth_episode(200, moving_camera=False, with_hamer=False, seed=0, fps=fps)
    fx_s = block3_fixed_extrinsics(ep_s, fps, [1.0])["right"]
    check("block3 static camera residual ~ 0",
          fx_s["residual_all_joints_mm"]["mean"] < 0.05,
          f"{fx_s['residual_all_joints_mm']['mean']} mm (expect <0.05)")

    bone_s = block7_bone_lengths(ep_s)["right"]
    check("block7 rigid template has zero bone CV",
          bone_s["gt_bone_cv"] < 1e-6,
          f"CV={bone_s['gt_bone_cv']} (expect ~0)")

    # --- moving camera ---
    ep_m = _synth_episode(200, moving_camera=True, with_hamer=True, seed=1, fps=fps)

    conv = block0_convention(ep_m)
    ok_conv = (conv["assumed"]["in_image_frac"] >= conv["flipped"]["in_image_frac"]
               and conv["assumed"]["reproj_median_px_after_similarity_fit"] < 5.0)
    check("block0 identifies the correct camera convention", ok_conv,
          f"assumed in-image={conv['assumed']['in_image_frac']} "
          f"reproj={conv['assumed']['reproj_median_px_after_similarity_fit']}px vs "
          f"flipped in-image={conv['flipped']['in_image_frac']}")

    cm = block1_camera_motion(ep_m, fps)
    check("block1 reports nonzero camera motion", cm["path_length_m"] > 0.1,
          f"path={cm['path_length_m']} m, lin={cm['lin_speed_mps']['mean']} m/s")

    scond = cm["slam_conditioning"]
    check("block1 reports SLAM conditioning",
          np.isfinite(scond["rot_over_trans_flow_median"]) and scond["max_baseline_over_depth"] > 0,
          f"baseline/depth={scond['max_baseline_over_depth']} "
          f"rot/trans={scond['rot_over_trans_flow_median']}")

    av = block2_apparent_vs_true(ep_m, fps)["right"]
    check("block2 apparent speed exceeds true speed",
          av["apparent_over_true"]["median"] > 1.0,
          f"ratio median={av['apparent_over_true']['median']} (expect >1)")
    check("block2 classifies a moving hand as active",
          av["is_idle_hand"] is False and len(av["by_speed"]) >= 1,
          f"extent={av['hand_extent_m']} m, strata={list(av['by_speed'])}")

    # An idle hand under a moving camera is the case that matters most: its apparent
    # motion is almost entirely camera motion, so its action labels are near-pure noise.
    ep_i = _synth_episode(200, moving_camera=True, with_hamer=False, seed=2, fps=fps,
                          static_hand=True)
    av_i = block2_apparent_vs_true(ep_i, fps)["right"]
    check("block2 flags an idle hand and its inflated ratio",
          av_i["is_idle_hand"] is True and av_i["apparent_over_true"]["median"] > 3.0,
          f"extent={av_i['hand_extent_m']} m, ratio={av_i['apparent_over_true']['median']} (expect >3)")

    fx_m = block3_fixed_extrinsics(ep_m, fps, [1.0, 2.0])["right"]
    check("block3 moving camera residual is large",
          fx_m["residual_all_joints_mm"]["mean"] > 10.0,
          f"{fx_m['residual_all_joints_mm']['mean']} mm (expect >10)")

    b4 = block4_hamer_camera_frame(ep_m)["right"]
    # Injected error: const 8mm + drift 10mm + jitter 12mm per axis -> ~30mm MPJPE.
    check("block4 recovers the injected error magnitude",
          10.0 < b4["mpjpe_mm"] < 80.0,
          f"MPJPE={b4['mpjpe_mm']} mm (expect 10-80)")
    check("block4 scale sanity: wrist depth ratio ~ 1",
          0.9 < b4["wrist_depth_ratio_hamer_over_gt"] < 1.1,
          f"ratio={b4['wrist_depth_ratio_hamer_over_gt']}")

    b5 = block5_world_error_split(ep_m, [2, 5, 10])["right"]
    split = b5["energy_split"]["sigma_5f"]
    check("block5 finds substantial high-frequency jitter",
          split["jitter_frac"] > 0.15,
          f"const={split['const_frac']} drift={split['drift_frac']} jitter={split['jitter_frac']}")
    # The const split is exactly orthogonal, but a Gaussian low-pass is not an
    # orthogonal projection, so drift and jitter carry a small positive cross term
    # that the three fractions do not account for. Expect a sum slightly under 1.
    frac_sum = split["const_frac"] + split["drift_frac"] + split["jitter_frac"]
    check("block5 energy fractions sum to ~1", 0.90 < frac_sum <= 1.01,
          f"sum={round(frac_sum, 3)} (expect 0.90-1.01)")

    b6 = block6_smoothing_ceiling(ep_m, [0, 2, 5, 10, 20], fps)["right"]
    check("block6 smoothing helps at all",
          b6["best_world_frame"]["w_mpjpe_offset_corrected_mm"] < b6["unsmoothed_mm"],
          f"unsmoothed={b6['unsmoothed_mm']} -> world={b6['best_world_frame']['w_mpjpe_offset_corrected_mm']} mm")
    check("block6 world-frame smoothing beats camera-frame smoothing",
          b6["world_minus_cam_mm"] < 0.0,
          f"world-cam={b6['world_minus_cam_mm']} mm "
          f"(cam={b6['best_cam_frame']['w_mpjpe_offset_corrected_mm']}, "
          f"world={b6['best_world_frame']['w_mpjpe_offset_corrected_mm']})")

    # --- end-to-end aggregation path ---
    rec = {
        "episode": ep_m.name, "index": 0, "has_hamer": True,
        "convention": conv, "camera_motion": cm,
        "apparent_vs_true": {"right": av, "left": av},
        "fixed_extrinsics": {"right": fx_m, "left": fx_m},
        "bone_lengths": block7_bone_lengths(ep_m),
        "hamer_camera_frame": {"right": b4, "left": b4},
        "world_error_split": {"right": b5, "left": b5},
        "smoothing_ceiling": {"right": b6, "left": b6},
    }
    try:
        summary = summarize([rec], [2, 5, 10])
        json.dumps(summary)
        print_report(summary, [rec], [2, 5, 10])
        check("summarize + print_report + JSON round-trip", True, "ok")
    except Exception as exc:  # noqa: BLE001 - self-test should report, not raise
        check("summarize + print_report + JSON round-trip", False, f"{type(exc).__name__}: {exc}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'=' * 78}")
    print(f"SELF-TEST: {len(results) - n_fail}/{len(results)} passed")
    if n_fail:
        print("Failures above indicate a bug in this script, not in your data.")
    print("=" * 78)
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantify how much ego camera motion corrupts the hand-pose pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--egodex-root", type=Path, default=Path("/mnt/r/DATA/EgoDex/test"),
                   help="Root holding <task>/<id>.hdf5")
    p.add_argument("--task", type=str, default=None, help="EgoDex task name (required unless --selftest)")
    p.add_argument("--max-episodes", type=int, default=10)
    p.add_argument("--processed-root", type=Path, default=None,
                   help="Processed demo tree (e.g. .../egodex_<task>). Enables blocks 4-7. "
                        "Episode i is looked up at <processed-root>/<i>, matching convert_egodex.py.")
    p.add_argument("--prefer-3d", action="store_true",
                   help="Use hand_data_3d_{side}.npz (post depth-alignment) when present")
    p.add_argument("--out", type=Path, default=Path("b/out/diag"))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--gt-conf-thresh", type=float, default=0.5,
                   help="Minimum EgoDex confidences/<side>Hand to trust a GT frame")
    p.add_argument("--camera-convention", choices=["opencv", "opengl"], default="opencv",
                   help="Convention of transforms/camera. 'opencv' assumes z-forward/y-down "
                        "(what convert_egodex.py assumes); 'opengl' applies a y/z flip. "
                        "Block 0 tells you which is right.")
    p.add_argument("--lowpass-sigmas", type=float, nargs="+", default=[0, 2, 5, 10, 20],
                   help="Gaussian sigmas in frames for blocks 5-6. Keep 0 first for the "
                        "unsmoothed reference.")
    p.add_argument("--window-secs", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0],
                   help="Window lengths for the fixed-extrinsics timescale sweep")
    p.add_argument("--plot", action="store_true", help="Save per-episode figures")
    p.add_argument("--selftest", action="store_true",
                   help="Validate every block against synthetic data with known answers, "
                        "then exit. Needs only numpy. Run this first.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.selftest:
        sys.exit(run_selftest(args.fps))
    if args.task is None:
        raise SystemExit("--task is required (or use --selftest)")

    task_dir = args.egodex_root / args.task
    if not task_dir.exists():
        raise SystemExit(f"task directory not found: {task_dir}")

    hdf5_files = sorted((f for f in task_dir.iterdir() if f.suffix == ".hdf5"),
                        key=lambda x: int(x.stem))
    if not hdf5_files:
        raise SystemExit(f"no HDF5 files in {task_dir}")
    hdf5_files = hdf5_files[: args.max_episodes]

    sigmas = list(args.lowpass_sigmas)
    to_opencv = args.camera_convention == "opengl"
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Task {args.task}: {len(hdf5_files)} episodes")
    print(f"Camera convention: {args.camera_convention}"
          f"{' (applying GL->CV flip)' if to_opencv else ''}")
    if args.processed_root is None:
        print("No --processed-root: running GT-only blocks 1-3 (+7 GT floor).")

    per_ep: List[dict] = []
    for idx, hdf5_path in enumerate(hdf5_files):
        try:
            ep = load_gt(hdf5_path, args.gt_conf_thresh, to_opencv)
        except (KeyError, OSError) as exc:
            print(f"  [{idx}] {hdf5_path.stem}: skipped ({exc})")
            continue

        has_hamer = False
        if args.processed_root is not None:
            demo_dir = args.processed_root / str(idx)
            if demo_dir.exists():
                has_hamer = attach_hamer(ep, demo_dir, args.prefer_3d)
                ep.img_wh = probe_video_size(demo_dir)
                if not has_hamer:
                    print(f"  [{idx}] no hand_processor/*.npz under {demo_dir}")
            else:
                print(f"  [{idx}] processed dir missing: {demo_dir}")
        if ep.img_wh is None:
            ep.img_wh = (int(round(2 * ep.K[0, 2])), int(round(2 * ep.K[1, 2])))

        rec: dict = {
            "episode": ep.name,
            "index": idx,
            "has_hamer": has_hamer,
            "convention": block0_convention(ep),
            "camera_motion": block1_camera_motion(ep, args.fps),
            "apparent_vs_true": block2_apparent_vs_true(ep, args.fps),
            "fixed_extrinsics": block3_fixed_extrinsics(ep, args.fps, args.window_secs),
            "bone_lengths": block7_bone_lengths(ep),
        }
        if has_hamer:
            rec["hamer_camera_frame"] = block4_hamer_camera_frame(ep)
            rec["world_error_split"] = block5_world_error_split(ep, [s for s in sigmas if s > 0])
            rec["smoothing_ceiling"] = block6_smoothing_ceiling(ep, sigmas, args.fps)
        per_ep.append(rec)

        fx = rec["fixed_extrinsics"].get("right") or rec["fixed_extrinsics"].get("left")
        fx_mm = fx["residual_all_joints_mm"]["mean"] if fx else None
        print(f"  [{idx}] {ep.name}: {ep.n_frames} frames, "
              f"fixed-extrinsics residual {fx_mm} mm, hamer={has_hamer}")

        if args.plot and idx == 0:
            for side in ("left", "right"):
                make_plots(ep, side, args.fps, args.out)

    if not per_ep:
        raise SystemExit("no episodes processed")

    summary = summarize(per_ep, [s for s in sigmas if s > 0] or [5])
    print_report(summary, per_ep, [s for s in sigmas if s > 0] or [5])

    payload = {"args": {k: str(v) for k, v in vars(args).items()},
               "summary": summary, "episodes": per_ep}
    out_json = args.out / f"diag_{args.task}.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
