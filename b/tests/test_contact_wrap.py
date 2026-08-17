"""Unit tests for wrap-aware fingertip contact (IntentProcessor._detect_contacts).

Usage (from phantom/):
    python b/tests/test_contact_wrap.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "phantom")))

from phantom.processors.intent_processor import FINGERTIP_IDXS, IntentProcessor


def _proc(**kw):
    ip = IntentProcessor.__new__(IntentProcessor)
    ip.contact_dist_in = 0.03
    ip.contact_dist_out = 0.05
    ip.contact_min_valid = 5
    ip.contact_min_run = 3
    ip.contact_motion_thresh = 0.004
    ip.grasp_window = 3
    ip.release_window = 3
    ip.contact_xy_inflate = 0.06
    ip.target_hand = "right"
    for k, v in kw.items():
        setattr(ip, k, v)
    return ip


def _cloud_at(center, n=80, half=(0.015, 0.015, 0.004)):
    rng = np.random.default_rng(0)
    half = np.asarray(half, np.float32)
    return (center + rng.uniform(-1.0, 1.0, size=(n, 3)) * half).astype(np.float32)


def _hand(n, fps_cam, detected=None):
    detected = np.ones(n, dtype=bool) if detected is None else detected
    ap = np.full(n, 0.04, dtype=np.float32)
    # Robot-frame copy unused for scoring when cam is present.
    return {
        "right": {
            "fingertips": fps_cam.copy(),
            "fingertips_cam": fps_cam,
            "detected": detected,
            "aperture": ap,
        }
    }


def _pcd(objs_cam):
    n = len(objs_cam)
    dummy_rf = [p.copy() for p in objs_cam]
    cents = np.stack([p.mean(0) if len(p) else np.zeros(3) for p in dummy_rf])
    return {
        "points_cam": objs_cam,
        "points_robot": dummy_rf,
        "centroids_robot": cents.astype(np.float32),
        "valid": np.ones(n, dtype=bool),
    }


def test_wrap_side_contact():
    """Thumb 5 cm in X, 3 mm in Z from the visible top face → contact via |Δz|."""
    n = 40
    obj0 = np.array([0.0, 0.0, 0.50], np.float32)
    clouds = [_cloud_at(obj0) for _ in range(n)]
    fps = np.zeros((n, len(FINGERTIP_IDXS), 3), np.float32)
    # Hover far, then slide in from the side at the same depth.
    for t in range(n):
        lat = 0.20 if t < 15 else 0.06
        fps[t, 0] = [lat, 0.0, 0.503]  # thumb
        fps[t, 1] = [lat + 0.02, 0.0, 0.50]
        fps[t, 2] = [lat + 0.03, 0.0, 0.50]
    ip = _proc()
    c = ip._detect_contacts(_pcd(clouds), _hand(n, fps))
    assert c["source"] == "fingertip_wrap"
    assert int(c["grasp_keyframe"]) >= 15, c["grasp_keyframe"]
    assert c["d_finger_obj"][20] < 0.03, float(c["d_finger_obj"][20])
    assert c["d_eucl"][20] > 0.04, float(c["d_eucl"][20])
    assert bool(c["contact"][20])
    print(f"[wrap-side] grasp_kf={int(c['grasp_keyframe'])} "
          f"score={c['d_finger_obj'][20]*100:.1f}cm eucl={c['d_eucl'][20]*100:.1f}cm")


def test_overhead_does_not_wrap():
    """10 cm above the object: dxy small but |Δz| large → stay free."""
    n = 30
    obj0 = np.array([0.0, 0.0, 0.50], np.float32)
    clouds = [_cloud_at(obj0) for _ in range(n)]
    fps = np.zeros((n, len(FINGERTIP_IDXS), 3), np.float32)
    for t in range(n):
        fps[t, :] = [0.0, 0.0, 0.40]  # 10 cm closer to camera
    ip = _proc()
    c = ip._detect_contacts(_pcd(clouds), _hand(n, fps))
    assert int(c["contact"].sum()) == 0, int(c["contact"].sum())
    assert c["d_finger_obj"][10] > 0.08, float(c["d_finger_obj"][10])
    print(f"[overhead] contacts=0  score={c['d_finger_obj'][10]*100:.1f}cm")


def test_true_surface_still_euclidean():
    """Fingertip 2 mm from the cloud: wrapScore == Euclidean, still contacts."""
    n = 30
    obj0 = np.array([0.0, 0.0, 0.50], np.float32)
    clouds = [_cloud_at(obj0, half=(0.01, 0.01, 0.004)) for _ in range(n)]
    fps = np.zeros((n, len(FINGERTIP_IDXS), 3), np.float32)
    for t in range(n):
        z = 0.70 if t < 8 else 0.502
        fps[t, :] = [0.0, 0.0, z]
    ip = _proc()
    c = ip._detect_contacts(_pcd(clouds), _hand(n, fps))
    assert c["d_finger_obj"][15] < 0.03
    assert abs(float(c["d_finger_obj"][15] - c["d_eucl"][15])) < 0.005
    assert int(c["grasp_keyframe"]) >= 0
    print(f"[surface] grasp_kf={int(c['grasp_keyframe'])} "
          f"score={c['d_finger_obj'][15]*100:.2f}cm")


def test_xy_inflate_zero_is_pure_euclidean():
    n = 20
    obj0 = np.array([0.0, 0.0, 0.50], np.float32)
    clouds = [_cloud_at(obj0) for _ in range(n)]
    fps = np.zeros((n, len(FINGERTIP_IDXS), 3), np.float32)
    for t in range(n):
        fps[t, 0] = [0.05, 0.0, 0.503]
        fps[t, 1:] = [0.06, 0.0, 0.50]
    ip = _proc(contact_xy_inflate=0.0)
    c = ip._detect_contacts(_pcd(clouds), _hand(n, fps))
    assert c["source"] == "fingertip"
    assert np.allclose(c["d_finger_obj"], c["d_eucl"], equal_nan=True)
    assert int(c["contact"].sum()) == 0
    print("[inflate=0] pure euclidean, no wrap contact")


if __name__ == "__main__":
    test_wrap_side_contact()
    test_overhead_does_not_wrap()
    test_true_surface_still_euclidean()
    test_xy_inflate_zero_is_pure_euclidean()
    print("ok")
