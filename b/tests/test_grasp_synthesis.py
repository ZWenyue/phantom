"""Smoke test for IntentProcessor grasp synthesis (Stage A, block 3).

Runs on the already-saved pick_and_place artifacts (object_pcd.npz +
contact_events.npz) so it needs no DINO/SAM2. Verifies:
  * object-PCA fallback path (this demo has no hand data);
  * a synthetic-hand path (thumb-index closing axis + approach direction).

Usage (from phantom/):
    python b/tests/test_grasp_synthesis.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "phantom")))

from phantom.processors.intent_processor import IntentProcessor, FINGERTIP_IDXS

DEMO = "/home/a26160/SRC/phantom/data/processed/pick_and_place/0/intent_processor"


def _make_processor():
    ip = IntentProcessor.__new__(IntentProcessor)  # bypass heavy __init__
    ip.grasp_precontact_window = 8
    ip.grasp_approach_frames = 4
    ip.gripper_max_width = 0.08
    ip.grasp_antipodal_pct = 2.0
    return ip


def _load_pcd_result():
    d = np.load(os.path.join(DEMO, "object_pcd.npz"), allow_pickle=True)
    return {
        "points_robot": list(d["points_robot"]),
        "colors": list(d["colors"]),
        "centroids_robot": d["centroids_robot"],
        "valid": d["valid"],
    }


def _load_contact_result():
    d = np.load(os.path.join(DEMO, "contact_events.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def test_pca_fallback():
    ip = _make_processor()
    pcd = _load_pcd_result()
    contact = _load_contact_result()
    g = ip._synthesize_grasp(pcd, contact, hands={})
    assert g["valid"], "grasp should be valid"
    assert g["source"] == "object_pca"
    R = np.asarray(g["G_rot"])
    assert abs(np.linalg.det(R) - 1.0) < 1e-4, f"R not a rotation, det={np.linalg.det(R)}"
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-4), "R not orthonormal"
    w = float(g["G_width"])
    assert 0.0 < w < 0.5, f"implausible grasp width {w}"
    # antipodal midpoint should equal grasp center.
    cps = np.asarray(g["contact_points"])
    assert np.allclose(0.5 * (cps[0] + cps[1]), g["G_center"], atol=1e-5)
    print(f"[pca] frame={int(g['grasp_frame'])} center={np.round(g['G_center'],3)} "
          f"width={w*100:.1f}cm closing={np.round(g['closing_axis'],2)} "
          f"approach={np.round(g['approach_axis'],2)} det(R)={np.linalg.det(R):.3f}")


def test_synthetic_hand():
    """Inject a synthetic hand whose thumb-index axis defines the closing axis
    and whose motion defines the approach, then check G* honors both."""
    ip = _make_processor()
    pcd = _load_pcd_result()
    contact = _load_contact_result()
    n = len(pcd["valid"])
    gf = ip._select_grasp_frame(pcd, int(contact["grasp_keyframe"]))
    center = np.asarray(pcd["points_robot"][gf]).mean(axis=0)

    # Build a hand: index/thumb straddle the object center along +x; the hand
    # approaches from +z (descending), so approach should resolve to ~ -z.
    fingertips = np.zeros((n, len(FINGERTIP_IDXS), 3), dtype=np.float32)
    detected = np.zeros(n, dtype=bool)
    for t in range(max(0, gf - 6), gf + 1):
        descend = np.array([0, 0, 0.02 * (gf - t)], dtype=np.float32)  # higher earlier
        thumb = center + np.array([0.03, 0, 0]) + descend
        index = center + np.array([-0.03, 0, 0]) + descend
        middle = center + np.array([-0.03, 0.01, 0]) + descend
        fingertips[t] = np.stack([thumb, index, middle])
        detected[t] = True
    hands = {"right": {"fingertips": fingertips, "detected": detected,
                        "aperture": np.full(n, 0.06, np.float32)}}

    g = ip._synthesize_grasp(pcd, contact, hands=hands)
    assert g["valid"] and g["source"] == "hand_anchored", g["source"]
    closing = np.asarray(g["closing_axis"])
    approach = np.asarray(g["approach_axis"])
    # closing axis should be ~ +/- x (thumb-index straddled x).
    assert abs(abs(closing[0]) - 1.0) < 0.35, f"closing not ~x: {closing}"
    # approach should point downward (hand descended in +z, moved toward -z).
    assert approach[2] < -0.5, f"approach not downward: {approach}"
    R = np.asarray(g["G_rot"])
    assert abs(np.linalg.det(R) - 1.0) < 1e-4
    print(f"[hand] frame={int(g['grasp_frame'])} closing={np.round(closing,2)} "
          f"approach={np.round(approach,2)} width={float(g['G_width'])*100:.1f}cm")


if __name__ == "__main__":
    test_pca_fallback()
    test_synthetic_hand()
    print("OK: grasp synthesis smoke tests passed")
