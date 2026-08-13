"""Smoke test for IntentProcessor intent integration (Stage A, block 4).

Runs on the already-saved pick_and_place artifacts (object_pcd.npz +
contact_events.npz) so it needs no DINO/SAM2. Verifies:
  * object-relative EE targets during the grasp episode (PCA-fallback grasp);
  * hand-following EE targets outside the episode (synthetic-hand path);
  * gap-free target trajectory, orthonormal R_target, phase-dependent weights,
    and the gripper command derived from the contact FSM.

Usage (from phantom/):
    python b/tests/test_intent_integration.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "phantom")))

from phantom.processors.intent_processor import (
    IntentProcessor, FINGERTIP_IDXS, PHASE_GRASP, PHASE_RELEASE,
)

DEMO = "/home/a26160/SRC/phantom/data/processed/pick_and_place/0/intent_processor"


def _make_processor():
    ip = IntentProcessor.__new__(IntentProcessor)  # bypass heavy __init__
    # block 3 params
    ip.grasp_precontact_window = 8
    ip.grasp_approach_frames = 4
    ip.gripper_max_width = 0.08
    ip.grasp_antipodal_pct = 2.0
    # block 4 params
    ip.wp_free = 1.0
    ip.wr_free = 0.1
    ip.wp_grasp = 5.0
    ip.wr_grasp = 5.0
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


def _check_common(intent):
    n = len(intent["p_target"])
    assert intent["p_valid"].all(), "EE target trajectory should be gap-free after fill"
    R = np.asarray(intent["R_target"])
    assert R.shape == (n, 3, 3)
    # every R_target orthonormal.
    for t in (0, n // 2, n - 1):
        Rt = R[t]
        assert np.allclose(Rt.T @ Rt, np.eye(3), atol=1e-4), f"R_target[{t}] not orthonormal"
    phase = np.asarray(intent["phase"])
    wp = np.asarray(intent["w_p"]); wr = np.asarray(intent["w_r"])
    key = np.isin(phase, [PHASE_GRASP, PHASE_RELEASE])
    if key.any():
        assert np.all(wp[key] == 5.0), "grasp/release should get high w_p"
        assert np.all(wr[key] == 5.0), "grasp/release should get high w_r"
    free = ~key
    if free.any():
        assert np.all(wp[free] == 1.0), "free/transport should get low w_p"


def test_object_relative_pca():
    ip = _make_processor()
    pcd = _load_pcd_result()
    contact = _load_contact_result()
    grasp = ip._synthesize_grasp(pcd, contact, hands={})
    assert grasp["valid"]
    intent = ip._integrate_intent(pcd, contact, grasp, hands={})
    _check_common(intent)
    assert intent["grasp_valid"]
    src = np.asarray(intent["p_source"], dtype=object)
    gk = int(contact["grasp_keyframe"]); rk = int(contact["release_keyframe"])
    # inside the grasp episode targets should be object-relative.
    episode = list(range(gk, rk + 1))
    n_obj_rel = sum(src[t] == "object_relative" for t in episode)
    assert n_obj_rel > 0, "expected object-relative targets inside the grasp episode"
    # gripper closes during transport (object_motion demo).
    gw = np.asarray(intent["gripper_width"]); gc = np.asarray(intent["g_closed"], dtype=bool)
    assert np.allclose(gw[gc], float(grasp["G_width"]), atol=1e-5)
    assert np.allclose(gw[~gc], ip.gripper_max_width, atol=1e-5)
    print(f"[pca] frames={len(src)} obj_rel_in_episode={n_obj_rel}/{len(episode)} "
          f"G_width={float(grasp['G_width'])*100:.1f}cm "
          f"kf={gk}/{rk} sources={set(map(str, src))}")


def test_hand_follow():
    """Inject a synthetic hand before the grasp so free-phase targets follow it."""
    ip = _make_processor()
    pcd = _load_pcd_result()
    contact = _load_contact_result()
    n = len(pcd["valid"])
    gk = int(contact["grasp_keyframe"])
    gf = ip._select_grasp_frame(pcd, gk)
    center = np.asarray(pcd["points_robot"][gf]).mean(axis=0)

    fingertips = np.zeros((n, len(FINGERTIP_IDXS), 3), dtype=np.float32)
    detected = np.zeros(n, dtype=bool)
    # a hand present a few frames before the grasp, offset from the object.
    for t in range(max(0, gf - 6), gf + 1):
        descend = np.array([0, 0, 0.02 * (gf - t)], dtype=np.float32)
        thumb = center + np.array([0.03, 0, 0.05]) + descend
        index = center + np.array([-0.03, 0, 0.05]) + descend
        middle = center + np.array([-0.03, 0.01, 0.05]) + descend
        fingertips[t] = np.stack([thumb, index, middle])
        detected[t] = True
    hands = {"right": {"fingertips": fingertips, "detected": detected,
                       "aperture": np.full(n, 0.06, np.float32)}}

    grasp = ip._synthesize_grasp(pcd, contact, hands=hands)
    intent = ip._integrate_intent(pcd, contact, grasp, hands=hands)
    _check_common(intent)
    src = np.asarray(intent["p_source"], dtype=object)
    # frames with the synthetic hand that are outside the episode should follow it.
    rk = int(contact["release_keyframe"])
    pre = [t for t in range(max(0, gf - 6), gf) if not (gk <= t <= rk)]
    n_hand = sum(src[t] == "hand" for t in pre)
    assert n_hand > 0, f"expected hand-following targets pre-grasp, got {[src[t] for t in pre]}"
    # the hand-followed target should equal the thumb-index midpoint.
    t0 = pre[0]
    mid = 0.5 * (fingertips[t0, 0] + fingertips[t0, 1])
    assert np.allclose(intent["p_target"][t0], mid, atol=1e-5)
    print(f"[hand] hand_frames_pre_grasp={n_hand}/{len(pre)} "
          f"p_target[{t0}]={np.round(intent['p_target'][t0],3)}")


if __name__ == "__main__":
    test_object_relative_pca()
    test_hand_follow()
    print("OK: intent integration smoke tests passed")
