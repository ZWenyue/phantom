"""Unit tests for hand-anchored seed selection / motion QA (no SAM2).

Usage (from phantom/):
    python b/tests/test_hand_anchored_tracking.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "phantom")))

from phantom.processors.intent_processor import IntentProcessor


def _hand_traj(n=80, grasp_t=40):
    """Synthetic pinch: approach, close at ``grasp_t``, hold, then open."""
    detected = np.ones(n, dtype=bool)
    aperture = np.full(n, 0.12, dtype=np.float32)
    tips = np.zeros((n, 3), dtype=np.float32)
    for t in range(n):
        tips[t, 0] = 0.01 * t if t < grasp_t else 0.01 * grasp_t
        if abs(t - grasp_t) <= 2:
            aperture[t] = 0.03
        elif t > grasp_t:
            aperture[t] = 0.03 if t < grasp_t + 20 else 0.12
        else:
            # linearly close over the last 10 approach frames
            if t > grasp_t - 10:
                aperture[t] = 0.12 - 0.09 * (t - (grasp_t - 10)) / 10.0
    return detected, aperture, tips


def test_select_seed_near_grasp():
    det, ap, tips = _hand_traj()
    seed = IntentProcessor._select_seed_frame_from_hand(
        det, ap, tips, approach_window=8, hold_window=8,
    )
    assert seed is not None
    assert 35 <= int(seed) <= 45, f"seed={seed} not near grasp_t=40"


def test_select_seed_fallback_min_aperture():
    n = 30
    det = np.ones(n, dtype=bool)
    ap = np.linspace(0.15, 0.04, n).astype(np.float32)
    tips = np.zeros((n, 3), dtype=np.float32)  # no speed drop → fallback
    seed = IntentProcessor._select_seed_frame_from_hand(det, ap, tips)
    assert seed is not None
    assert ap[seed] == ap.min() or seed >= int(0.1 * n)


def test_select_seed_none_when_undetected():
    n = 20
    det = np.zeros(n, dtype=bool)
    ap = np.ones(n, dtype=np.float32)
    tips = np.zeros((n, 3), dtype=np.float32)
    assert IntentProcessor._select_seed_frame_from_hand(det, ap, tips) is None


def test_inhand_window():
    det, ap, _ = _hand_traj()
    inhand = IntentProcessor._inhand_from_aperture(ap, det, seed_idx=40, slack=0.03)
    assert inhand[40]
    assert inhand[50]
    assert not inhand[10]
    assert not inhand[75]


def test_vel_corr_attached():
    n = 40
    tip = np.cumsum(np.ones((n, 3), dtype=np.float32) * 0.01, axis=0)
    obj = tip + np.array([0.02, 0.0, 0.0], dtype=np.float32)  # rigid offset
    tip_v = np.diff(tip, axis=0, prepend=tip[:1])
    obj_v = np.diff(obj, axis=0, prepend=obj[:1])
    corr = IntentProcessor._vel_corr(obj_v[5:], tip_v[5:])
    assert corr > 0.99, corr


def test_vel_corr_static_vs_moving():
    n = 40
    tip_v = np.ones((n, 3), dtype=np.float32) * 0.01
    obj_v = np.zeros((n, 3), dtype=np.float32)
    corr = IntentProcessor._vel_corr(obj_v, tip_v)
    assert abs(corr) < 0.2 or not np.isfinite(corr)


def test_clip_uv():
    pts = np.array([[10.0, 10.0], [-1.0, 5.0], [50.0, 200.0], [np.nan, 0.0]])
    out = IntentProcessor._clip_uv(pts, h=100, w=80)
    assert out.shape == (1, 2)
    assert np.allclose(out[0], [10.0, 10.0])


def test_squeeze_video_mask():
    m = np.zeros((1, 4, 5), dtype=bool)
    m[0, 1, 2] = True
    out = IntentProcessor._squeeze_video_mask(m)
    assert out.shape == (4, 5)
    assert out.dtype == np.uint8
    assert out[1, 2] == 1


def test_find_transport_window_prefers_net_disp():
    """A long fidget then a shorter carry: pick the carry (larger net disp)."""
    n = 80
    det = np.ones(n, dtype=bool)
    tips = np.zeros((n, 3), dtype=np.float32)
    # fidget: 30 frames oscillating 2 cm (high speed, tiny net)
    for t in range(1, 30):
        tips[t, 0] = 0.02 if t % 2 else 0.0
    # pause
    for t in range(30, 40):
        tips[t] = tips[29]
    # carry: 20 frames translating 20 cm
    for t in range(40, 60):
        tips[t] = tips[39] + np.array([(t - 40) * 0.01, 0.0, 0.0], dtype=np.float32)
    for t in range(60, n):
        tips[t] = tips[59]
    win = IntentProcessor._find_transport_window(det, tips, move_vel=0.005, move_run=4, min_net=0.05)
    assert win is not None
    w0, w1 = win
    assert w0 >= 38 and w1 <= 62, f"window={win} should cover the carry ~40-59"


def test_find_transport_window_none_when_still():
    n = 30
    det = np.ones(n, dtype=bool)
    tips = np.zeros((n, 3), dtype=np.float32)
    assert IntentProcessor._find_transport_window(det, tips, min_net=0.05) is None


def test_sample_candidate_frames():
    det = np.ones(20, dtype=bool)
    det[5] = False
    cands = IntentProcessor._sample_candidate_frames(0, 19, k=5, detected=det)
    assert len(cands) >= 4
    assert all(0 <= c <= 19 for c in cands)
    assert 5 not in cands


def test_score_probe_prefers_comotion():
    n = 17
    tips = np.cumsum(np.ones((n, 3), dtype=np.float32) * 0.01, axis=0)
    attached = tips + np.array([0.03, 0.0, 0.0], dtype=np.float32)
    static = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    s_att, c_att, _ = IntentProcessor._score_probe_window(attached, tips)
    s_stat, c_stat, _ = IntentProcessor._score_probe_window(static, tips)
    assert s_att > s_stat, (s_att, s_stat, c_att, c_stat)
    assert c_att > 0.9
    assert abs(c_stat) < 0.2 or not np.isfinite(c_stat)


def test_grasp_disp_qa_fails_on_static_object():
    ip = IntentProcessor.__new__(IntentProcessor)
    ip.min_object_pts = 1
    ip.track_motion_corr_min = 0.3
    ip.track_static_vel_max = 0.02
    ip.track_min_disp = 0.05
    ip.track_backend = "hand_sam2"
    ip._seed_low_conf = False
    n, h, w = 20, 8, 8
    masks = np.ones((n, h, w), dtype=np.uint8)
    centroids = np.tile(np.array([0.5, 0.0, 0.0], dtype=np.float32), (n, 1))
    pcd = {
        "centroids_robot": centroids,
        "valid": np.ones(n, dtype=bool),
    }
    phase = np.zeros(n, dtype=int)
    phase[:10] = 1  # grasp/transport on a static box
    contact = {"phase": phase}
    tips = np.cumsum(np.ones((n, 3, 3), dtype=np.float32) * 0.01, axis=0)
    hands = {
        "right": {
            "detected": np.ones(n, dtype=bool),
            "fingertips": tips,
        }
    }
    ip.target_hand = "right"
    qa = ip._track_quality_gate(masks, pcd, contact, hands, seed_idx=2)
    assert not qa["pass"], qa
    assert any("grasp_disp" in r for r in qa["reasons"])


def test_hand_region_covers_palm_not_pinch_gap():
    """Skeleton+palm hull must not fill the pinch gap the way a 21-pt hull would."""
    h, w = 96, 96
    kpts = np.full((21, 2), np.nan, dtype=np.float32)
    # Palm / MCPs clustered on the left.
    palm = {
        0: (22.0, 60.0),
        1: (18.0, 50.0),
        5: (26.0, 48.0),
        9: (28.0, 54.0),
        13: (26.0, 60.0),
        17: (20.0, 64.0),
    }
    for i, xy in palm.items():
        kpts[i] = xy
    # Thumb goes up, index goes right — pinch midpoint around (40, 34).
    kpts[2] = (18.0, 40.0)
    kpts[3] = (18.0, 28.0)
    kpts[4] = (20.0, 16.0)
    kpts[6] = (40.0, 46.0)
    kpts[7] = (54.0, 44.0)
    kpts[8] = (68.0, 42.0)
    for i, src in [(10, 9), (11, 9), (12, 9), (14, 13), (15, 13), (16, 13), (18, 17), (19, 17), (20, 17)]:
        kpts[i] = kpts[src]
    region = IntentProcessor._hand_region_mask(kpts, (h, w), dilate_px=7)
    assert region is not None
    assert int(region.sum()) > 50
    assert region[60, 22] == 1  # palm
    mid = 0.5 * (kpts[4] + kpts[8])
    my, mx = int(round(float(mid[1]))), int(round(float(mid[0])))
    assert region[my, mx] == 0, "pinch gap should be empty (not a filled 21-pt hull)"


def test_overlap_veto_rejects_hand_mask():
    region = np.zeros((32, 32), dtype=np.uint8)
    region[8:24, 8:24] = 1
    out, ov = IntentProcessor._apply_hand_overlap_veto(region, region, max_overlap=0.5, min_pts=10)
    assert out is None
    assert ov > 0.9


def test_overlap_veto_keeps_adjacent_object():
    region = np.zeros((32, 32), dtype=np.uint8)
    region[0:32, 0:10] = 1
    obj = np.zeros((32, 32), dtype=np.uint8)
    obj[8:24, 16:28] = 1
    out, ov = IntentProcessor._apply_hand_overlap_veto(obj, region, max_overlap=0.5, min_pts=10)
    assert out is not None
    assert ov < 0.05
    assert int(out.sum()) == int(obj.sum())


def test_overlap_veto_subtracts_high_overlap_leftover():
    region = np.zeros((32, 32), dtype=np.uint8)
    region[0:32, 0:20] = 1
    blob = np.zeros((32, 32), dtype=np.uint8)
    blob[8:24, 4:28] = 1  # overlaps hand on the left, object on the right
    out, ov = IntentProcessor._apply_hand_overlap_veto(blob, region, max_overlap=0.5, min_pts=10)
    assert ov > 0.5
    assert out is not None
    assert int((out.astype(bool) & region.astype(bool)).sum()) == 0
    assert int(out.sum()) >= 10


def test_moving_frames_keeps_all_runs():
    n = 80
    det = np.ones(n, dtype=bool)
    tips = np.zeros((n, 3), dtype=np.float32)
    for t in range(10, 25):
        tips[t, 0] = (t - 10) * 0.02  # empty reach, large net
    for t in range(25, 50):
        tips[t] = tips[24]
    for t in range(50, 65):
        tips[t] = tips[49] + np.array([(t - 50) * 0.008, 0.0, 0.0], dtype=np.float32)
    for t in range(65, n):
        tips[t] = tips[64]
    moving = IntentProcessor._moving_hand_frames(det, tips, move_vel=0.005, move_run=4)
    assert moving[12:24].any(), "reach run should be kept"
    assert moving[52:64].any(), "carry run should be kept"
    win = IntentProcessor._find_transport_window(det, tips, move_vel=0.005, move_run=4, min_net=0.05)
    assert win is not None
    # The single-window heuristic prefers the larger net reach; v3 must still sample both.
    cands = IntentProcessor._sample_global_candidates(det, moving, stride=5, max_n=24)
    assert any(10 <= c <= 24 for c in cands)
    assert any(50 <= c <= 65 for c in cands)


def test_sample_global_candidates_fallback_detected():
    det = np.zeros(20, dtype=bool)
    det[2:18] = True
    moving = np.zeros(20, dtype=bool)
    cands = IntentProcessor._sample_global_candidates(det, moving, stride=5, max_n=24)
    assert cands
    assert all(det[c] for c in cands)


def test_score_probe_static_bonus_penalizes_always_still():
    n = 16
    tips = np.cumsum(np.ones((n, 3), dtype=np.float32) * 0.01, axis=0)
    still = np.tile(np.array([0.5, 0.0, 0.0], dtype=np.float32), (n, 1))
    s0, _, _ = IntentProcessor._score_probe_window(still, tips, w_static=0.0)
    s1, _, _ = IntentProcessor._score_probe_window(
        still, tips, static_vel_max=0.005, w_static=0.3,
    )
    assert s1 < s0, (s0, s1)


def test_score_probe_static_bonus_rewards_pickup():
    n = 16
    tips = np.zeros((n, 3), dtype=np.float32)
    obj = np.zeros((n, 3), dtype=np.float32)
    for t in range(n):
        if t < 6:
            tips[t] = np.array([0.01 * t, 0.0, 0.0], dtype=np.float32)
            obj[t] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # still on table
        else:
            delta = np.array([0.01 * t, 0.0, 0.0], dtype=np.float32)
            tips[t] = delta
            obj[t] = np.array([1.0, 0.0, 0.0], dtype=np.float32) + (delta - tips[5])
    s_off, _, _ = IntentProcessor._score_probe_window(obj, tips, w_static=0.0, static_vel_max=0.005)
    s_on, _, _ = IntentProcessor._score_probe_window(obj, tips, w_static=0.3, static_vel_max=0.005)
    assert s_on > s_off, (s_off, s_on)


def test_qa_fails_on_seed_hand_overlap():
    ip = IntentProcessor.__new__(IntentProcessor)
    ip.min_object_pts = 1
    ip.track_motion_corr_min = 0.0
    ip.track_static_vel_max = 1.0
    ip.track_min_disp = 0.0
    ip.track_hand_overlap_max = 0.5
    ip.track_backend = "hand_sam2"
    ip._seed_low_conf = False
    ip._seed_hand_overlap = 0.85
    n, h, w = 20, 8, 8
    masks = np.ones((n, h, w), dtype=np.uint8)
    tips_xyz = np.cumsum(np.ones((n, 3), dtype=np.float32) * 0.01, axis=0)
    centroids = tips_xyz + np.array([0.02, 0.0, 0.0], dtype=np.float32)
    pcd = {"centroids_robot": centroids, "valid": np.ones(n, dtype=bool)}
    phase = np.ones(n, dtype=int)
    contact = {"phase": phase}
    fingertips = np.repeat(tips_xyz[:, None, :], 3, axis=1)
    hands = {"right": {"detected": np.ones(n, dtype=bool), "fingertips": fingertips}}
    ip.target_hand = "right"
    qa = ip._track_quality_gate(masks, pcd, contact, hands, seed_idx=12)
    assert not qa["pass"], qa
    assert any("seed_hand_overlap" in r for r in qa["reasons"])


def test_grasp_disp_qa_passes_when_object_moves():
    ip = IntentProcessor.__new__(IntentProcessor)
    ip.min_object_pts = 1
    ip.track_motion_corr_min = 0.3
    ip.track_static_vel_max = 1.0  # don't fail on free-phase speed in this unit test
    ip.track_min_disp = 0.05
    ip.track_backend = "hand_sam2"
    ip._seed_low_conf = False
    n, h, w = 20, 8, 8
    masks = np.ones((n, h, w), dtype=np.uint8)
    tips_xyz = np.cumsum(np.ones((n, 3), dtype=np.float32) * 0.01, axis=0)
    centroids = tips_xyz + np.array([0.02, 0.0, 0.0], dtype=np.float32)
    pcd = {"centroids_robot": centroids, "valid": np.ones(n, dtype=bool)}
    phase = np.ones(n, dtype=int)  # all grasp/transport
    contact = {"phase": phase}
    fingertips = np.repeat(tips_xyz[:, None, :], 3, axis=1)
    hands = {"right": {"detected": np.ones(n, dtype=bool), "fingertips": fingertips}}
    ip.target_hand = "right"
    qa = ip._track_quality_gate(masks, pcd, contact, hands, seed_idx=12)
    assert qa["pass"], qa
    assert qa["grasp_disp"] >= 0.05


def test_pick_inhand_box_prefers_near_hand():
    bboxes = np.array([
        [0.0, 0.0, 20.0, 20.0],
        [90.0, 90.0, 110.0, 110.0],
        [200.0, 200.0, 220.0, 220.0],
    ], dtype=np.float32)
    scores = np.array([0.99, 0.40, 0.80], dtype=np.float32)
    hand = np.array([100.0, 100.0], dtype=np.float32)
    picked = IntentProcessor._pick_inhand_box(
        bboxes, scores, hand, hand_px=40.0, max_area=1e6,
    )
    assert picked is not None
    box, score = picked
    assert np.allclose(box, bboxes[1])
    assert abs(score - 0.40) < 1e-6


def test_pick_inhand_box_rejects_table_sized():
    bboxes = np.array([
        [0.0, 0.0, 640.0, 480.0],
        [300.0, 200.0, 360.0, 280.0],
    ], dtype=np.float32)
    scores = np.array([0.99, 0.35], dtype=np.float32)
    hand = np.array([330.0, 240.0], dtype=np.float32)
    max_area = 0.20 * 480.0 * 640.0
    picked = IntentProcessor._pick_inhand_box(
        bboxes, scores, hand, hand_px=200.0, max_area=max_area,
    )
    assert picked is not None
    assert np.allclose(picked[0], bboxes[1])


def test_pick_inhand_box_none_when_far():
    bboxes = np.array([[0.0, 0.0, 30.0, 30.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    hand = np.array([400.0, 400.0], dtype=np.float32)
    assert IntentProcessor._pick_inhand_box(
        bboxes, scores, hand, hand_px=50.0, max_area=1e6,
    ) is None


def test_pick_inhand_box_empty():
    assert IntentProcessor._pick_inhand_box(
        np.array([]), np.array([]), np.array([10.0, 10.0]), 50.0, 1e6,
    ) is None


def test_get_object_nouns_parses_objects_json():
    import json
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    payload = {"objects": ["iphone", "table", "box lid"], "prompt": "iphone"}
    assert IntentProcessor._parse_object_nouns(payload) == ["iphone", "table", "box lid"]
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "objects.json").write_text(json.dumps(payload))
        ip = IntentProcessor.__new__(IntentProcessor)
        ip.cfg_object_prompt = "fallback"
        nouns = ip._get_object_nouns(SimpleNamespace(data_path=root))
        assert nouns == ["iphone", "table", "box lid"]


def test_get_object_nouns_fallback_prompt():
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as d:
        ip = IntentProcessor.__new__(IntentProcessor)
        ip.cfg_object_prompt = "black stapler"
        nouns = ip._get_object_nouns(SimpleNamespace(data_path=Path(d)))
        assert nouns == ["black stapler"]


def test_pinch_uv_thumb_index_midpoint():
    k = np.zeros((21, 2), dtype=np.float32)
    k[4] = [10.0, 20.0]
    k[8] = [30.0, 40.0]
    uv = IntentProcessor._pinch_uv(k)
    assert uv is not None
    assert np.allclose(uv, [20.0, 30.0])


def _synthetic_hand_kpts(theta_deg: float) -> np.ndarray:
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    k = np.full((21, 3), np.nan, dtype=np.float32)
    base = {
        0: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        5: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        17: np.array([0.0, 1.0, 0.0], dtype=np.float32),
        4: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        8: np.array([1.0, 0.0, 1.0], dtype=np.float32),
        12: np.array([0.5, 0.2, 1.0], dtype=np.float32),
    }
    for idx, pt in base.items():
        k[idx] = Rz @ pt
    return k


def test_hand_rot_from_kpts_tracks_inplane_rotation():
    R0 = IntentProcessor._hand_rot_from_kpts(_synthetic_hand_kpts(0.0))
    R1 = IntentProcessor._hand_rot_from_kpts(_synthetic_hand_kpts(90.0))
    assert R0 is not None and R1 is not None
    rel = R1 @ R0.T
    want = np.array(
        [[0.0, -1.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    assert np.allclose(rel, want, atol=1e-5), rel


def test_integrate_intent_rotates_object_relative_target():
    ip = IntentProcessor.__new__(IntentProcessor)
    ip.grasp_offset_from = "hand"
    ip.gripper_max_width = 0.08
    ip.wp_grasp = 5.0
    ip.wp_free = 1.0
    ip.wr_grasp = 5.0
    ip.wr_free = 0.1
    ip.target_hand = "right"

    n = 3
    pcd = {
        "valid": np.ones(n, dtype=bool),
        "centroids_robot": np.zeros((n, 3), dtype=np.float32),
    }
    contact = {
        "phase": np.array([0, 1, 3], dtype=np.int64),
        "phase_names": np.array(["free", "grasp", "release"], dtype=object),
        "g_closed": np.array([False, True, False], dtype=bool),
        "grasp_keyframe": np.int64(1),
        "release_keyframe": np.int64(2),
    }
    grasp = {
        "valid": True,
        "G_center": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "G_rot": np.eye(3, dtype=np.float32),
        "G_width": np.float32(0.04),
        "grasp_frame": np.int64(1),
    }
    fingertips = np.zeros((n, 3, 3), dtype=np.float32)
    fingertips[0] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.5, 0.2, 1.0]], dtype=np.float32)
    fingertips[1] = fingertips[0]
    fingertips[2] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-0.2, 0.5, 1.0]], dtype=np.float32)
    hands = {
        "right": {
            "detected": np.ones(n, dtype=bool),
            "aperture": np.array([0.1, 0.03, 0.03], dtype=np.float32),
            "fingertips": fingertips,
            "kpts_rf": np.stack([
                _synthetic_hand_kpts(0.0),
                _synthetic_hand_kpts(0.0),
                _synthetic_hand_kpts(90.0),
            ], axis=0),
        }
    }
    out = ip._integrate_intent(pcd, contact, grasp, hands)
    assert np.allclose(out["R_target"][1], np.eye(3), atol=1e-6)
    want_R2 = np.array(
        [[0.0, -1.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    assert np.allclose(out["R_target"][2], want_R2, atol=1e-5), out["R_target"][2]
    assert np.allclose(out["p_target"][1], np.array([0.5, 0.0, 1.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(out["p_target"][2], np.array([0.0, 0.5, 1.0], dtype=np.float32), atol=1e-5), out["p_target"][2]


if __name__ == "__main__":
    tests = [
        test_select_seed_near_grasp,
        test_select_seed_fallback_min_aperture,
        test_select_seed_none_when_undetected,
        test_inhand_window,
        test_vel_corr_attached,
        test_vel_corr_static_vs_moving,
        test_clip_uv,
        test_squeeze_video_mask,
        test_find_transport_window_prefers_net_disp,
        test_find_transport_window_none_when_still,
        test_sample_candidate_frames,
        test_score_probe_prefers_comotion,
        test_grasp_disp_qa_fails_on_static_object,
        test_grasp_disp_qa_passes_when_object_moves,
        test_hand_region_covers_palm_not_pinch_gap,
        test_overlap_veto_rejects_hand_mask,
        test_overlap_veto_keeps_adjacent_object,
        test_overlap_veto_subtracts_high_overlap_leftover,
        test_moving_frames_keeps_all_runs,
        test_sample_global_candidates_fallback_detected,
        test_score_probe_static_bonus_penalizes_always_still,
        test_score_probe_static_bonus_rewards_pickup,
        test_qa_fails_on_seed_hand_overlap,
        test_pick_inhand_box_prefers_near_hand,
        test_pick_inhand_box_rejects_table_sized,
        test_pick_inhand_box_none_when_far,
        test_pick_inhand_box_empty,
        test_get_object_nouns_parses_objects_json,
        test_get_object_nouns_fallback_prompt,
        test_pinch_uv_thumb_index_midpoint,
        test_hand_rot_from_kpts_tracks_inplane_rotation,
        test_integrate_intent_rotates_object_relative_target,
    ]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")
