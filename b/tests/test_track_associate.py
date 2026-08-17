"""Association helpers for YOLO + previous-mask object tracking.

Usage (from phantom/):
    python b/tests/test_track_associate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "phantom")))

from phantom.processors.intent_processor import IntentProcessor


def _mask_at(h, w, box):
    m = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 1
    return m


def test_iou_identity():
    m = _mask_at(40, 40, (10, 10, 20, 20))
    iou = IntentProcessor._box_mask_iou(m, np.array([10, 10, 20, 20], dtype=np.float32))
    assert abs(iou - 1.0) < 1e-6, iou
    print(f"[iou-id] {iou:.3f}")


def test_assoc_prefers_overlap():
    prev = _mask_at(80, 80, (5, 5, 25, 25))
    left = np.array([6, 6, 24, 24], np.float32)
    right = np.array([50, 50, 70, 70], np.float32)
    bboxes = np.stack([right, left])
    scores = np.array([0.4, 0.5])  # left is also higher YOLO and overlaps
    box, how, iou = IntentProcessor._associate_yolo_box(prev, bboxes, scores, 0.15)
    assert how == "assoc", how
    assert np.allclose(box, left), box
    print(f"[assoc] how={how} iou={iou:.3f}")


def test_reid_when_object_jumped():
    prev = _mask_at(80, 80, (5, 5, 25, 25))  # table remnant
    hand = np.array([50, 20, 70, 40], np.float32)  # lifted stapler, no overlap
    bboxes = hand.reshape(1, 4)
    scores = np.array([0.4])
    box, how, iou = IntentProcessor._associate_yolo_box(prev, bboxes, scores, 0.15)
    assert how == "reid", how
    assert iou < 0.15, iou
    assert np.allclose(box, hand), box
    print(f"[reid] how={how} iou={iou:.3f}")


def test_reid_top_score_jumps_off_remnant():
    """Without a hand prior, overlapping remnant beats a higher-score distractor."""
    prev = _mask_at(80, 80, (5, 5, 25, 25))
    remnant = np.array([6, 6, 24, 24], np.float32)
    lifted = np.array([50, 50, 70, 70], np.float32)
    bboxes = np.stack([remnant, lifted])
    scores = np.array([0.3, 0.9])
    box, how, iou = IntentProcessor._associate_yolo_box(prev, bboxes, scores, 0.15)
    assert how == "assoc", how
    assert np.allclose(box, remnant), box
    print(f"[assoc-remnant] how={how} iou={iou:.3f}")


def test_assoc_among_overlapping():
    """When the top-score box still overlaps, keep max IoU (same instance)."""
    prev = _mask_at(80, 80, (10, 10, 30, 30))
    tight = np.array([12, 12, 28, 28], np.float32)
    loose = np.array([0, 0, 50, 50], np.float32)
    bboxes = np.stack([loose, tight])
    scores = np.array([0.9, 0.4])  # loose is higher YOLO but worse IoU
    box, how, iou = IntentProcessor._associate_yolo_box(prev, bboxes, scores, 0.15)
    assert how == "assoc", how
    assert np.allclose(box, tight), box
    print(f"[assoc-overlap] how={how} iou={iou:.3f}")


def test_hand_prior_steals_from_remnant():
    """In-hand box near the fingertip wins even if the remnant scores higher."""
    prev = _mask_at(80, 80, (5, 5, 25, 25))
    remnant = np.array([6, 6, 24, 24], np.float32)
    inhand = np.array([50, 20, 70, 40], np.float32)
    bboxes = np.stack([remnant, inhand])
    scores = np.array([0.9, 0.2])  # remnant is the YOLO winner
    hand_uv = np.array([60.0, 30.0], np.float32)  # in-hand centre
    box, how, iou = IntentProcessor._associate_yolo_box(
        prev, bboxes, scores, 0.15, hand_uv=hand_uv, hand_px=20.0,
    )
    assert how == "hand", how
    assert np.allclose(box, inhand), box
    print(f"[hand] how={how} iou={iou:.3f}")


def test_hand_prior_keeps_table_when_hand_is_there():
    """Approach: hand is on the table object, keep the overlapping box."""
    prev = _mask_at(80, 80, (5, 5, 25, 25))
    table = np.array([6, 6, 24, 24], np.float32)
    clutter = np.array([50, 50, 70, 70], np.float32)
    bboxes = np.stack([table, clutter])
    scores = np.array([0.4, 0.9])
    hand_uv = np.array([15.0, 15.0], np.float32)
    box, how, iou = IntentProcessor._associate_yolo_box(
        prev, bboxes, scores, 0.15, hand_uv=hand_uv, hand_px=20.0,
    )
    assert how == "assoc", how
    assert np.allclose(box, table), box
    print(f"[hand-approach] how={how} iou={iou:.3f}")


def test_size_skips_huge_hand_box():
    """Hand prior ignores a giant box (Connect-4 / arm) and keeps the remnant."""
    prev = _mask_at(80, 80, (5, 5, 25, 25))  # area 400
    remnant = np.array([6, 6, 24, 24], np.float32)
    giant = np.array([0, 0, 80, 80], np.float32)  # area 6400 = 16x
    bboxes = np.stack([remnant, giant])
    scores = np.array([0.2, 0.9])
    hand_uv = np.array([40.0, 40.0], np.float32)  # giant centre
    box, how, _ = IntentProcessor._associate_yolo_box(
        prev, bboxes, scores, 0.15,
        hand_uv=hand_uv, hand_px=30.0, max_area_ratio=3.5,
    )
    assert how == "assoc", how
    assert np.allclose(box, remnant), box
    print("[size] skipped giant hand box")


def test_far_reid_holds():
    """A far non-overlapping box is a table leftover — hold, do not snap back."""
    prev = _mask_at(80, 80, (5, 5, 25, 25))
    far = np.array([50, 50, 70, 70], np.float32)
    box, how, _ = IntentProcessor._associate_yolo_box(
        prev, far.reshape(1, 4), np.array([0.4]), 0.15, jump_px=40.0,
    )
    assert box is None and how == "hold"
    print("[hold-far] skipped distant reid")


def test_huge_reid_holds():
    """A nearby but giant box must not explode the mask; hold instead."""
    prev = _mask_at(80, 80, (20, 20, 40, 40))  # area 400
    giant = np.array([10, 10, 70, 70], np.float32)  # area 3600 = 9x, little overlap
    box, how, _ = IntentProcessor._associate_yolo_box(
        prev, giant.reshape(1, 4), np.array([0.8]), 0.15, max_area_ratio=3.5,
    )
    assert box is None and how == "hold", (box, how)
    print("[hold-huge] skipped giant reid")


def test_hold_when_empty():
    prev = _mask_at(20, 20, (0, 0, 5, 5))
    box, how, iou = IntentProcessor._associate_yolo_box(
        prev, np.array([]), np.array([]), 0.15
    )
    assert box is None and how == "hold"
    print("[hold] empty yolo")


def test_select_prefers_dark_core():
    """Grey high-score hammer loses to a lower-score black stapler."""
    frame = np.full((80, 80, 3), 200, np.uint8)
    frame[10:30, 10:40] = 110  # grey hammer
    frame[45:70, 15:70] = 20   # black stapler
    hammer = np.array([10, 10, 40, 30], np.float32)
    stapler = np.array([15, 45, 70, 70], np.float32)
    bboxes = np.stack([hammer, stapler])
    scores = np.array([0.59, 0.27])
    box, score, how = IntentProcessor._select_yolo_box(
        frame, bboxes, scores, prefer_dark=True, dark_luma=60.0, score_keep=0.2,
    )
    assert how == "dark", how
    assert np.allclose(box, stapler), box
    assert abs(score - 0.27) < 1e-6
    print(f"[seed-dark] box={box.astype(int)} score={score:.2f}")


def test_select_score_when_not_dark():
    frame = np.full((40, 40, 3), 200, np.uint8)
    a = np.array([0, 0, 10, 10], np.float32)
    b = np.array([20, 20, 35, 35], np.float32)
    box, score, how = IntentProcessor._select_yolo_box(
        frame, np.stack([a, b]), np.array([0.9, 0.2]), prefer_dark=False,
    )
    assert how == "score" and np.allclose(box, a)
    print("[seed-score] top YOLO")


def test_tighten_dark_box():
    frame = np.full((60, 80, 3), 180, np.uint8)
    frame[20:35, 10:55] = 15
    loose = np.array([0, 0, 80, 60], np.float32)
    tight = IntentProcessor._tighten_dark_box(frame, loose, luma_thr=60.0, pad=2)
    # tightened around [10,20,55,35] + pad 2
    assert tight[0] <= 10 and tight[2] >= 55
    assert tight[1] <= 20 and tight[3] >= 35
    assert (tight[2] - tight[0]) < 80 and (tight[3] - tight[1]) < 60
    print(f"[tighten] {tight.astype(int)}")


def test_compose_T_cam2robot_seq_static_world():
    """A world-static point stays still in robot frame after composing T(t)."""
    T_calib = np.eye(4)
    T_calib[:3, 3] = [0.3, -0.1, 0.2]
    n = 8
    T_c2w = np.tile(np.eye(4), (n, 1, 1))
    for t in range(n):
        ang = 0.2 * t
        c, s = np.cos(ang), np.sin(ang)
        T_c2w[t, :3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        T_c2w[t, :3, 3] = [0.1 * t, 0.0, 0.5]
    T_c2r = IntentProcessor._compose_T_cam2robot_seq(T_c2w, T_calib, ref_idx=0)
    p_w = np.array([1.0, 0.2, 0.3, 1.0])
    robots = []
    for t in range(n):
        p_cam = np.linalg.inv(T_c2w[t]) @ p_w
        p_r = T_c2r[t] @ p_cam
        robots.append(p_r[:3])
    robots = np.stack(robots)
    assert np.allclose(robots, robots[0], atol=1e-9), robots
    p_r0 = (T_calib @ np.linalg.inv(T_c2w[0]) @ p_w)[:3]
    assert np.allclose(robots[0], p_r0, atol=1e-9)
    print("[T_camera] static world → constant robot", np.round(robots[0], 3))


def test_place_T_w2r_maps_ref_and_up():
    p_ref = np.array([1.0, 2.0, 3.0])
    look = np.array([0.0, 0.0, 1.0])
    p_star = np.array([0.50, 0.00, 0.05])
    T = IntentProcessor._place_T_w2r(p_ref, look, p_star)
    ph = np.append(p_ref, 1.0)
    assert np.allclose((T @ ph)[:3], p_star, atol=1e-9)
    up = np.array([0.0, 1.0, 0.0, 0.0])  # world +Y, homogeneous direction
    assert np.allclose((T @ up)[:3], [0.0, 0.0, 1.0], atol=1e-9)  # robot +Z
    fwd = np.array([0.0, 0.0, 1.0, 0.0])
    assert np.allclose((T @ fwd)[:3], [1.0, 0.0, 0.0], atol=1e-9)  # robot +X
    print("[T_place] ref->star", np.round(p_star, 3), "up->+Z look->+X")


if __name__ == "__main__":
    test_iou_identity()
    test_assoc_prefers_overlap()
    test_reid_when_object_jumped()
    test_reid_top_score_jumps_off_remnant()
    test_assoc_among_overlapping()
    test_hand_prior_steals_from_remnant()
    test_hand_prior_keeps_table_when_hand_is_there()
    test_size_skips_huge_hand_box()
    test_far_reid_holds()
    test_huge_reid_holds()
    test_hold_when_empty()
    test_select_prefers_dark_core()
    test_select_score_when_not_dark()
    test_tighten_dark_box()
    test_compose_T_cam2robot_seq_static_world()
    test_place_T_w2r_maps_ref_and_up()
    print("ok")
