"""Unit tests for scheme-A shared T_place + per-hand path binding.

Usage (from phantom/):
    python b/tests/test_contact_bimanual.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "phantom")))

from phantom.processors.intent_processor import IntentProcessor
from phantom.processors.paths import Paths
from phantom.processors.retarget_inpaint_processor import RetargetInpaintProcessor
from phantom.processors.segmentation_processor import ArmSegmentationProcessor
from phantom.processors.stageb_processor import ROBOT_IDX, StageBProcessor


def test_pick_place_primary_prefers_valid_grasp():
    per = {
        "left": {
            "grasp_result": {"valid": True},
            "contact_result": {"grasp_keyframe": 3},
        },
        "right": {
            "grasp_result": {"valid": False},
            "contact_result": {"grasp_keyframe": -1},
        },
    }
    assert IntentProcessor._pick_place_primary(per) == "left"
    per["right"]["grasp_result"]["valid"] = True
    assert IntentProcessor._pick_place_primary(per) == "right"


def test_shared_t_place_preserves_relative_offset():
    """Left and right points keep their offset after one T_place."""
    p_right = np.array([1.0, 2.0, 3.0])
    p_left = p_right + np.array([0.0, 0.25, 0.0])
    look = np.array([0.0, 0.0, 1.0])
    T = IntentProcessor._place_T_w2r(p_right, look, np.array([0.5, 0.0, 0.05]))
    r = IntentProcessor._apply_T_xyz(p_right, T)
    l = IntentProcessor._apply_T_xyz(p_left, T)
    assert np.allclose(r, [0.5, 0.0, 0.05], atol=1e-8)
    rel_before = p_left - p_right
    R = T[:3, :3]
    rel_after = l - r
    assert np.allclose(rel_after, R @ rel_before, atol=1e-8)


def test_paths_for_hand(tmp_path=None):
    root = Path(tempfile.mkdtemp())
    (root / "dummy").mkdir()
    # Paths requires data_path exists
    p = Paths(data_path=root)
    left = p.for_hand("left")
    assert str(left.intent).endswith("intent_left.npz")
    assert str(left.joint_trajectory).endswith("q_trajectory_left.npz")
    assert str(p.intent).endswith("intent.npz")


def test_robot_idx_mapping():
    assert ROBOT_IDX["right"] == 0
    assert ROBOT_IDX["left"] == 1


def test_no_grasp_moving_hand_is_not_parked():
    points = np.zeros((20, 3), dtype=np.float32)
    points[:, 0] = np.linspace(0.0, 0.10, len(points))
    intent = {
        "p_target": points,
        "p_valid": np.ones(len(points), dtype=bool),
        "p_source": np.full(len(points), "hand", dtype=object),
    }
    span, frames = StageBProcessor._hand_motion_span(intent)
    proc = StageBProcessor.__new__(StageBProcessor)
    proc.park_idle_arms = True
    proc.idle_hand_span_thresh = 0.02
    proc.idle_hand_min_frames = 5
    assert span > 0.02
    assert frames == len(points)
    assert not proc._should_park_arm(False, span, frames)


def test_no_grasp_static_hand_is_parked():
    points = np.zeros((20, 3), dtype=np.float32)
    points[:, 0] = np.linspace(0.0, 0.005, len(points))
    intent = {
        "p_target": points,
        "p_valid": np.ones(len(points), dtype=bool),
        "p_source": np.full(len(points), "hand", dtype=object),
    }
    span, frames = StageBProcessor._hand_motion_span(intent)
    proc = StageBProcessor.__new__(StageBProcessor)
    proc.park_idle_arms = True
    proc.idle_hand_span_thresh = 0.02
    proc.idle_hand_min_frames = 5
    assert proc._should_park_arm(False, span, frames)
    assert not proc._should_park_arm(True, span, frames)


def test_shared_world_not_per_base():
    """Mapping both targets through the SAME origin keeps Δ; through different bases does not."""
    p_r = np.array([0.50, 0.00, 0.05])
    p_l = np.array([0.50, 0.30, 0.05])
    torso = np.array([0.0, 0.0, 1.5])
    base_r = np.array([-0.18, -0.30, 1.40])
    world_r = p_r + torso
    world_l_shared = p_l + torso
    world_l_wrong = p_l + base_r
    assert np.allclose(world_l_shared - world_r, p_l - p_r)
    assert not np.allclose(world_l_shared - world_r, world_l_wrong - (p_r + torso))


def test_workspace_frame_torso():
    from phantom.processors.stageb_processor import WorkspaceFrame
    wf = WorkspaceFrame(np.eye(3), np.array([0.0, 0.0, 1.5]))
    p = np.array([[0.5, 0.0, 0.05], [0.5, 0.3, 0.05]])
    w = wf.world_pos(p)
    assert np.allclose(w[0], [0.5, 0.0, 1.55])
    assert np.allclose(w[1] - w[0], [0.0, 0.3, 0.0])


def test_person_mask_overlaps_hand_bbox():
    mask = np.zeros((100, 200), dtype=bool)
    mask[40:80, 60:120] = True
    assert ArmSegmentationProcessor._mask_overlaps_bbox(
        mask, np.array([70, 50, 100, 70]), image_hw=(100, 200)
    )
    assert not ArmSegmentationProcessor._mask_overlaps_bbox(
        mask, np.array([150, 10, 190, 30]), image_hw=(100, 200)
    )


def test_distal_robot_mask_filters_proximal_geoms():
    class Model:
        ngeom = 4
        geom_bodyid = np.array([0, 1, 2, 3])

        @staticmethod
        def body_id2name(body_id):
            return [
                "world",
                "robot0_link3",
                "robot0_link6",
                "gripper0_right_inner_finger",
            ][body_id]

    class Sim:
        model = Model()

    proc = RetargetInpaintProcessor.__new__(RetargetInpaintProcessor)
    proc.distal_only = True
    proc.distal_body_tokens = ("link7", "gripper", "finger")
    proc._distal_geom_ids_cache = {}
    proc._distal_filter_warned = False
    instance = np.ones((2, 2), dtype=np.int32)
    element = np.array([[0, 1], [2, 3]], dtype=np.int32)
    expected = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    assert np.array_equal(proc._robot_render_mask(instance, element, Sim()), expected)


if __name__ == "__main__":
    test_pick_place_primary_prefers_valid_grasp()
    test_shared_t_place_preserves_relative_offset()
    test_paths_for_hand()
    test_robot_idx_mapping()
    test_no_grasp_moving_hand_is_not_parked()
    test_no_grasp_static_hand_is_parked()
    test_shared_world_not_per_base()
    test_workspace_frame_torso()
    test_person_mask_overlaps_hand_bbox()
    test_distal_robot_mask_filters_proximal_geoms()
    print("ok")
