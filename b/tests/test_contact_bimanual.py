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
from phantom.processors.stageb_processor import ROBOT_IDX


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


if __name__ == "__main__":
    test_pick_place_primary_prefers_valid_grasp()
    test_shared_t_place_preserves_relative_offset()
    test_paths_for_hand()
    test_robot_idx_mapping()
    test_shared_world_not_per_base()
    test_workspace_frame_torso()
    print("ok")
