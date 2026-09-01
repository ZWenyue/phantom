"""Unit tests for Stage C depth-aware retarget overlay.

Usage (from phantom/):
    python b/tests/test_retarget_depth_overlay.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "phantom")))

from phantom.processors.retarget_inpaint_processor import (
    crop_square_if_needed,
    prepare_scene_depth_for_overlay,
)
from phantom.processors.robotinpaint_processor import RobotInpaintProcessor


def test_crop_square_center():
    img = np.zeros((4, 8), np.float32)
    img[:, 2:6] = 1.0
    cropped = crop_square_if_needed(img, square=True)
    assert cropped.shape == (4, 4)
    assert np.allclose(cropped, 1.0)
    assert crop_square_if_needed(img, square=False).shape == (4, 8)


def test_invalid_depth_does_not_occlude():
    d = np.array([[0.0, np.nan], [0.4, 0.8]], np.float32)
    out = prepare_scene_depth_for_overlay(d, (2, 2))
    assert np.isinf(out[0, 0])
    assert np.isinf(out[0, 1])
    assert np.isclose(out[1, 0], 0.4)


def test_resize_to_mask_hw():
    d = np.array([[0.5, 1.5]], np.float32)
    out = prepare_scene_depth_for_overlay(d, (2, 2))
    assert out.shape == (2, 2)


def test_nearer_scene_hides_robot():
    robot = np.ones((2, 2), np.uint8)
    grip = np.zeros((2, 2), np.uint8)
    hand = np.zeros((2, 2), np.uint8)
    real = np.full((2, 2), 0.3, np.float32)   # scene in front
    sim = np.full((2, 2), 0.8, np.float32)
    mask = RobotInpaintProcessor._create_overlay_mask(None, robot, grip, real, sim, hand)
    assert not np.any(mask)


def test_farther_scene_keeps_robot():
    robot = np.ones((2, 2), np.uint8)
    grip = np.zeros((2, 2), np.uint8)
    hand = np.zeros((2, 2), np.uint8)
    real = np.full((2, 2), 1.2, np.float32)   # scene behind the arm
    sim = np.full((2, 2), 0.8, np.float32)
    mask = RobotInpaintProcessor._create_overlay_mask(None, robot, grip, real, sim, hand)
    assert np.all(mask)


if __name__ == "__main__":
    test_crop_square_center()
    test_invalid_depth_does_not_occlude()
    test_resize_to_mask_hw()
    test_nearer_scene_hides_robot()
    test_farther_scene_keeps_robot()
    print("ok")
