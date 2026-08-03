#!/usr/bin/env python3
"""Probe short TCP with kp vector applied AFTER TwinBimanualRobot init."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submodules" / "phantom-robosuite"))
os.environ.setdefault("MUJOCO_GL", "egl")

import robosuite.environments.manipulation.phantom_bimanual as pb
from phantom.twin_bimanual_robot import TwinBimanualRobot, MujocoCameraParams, convert_real_camera_ori_to_mujoco
from phantom.utils.image_utils import get_intrinsics_from_json

DEMO = Path("/mnt/r/DATA/EgoDex/test_phantom_processed/egodex_make_sandwich_test/0_useful")
FRAMES = [0, 30, 60, 100, 200, 300, 400]
THRESH = 0.05

BASE_CFG = dict(
    r_pos=(-0.18, -0.40, 1.70), r_rot=(0.0, 0.0, np.pi / 2),
    l_pos=(-0.15, 0.42, 1.55), l_rot=(0.0, 0.0, -np.pi / 2),
)


def install_bases(cfg):
    from robosuite.models.arenas import EmptyArena
    from robosuite.models.tasks import ManipulationTask

    def _load_model(self):
        super(pb.PhantomBimanual, self)._load_model()
        for count, robot in enumerate(self.robots):
            key = "r" if count == 0 else "l"
            pos, rot = cfg[f"{key}_pos"], cfg[f"{key}_rot"]
            xpos = np.array([pos[0], pos[1], pos[2] + robot.robot_model.bottom_offset[2]])
            robot.robot_model.set_base_xpos(xpos)
            robot.robot_model.set_base_ori(np.array(rot))
        mujoco_arena = EmptyArena()
        mujoco_arena.set_origin([0, 0, 0])
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[r.robot_model for r in self.robots],
        )
        if self.camera_pos is not None:
            mujoco_arena.set_camera(
                camera_name="zed", pos=self.camera_pos, quat=self.camera_quat_wxyz,
                camera_attribs={
                    "sensorsize": np.array2string(self.camera_sensorsize)[1:-1],
                    "resolution": f"{self.camera_widths[0]} {self.camera_heights[0]}",
                    "principalpixel": np.array2string(self.camera_principalpixel)[1:-1],
                    "focalpixel": np.array2string(self.camera_focalpixel)[1:-1],
                },
            )

    pb.PhantomBimanual._load_model = _load_model


def build_camera():
    _, intr = get_intrinsics_from_json(str(ROOT / "phantom/camera/camera_intrinsics_egodex.json"))
    with open(ROOT / "phantom/camera/camera_extrinsics_ego_bimanual_shoulders.json") as f:
        ext = json.load(f)[0]
    img_w, img_h = 1920, 1080
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    return MujocoCameraParams(
        name="zed", pos=np.array(ext["camera_base_pos"]),
        ori_wxyz=convert_real_camera_ori_to_mujoco(np.array(ext["camera_base_ori"])),
        fov=intr["v_fov"], resolution=(img_h, img_w),
        sensorsize=np.array([img_w / fy / 1000, img_h / fx / 1000]),
        principalpixel=np.array([img_w / 2 - cx, cy - img_h / 2]),
        focalpixel=np.array([fx, fy]),
    )


def apply_kp(robot, kp):
    kp = np.array(kp, dtype=float)
    for r in robot.env.env.robots:
        ctrl = r.controller
        ctrl.kp = kp
        if hasattr(ctrl, "kd"):
            # critically damped-ish: kd = 2*sqrt(kp) per dim
            ctrl.kd = 2.0 * np.sqrt(np.maximum(kp, 1e-6))
        print("  controller kp=", ctrl.kp)


def main():
    right = np.load(DEMO / "smoothing_processor/smoothed_actions_right_r1pro.npz")
    left = np.load(DEMO / "smoothing_processor/smoothed_actions_left_r1pro.npz")
    cam = build_camera()
    install_bases(BASE_CFG)

    for name, kp in [
        ("scalar200", 200),
        ("pos300_ori5", [300, 300, 300, 5, 5, 5]),
        ("pos300_ori1", [300, 300, 300, 1, 1, 1]),
        ("pos400_ori0", [400, 400, 400, 0, 0, 0]),
    ]:
        print(f"\n=== {name} ===")
        robot = TwinBimanualRobot(
            ["R1ProRightArm", "R1ProLeftArm"], "R1Pro", "r1pro", cam,
            camera_height=1080, camera_width=1920, render=False,
            n_steps_short=80, n_steps_long=150, epic=True, joint_controller=False,
        )
        apply_kp(robot, kp if np.iterable(kp) else [kp] * 6)
        errs_l, errs_r = [], []
        for i, idx in enumerate(FRAMES):
            state = {
                "pos": [right["ee_pts"][idx], left["ee_pts"][idx]],
                "ori_xyzw": [
                    Rotation.from_matrix(right["ee_oris"][idx]).as_quat(),
                    Rotation.from_matrix(left["ee_oris"][idx]).as_quat(),
                ],
                "gripper_pos": [float(right["ee_widths"][idx]), float(left["ee_widths"][idx])],
            }
            out = robot.move_to_target_state(state, init=(i == 0))
            errs_l.append(out["left_pos_err"])
            errs_r.append(out["right_pos_err"])
            ok = out["left_pos_err"] <= THRESH and out["right_pos_err"] <= THRESH
            print(f"  frame {idx:3d}: L={out['left_pos_err']:.3f} R={out['right_pos_err']:.3f} {'OK' if ok else 'FAIL'}")
        n_ok = sum(1 for l, r in zip(errs_l, errs_r) if l <= THRESH and r <= THRESH)
        print(f"  pass {n_ok}/7  mean L={np.mean(errs_l):.3f} R={np.mean(errs_r):.3f}")
        robot.close()


if __name__ == "__main__":
    main()
