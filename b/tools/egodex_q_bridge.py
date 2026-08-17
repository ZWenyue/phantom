#!/usr/bin/env python3
"""Bridge legacy EgoDex robot_inpaint joint labels -> Stage B ``q_trajectory.npz``.

EgoDex has no RGB-D depth, so intent/stageb cannot run natively. For Stage C
visual testing we reuse ``joint_pos_*`` from the existing ``training_data_*.npz``
(produced by robot_inpaint) and compute ``FK(q)`` labels with the same single-arm
Phantom env Stage B/C use.

Example:
    python b/tools/egodex_q_bridge.py --demo 0 --hand left
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "phantom"))

from phantom.panda_frantik_ik import PANDA_JOINT_LIMITS
from phantom.processors.stageb_processor import MujocoPandaArm, _make_single_arm_env


def main() -> None:
    ap = argparse.ArgumentParser(description="EgoDex legacy joints -> q_trajectory.npz")
    ap.add_argument("--processed-root", default="/home/a26160/DATA/test_phantom_processed")
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--hand", choices=("left", "right"), default="left")
    ap.add_argument(
        "--training-tag", default="shoulders",
        help="Suffix on training_data_<tag>.npz (default shoulders for Panda)",
    )
    args = ap.parse_args()

    demo_dir = os.path.join(
        args.processed_root, "egodex_basic_pick_place", str(args.demo),
    )
    td_path = os.path.join(
        demo_dir, "inpaint_processor", f"training_data_{args.training_tag}.npz",
    )
    if not os.path.isfile(td_path):
        raise FileNotFoundError(f"Missing legacy training data: {td_path}")

    td = np.load(td_path, allow_pickle=True)
    key = f"joint_pos_{args.hand}"
    q = np.asarray(td[key], dtype=float)
    valid = np.asarray(td["valid"], dtype=bool)
    if not valid.all():
        q = q[valid]
        grip_w = np.asarray(td[f"gripper_width_{args.hand}"], dtype=float)[valid]
    else:
        grip_w = np.asarray(td[f"gripper_width_{args.hand}"], dtype=float)

    env = _make_single_arm_env()
    kin = MujocoPandaArm(env, 0)
    T = len(q)
    ee_pos_world = np.empty((T, 3))
    ee_R_world = np.empty((T, 3, 3))
    for t in range(T):
        p, R = kin.fk(q[t])
        ee_pos_world[t] = p
        ee_R_world[t] = R
    ee_pos_robot = kin.robot_pos(ee_pos_world)
    ee_R_robot = np.einsum("ij,tjk->tik", kin.base_R.T, ee_R_world)

    out_dir = os.path.join(demo_dir, "stageb_processor")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "q_trajectory.npz")
    np.savez(
        out_path,
        q=q.astype(np.float32),
        pos_err=np.zeros(T, dtype=np.float32),
        ori_err=np.zeros(T, dtype=np.float32),
        ee_pos_world=ee_pos_world.astype(np.float32),
        ee_R_world=ee_R_world.astype(np.float32),
        ee_pos_robot=ee_pos_robot.astype(np.float32),
        ee_R_robot=ee_R_robot.astype(np.float32),
        gripper_width=grip_w.astype(np.float32),
        phase=np.zeros(T, dtype=np.int32),
        valid=np.ones(T, dtype=bool),
        arm_side=args.hand,
        robot_idx=np.int64(0),
        joint_limits=PANDA_JOINT_LIMITS,
        cost_initial=np.float32(0.0),
        cost_final=np.float32(0.0),
        jerk_rms=np.float32(0.0),
        nfev=np.int64(0),
        success=True,
        grip_rot_offset_deg=np.float32(90.0),
        source="egodex_legacy_training_data",
    )
    print(f"Wrote {out_path}  T={T} hand={args.hand}")


if __name__ == "__main__":
    main()
