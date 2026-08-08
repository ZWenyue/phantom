"""
Franka (Panda) analytical IK via frantik, validated/refined in MuJoCo.

frantik uses Franka's nominal kinematics; robosuite Panda + Robotiq85 differs slightly.
We therefore:
  1. Transform world-frame targets into each arm's base frame.
  2. Seed with frantik.ik over a q7 grid (case-consistent selection vs q_seed).
  3. Refine joint angles with damped least-squares on the MuJoCo grip_site FK.
  4. Optionally snap targets to a precomputed reachable position cloud (see compute_reachability.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import frantik
import numpy as np
from scipy.spatial.transform import Rotation

from phantom.twin_bimanual_robot import BASE_T_1


@dataclass
class ReachabilityIndex:
    """Nearest-neighbour index over precomputed EEF position clouds."""

    left_positions: np.ndarray
    right_positions: np.ndarray
    tol: float = 0.02

    @classmethod
    def load(cls, npz_path: str, tol: float = 0.02) -> "ReachabilityIndex":
        data = np.load(npz_path)
        return cls(
            left_positions=data["left_positions"],
            right_positions=data["right_positions"],
            tol=tol,
        )

    def snap(self, arm: str, target: np.ndarray) -> tuple[np.ndarray, float, bool]:
        positions = self.left_positions if arm == "left" else self.right_positions
        dists = np.linalg.norm(positions - target, axis=1)
        idx = int(dists.argmin())
        return positions[idx].copy(), float(dists[idx]), bool(dists[idx] <= self.tol)


def pose_to_matrix(pos: np.ndarray, ori) -> np.ndarray:
    T = np.eye(4)
    ori_arr = np.asarray(ori)
    if ori_arr.shape == (3, 3):
        T[:3, :3] = ori_arr
    else:
        T[:3, :3] = Rotation.from_quat(ori_arr).as_matrix()
    T[:3, 3] = pos
    return T


def epic_robot_pose_to_world(pos: np.ndarray, ori) -> np.ndarray:
    T_rf = pose_to_matrix(pos, ori)
    return BASE_T_1 @ T_rf


def base_T_world(sim, root_body: str) -> np.ndarray:
    bid = sim.model.body_name2id(root_body)
    T = np.eye(4)
    T[:3, :3] = sim.data.body_xmat[bid].reshape(3, 3)
    T[:3, 3] = sim.data.body_xpos[bid]
    return T


PANDA_JOINT_LIMITS = np.array([
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973],
])


class PandaFrantikIKSolver:
    """frantik + MuJoCo refinement IK for Phantom Panda bimanual overlay."""

    Q7_SAMPLES = 40
    FRANTIK_ITERS = 1
    REFINE_ITERS = 25
    REFINE_LAMBDA = 0.05
    REFINE_TOL = 0.002

    def __init__(
        self,
        env,
        reachability: Optional[ReachabilityIndex] = None,
        snap_unreachable: bool = True,
    ):
        self.env = env
        self.sim = env.env.sim
        self.robots = env.env.robots
        self.reachability = reachability
        self.snap_unreachable = snap_unreachable
        self._q_seeds = {
            0: np.array(self.robots[0].robot_model.init_qpos, dtype=float),
            1: np.array(self.robots[1].robot_model.init_qpos, dtype=float),
        }

    def reset_seeds(self) -> None:
        for idx in (0, 1):
            self._q_seeds[idx] = np.array(self.robots[idx].robot_model.init_qpos, dtype=float)

    def solve_bimanual(
        self,
        right_pos: np.ndarray,
        right_ori_xyzw: np.ndarray,
        left_pos: np.ndarray,
        left_ori_xyzw: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
        q_r, err_r = self.solve_arm(0, right_pos, right_ori_xyzw)
        q_l, err_l = self.solve_arm(1, left_pos, left_ori_xyzw)
        return q_r, q_l, err_r, err_l

    def solve_arm(
        self,
        robot_idx: int,
        pos: np.ndarray,
        ori_xyzw: np.ndarray,
    ) -> tuple[Optional[np.ndarray], float]:
        arm = "right" if robot_idx == 0 else "left"
        T_world = epic_robot_pose_to_world(pos, ori_xyzw)
        target_world = T_world[:3, 3].copy()

        if self.reachability is not None and self.snap_unreachable:
            snapped, dist, reachable = self.reachability.snap(arm, target_world)
            if not reachable:
                T_world = T_world.copy()
                T_world[:3, 3] = snapped
                target_world = snapped.copy()

        T_be = np.linalg.inv(base_T_world(self.sim, self.robots[robot_idx].robot_model.root_body)) @ T_world
        q_seed = self._q_seeds[robot_idx]
        q = self._frantik_seed(T_be, q_seed)
        if q is None:
            return None, float("inf")

        q = self._refine_mujoco(robot_idx, target_world, q)
        self._q_seeds[robot_idx] = q.copy()

        site_id = self.sim.model.site_name2id(f"gripper{robot_idx}_grip_site")
        self.sim.forward()
        err = float(np.linalg.norm(self.sim.data.site_xpos[site_id] - target_world))
        return q, err

    def _frantik_seed(self, T_be: np.ndarray, q_seed: np.ndarray) -> Optional[np.ndarray]:
        best_q = None
        best_err = float("inf")

        for q7 in np.linspace(-2.8973, 2.8973, self.Q7_SAMPLES):
            sols = frantik.ik(T_be, float(q7), q_seed)
            for s in sols:
                if np.any(np.isnan(s)):
                    continue
                err = float(np.linalg.norm(s - q_seed))
                if err < best_err:
                    best_err = err
                    best_q = s.copy()
            q_cc = frantik.cc_ik(T_be, float(q7), q_seed)
            if not np.any(np.isnan(q_cc)):
                err = float(np.linalg.norm(q_cc - q_seed))
                if err < best_err:
                    best_err = err
                    best_q = q_cc.copy()

        return best_q

    def _refine_mujoco(
        self,
        robot_idx: int,
        target_world: np.ndarray,
        q_init: np.ndarray,
    ) -> np.ndarray:
        robot = self.robots[robot_idx]
        joint_addrs = robot.joint_indexes
        site_id = self.sim.model.site_name2id(f"gripper{robot_idx}_grip_site")
        q = q_init.copy()

        for _ in range(self.REFINE_ITERS):
            self.sim.data.qpos[joint_addrs] = q
            self.sim.forward()
            pos = self.sim.data.site_xpos[site_id]
            err = target_world - pos
            if float(np.linalg.norm(err)) < self.REFINE_TOL:
                break

            import mujoco
            jacp = np.zeros((3, self.sim.model.nv))
            mujoco.mj_jacSite(self.sim.model._model, self.sim.data._data, jacp, None, site_id)
            J = jacp[:, joint_addrs]
            dq = J.T @ np.linalg.solve(J @ J.T + self.REFINE_LAMBDA * np.eye(3), err)
            q = q + dq
            lo, hi = PANDA_JOINT_LIMITS[:, 0], PANDA_JOINT_LIMITS[:, 1]
            q = np.clip(q, lo, hi)

        return q

    @staticmethod
    def default_reachability_path() -> str:
        return os.path.join(os.path.dirname(__file__), "..", "b", "reachability_panda_bimanual.npz")
