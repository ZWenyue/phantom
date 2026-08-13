"""
Stage B Processor (Contact-Grounded Retargeting) — whole-trajectory optimization.

Consumes the Stage A intent (``intent.npz``) and solves for a feasible, smooth,
consistent joint trajectory ``q_{1:T}`` with MuJoCo forward kinematics in the loop
(design doc §2.6). It **replaces** the legacy per-frame analytical IK + GP/SLERP
smoothing + frame dropping:

  * no hard IK "no solution" (feasible-by-construction best-effort optimum);
  * no dropped frames (every frame gets a ``q_t``);
  * obs-action consistency by construction (labels are ``q_t`` / ``FK(q_t)``).

The optimization math lives in ``phantom/traj_opt.py`` (MuJoCo-free, unit-tested).
This module provides the Panda FK/Jacobian (:class:`MujocoPandaArm`), the
robot-frame -> world-frame target transform, an optional per-frame frantik/DLS
warm start, and result saving + diagnostics.
"""

import logging
import os
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from phantom.processors.base_processor import BaseProcessor
from phantom.processors.paths import Paths
from phantom.panda_frantik_ik import PANDA_JOINT_LIMITS
from phantom.traj_opt import TrajectoryOptimizer, TrajOptConfig, ArmKinematics

logger = logging.getLogger(__name__)


def _make_single_arm_env():
    """Headless robosuite single-arm ``Phantom`` env for FK/Jacobian (no rendering).

    Uses the **same env as the single-arm renderer** (``TwinRobot`` / Stage C) so the
    optimized ``q`` is rendered consistently: same robot base (translation-only, at
    ``[-0.56, 0, 0.912]``, identity rotation) and same ``gripper0_grip_site``.
    """
    from robosuite.controllers import load_controller_config
    from robomimic.envs.env_robosuite import EnvRobosuite
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs=dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=["frontview_image"]))
    )
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["control_delta"] = False
    controller_config["uncouple_pos_ori"] = False
    options = dict(
        env_name="Phantom",
        robots=["Panda"],
        gripper_types=["Robotiq85Gripper"],
        controller_configs=controller_config,
        camera_heights=240,
        camera_widths=240,
        camera_segmentations="instance",
        direct_gripper_control=True,
        use_depth_obs=False,
        camera_pos=np.array([0, 0, 1.5]),
        camera_quat_wxyz=np.array([1, 0, 0, 0]),
        camera_sensorsize=np.array([6.0, 6.0]),
        camera_principalpixel=np.array([0.0, 0.0]),
        camera_focalpixel=np.array([400.0, 400.0]),
    )
    env = EnvRobosuite(
        **options, render=False, render_offscreen=False, use_image_obs=False,
        camera_names=["frontview"], control_freq=20,
    )
    env.reset()
    return env


class MujocoPandaArm(ArmKinematics):
    """Panda arm FK + world-frame Jacobians via MuJoCo.

    Also exposes the robot base pose (read from the env) so intent targets given in
    the pipeline *robot frame* (base-relative) can be mapped to the world frame the
    FK site lives in — no hard-coded transform.
    """

    def __init__(self, env, robot_idx: int = 0):
        import mujoco
        self._mj = mujoco
        self.env = env
        self.sim = env.env.sim
        self.robot = env.env.robots[robot_idx]
        self.robot_idx = robot_idx
        self.joint_addrs = np.asarray(self.robot.joint_indexes)
        self.site_id = self.sim.model.site_name2id(f"gripper{robot_idx}_grip_site")
        self.n_dof = len(self.joint_addrs)
        self.q_min = PANDA_JOINT_LIMITS[:, 0].copy()
        self.q_max = PANDA_JOINT_LIMITS[:, 1].copy()
        self.q_neutral = np.array(self.robot.robot_model.init_qpos, dtype=float)
        self._nv = self.sim.model.nv
        # Base pose (robot frame -> world). For env_name="Phantom" this is a pure
        # translation with identity rotation, but we read it to stay general.
        bid = self.sim.model.body_name2id(f"robot{robot_idx}_base")
        self.base_t = self.sim.data.body_xpos[bid].copy()
        self.base_R = self.sim.data.body_xmat[bid].reshape(3, 3).copy()

    def world_pos(self, p_robot: np.ndarray) -> np.ndarray:
        return (self.base_R @ np.asarray(p_robot).T).T + self.base_t

    def world_R(self, R_robot: np.ndarray) -> np.ndarray:
        return self.base_R @ R_robot

    def robot_pos(self, p_world: np.ndarray) -> np.ndarray:
        return (self.base_R.T @ (np.asarray(p_world) - self.base_t).T).T

    def robot_R(self, R_world: np.ndarray) -> np.ndarray:
        return self.base_R.T @ R_world

    def _forward(self, q: np.ndarray) -> None:
        self.sim.data.qpos[self.joint_addrs] = q
        self.sim.forward()

    def fk(self, q):
        self._forward(q)
        pos = self.sim.data.site_xpos[self.site_id].copy()
        R = self.sim.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, R

    def jac(self, q):
        self._forward(q)
        jacp = np.zeros((3, self._nv))
        jacr = np.zeros((3, self._nv))
        self._mj.mj_jacSite(self.sim.model._model, self.sim.data._data, jacp, jacr, self.site_id)
        return jacp[:, self.joint_addrs], jacr[:, self.joint_addrs]


class StageBProcessor(BaseProcessor):
    """Stage B: whole-trajectory joint-space optimization consuming ``intent.npz``.

    Config keys (all optional):
        target_hand (str):            "left" (default) or "right" -> arm to optimize.
        stageb_warm_start (bool):     seed q_{1:T} with per-frame frantik/DLS IK.
        stageb_grip_rot_offset_deg (float): Rz offset to align the intent grasp
            frame with the pipeline gripper convention (HandModel uses Rz(90)).
        stageb_w_smooth / stageb_w_reg / stageb_w_vel / stageb_dq_max / stageb_max_nfev.
    """

    def __init__(self, args):
        super().__init__(args)
        self.arm_side = str(getattr(self.cfg, "target_hand", "left"))
        # Single-arm "Phantom" render env has one robot (robot0) regardless of side.
        self.robot_idx = 0
        self.warm_start = bool(getattr(self.cfg, "stageb_warm_start", True))
        self.grip_rot_offset_deg = float(getattr(self.cfg, "stageb_grip_rot_offset_deg", 90.0))
        # Orientation is in radians, position in meters; this scale rebalances the
        # two so orientation (subject to the parallel-jaw grasp's rotational slack)
        # does not overpower position at the high-weight grasp/release keyframes.
        self.ori_scale = float(getattr(self.cfg, "stageb_ori_scale", 0.2))
        self.opt_cfg = TrajOptConfig(
            w_smooth=float(getattr(self.cfg, "stageb_w_smooth", 1.0)),
            w_reg=float(getattr(self.cfg, "stageb_w_reg", 0.01)),
            w_vel=float(getattr(self.cfg, "stageb_w_vel", 0.0)),
            dq_max=float(getattr(self.cfg, "stageb_dq_max", 0.3)),
            max_nfev=int(getattr(self.cfg, "stageb_max_nfev", 200)),
            verbose=int(getattr(self.cfg, "stageb_verbose", 0)),
        )
        self._env = None

    # ------------------------------------------------------------------
    def _get_env(self):
        if self._env is None:
            logger.info("[stageb] building headless single-arm Panda env for FK/Jacobian")
            self._env = _make_single_arm_env()
        return self._env

    def process_one_demo(self, data_sub_folder: str) -> None:
        save_folder = self.get_save_folder(data_sub_folder)
        paths = self.get_paths(save_folder)

        if not os.path.exists(paths.intent):
            logger.warning("[stageb] no intent.npz at %s; run mode=intent first. Skipping.",
                           paths.intent)
            return

        intent = np.load(paths.intent, allow_pickle=True)
        p_robot = np.asarray(intent["p_target"], dtype=float)     # (T,3) robot frame
        R_robot = np.asarray(intent["R_target"], dtype=float)     # (T,3,3) robot frame
        p_valid = np.asarray(intent["p_valid"], dtype=bool)
        w_p = np.asarray(intent["w_p"], dtype=float)
        w_r = np.asarray(intent["w_r"], dtype=float) * self.ori_scale  # rad/m unit rebalance
        phase = np.asarray(intent["phase"])
        gripper_width = np.asarray(intent["gripper_width"], dtype=float)
        n = len(p_robot)
        logger.info("[stageb] demo=%s T=%d arm=%s", data_sub_folder, n, self.arm_side)

        env = self._get_env()
        kin = MujocoPandaArm(env, self.robot_idx)

        # Robot-frame intent targets -> world frame using the env's actual base pose
        # (single-arm "Phantom": translation-only). Apply the pipeline gripper
        # convention offset (Rz) to the orientation.
        p_world, R_world = self._targets_to_world(kin, p_robot, R_robot)

        # Warm start: cheap per-frame position-only DLS (temporally seeded) using the
        # same MuJoCo Jacobian; gives the optimizer a feasible neighborhood.
        q_init = self._warm_start(kin, p_world, n)

        optimizer = TrajectoryOptimizer(kin, self.opt_cfg)
        result = optimizer.optimize(p_world, R_world, w_p, w_r, p_valid=p_valid, q_init=q_init)

        self._save_results(paths, result, intent, kin, phase, gripper_width)
        self._save_diagnostic(paths, result, phase)
        logger.info("[stageb] done demo=%s -> %s", data_sub_folder, paths.joint_trajectory)

    # ------------------------------------------------------------------
    def _targets_to_world(self, kin: "MujocoPandaArm", p_robot: np.ndarray, R_robot: np.ndarray):
        n = len(p_robot)
        R_off = Rotation.from_euler("Z", self.grip_rot_offset_deg, degrees=True).as_matrix()
        p_world = kin.world_pos(p_robot)
        R_world = np.empty((n, 3, 3))
        for t in range(n):
            R_world[t] = kin.world_R(R_robot[t] @ R_off)   # intent grasp frame -> pipeline gripper
        return p_world, R_world

    def _warm_start(self, kin: "MujocoPandaArm", p_world: np.ndarray, n: int,
                    iters: int = 20, lam: float = 0.05) -> np.ndarray:
        q_init = np.tile(kin.q_neutral, (n, 1))
        if not self.warm_start:
            return q_init
        q = kin.q_neutral.copy()
        for t in range(n):
            for _ in range(iters):
                pos, _ = kin.fk(q)
                err = p_world[t] - pos
                if float(np.linalg.norm(err)) < 2e-3:
                    break
                jacp, _ = kin.jac(q)
                dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + lam * np.eye(3), err)
                q = np.clip(q + dq, kin.q_min, kin.q_max)
            q_init[t] = q
        logger.info("[stageb] DLS position warm-start done (%d frames)", n)
        return q_init

    # ------------------------------------------------------------------
    def _save_results(self, paths: Paths, result: dict, intent, kin, phase, gripper_width) -> None:
        os.makedirs(paths.stageb_processor, exist_ok=True)
        # World EE pose -> robot frame (for Stage C labels consistent with intent frame).
        ee_pos_world = np.asarray(result["ee_pos_world"], dtype=float)
        ee_R_world = np.asarray(result["ee_R_world"], dtype=float)
        ee_pos_robot = kin.robot_pos(ee_pos_world)
        ee_R_robot = np.einsum("ij,tjk->tik", kin.base_R.T, ee_R_world)

        np.savez(
            paths.joint_trajectory,
            q=result["q"],
            pos_err=result["pos_err"],
            ori_err=result["ori_err"],
            ee_pos_world=result["ee_pos_world"],
            ee_R_world=result["ee_R_world"],
            ee_pos_robot=ee_pos_robot.astype(np.float32),
            ee_R_robot=ee_R_robot.astype(np.float32),
            gripper_width=gripper_width.astype(np.float32),
            phase=phase,
            valid=np.ones(len(result["q"]), dtype=bool),
            arm_side=self.arm_side,
            robot_idx=np.int64(self.robot_idx),
            joint_limits=PANDA_JOINT_LIMITS,
            cost_initial=result["cost_initial"],
            cost_final=result["cost_final"],
            jerk_rms=result["jerk_rms"],
            nfev=result["nfev"],
            success=bool(result["success"]),
            grip_rot_offset_deg=np.float32(self.grip_rot_offset_deg),
        )
        logger.info(
            "[stageb] saved q_trajectory: pos_err mean=%.4f max=%.4f m | ori_err mean=%.3f rad | "
            "jerk_rms=%.4f | cost %.4g->%.4g",
            float(result["pos_err"].mean()), float(result["pos_err"].max()),
            float(result["ori_err"].mean()), float(result["jerk_rms"]),
            float(result["cost_initial"]), float(result["cost_final"]),
        )

    def _save_diagnostic(self, paths: Paths, result: dict, phase) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            logger.warning("[stageb] matplotlib unavailable, skipping diagnostic: %s", e)
            return
        pos_err = np.asarray(result["pos_err"]); ori_err = np.asarray(result["ori_err"])
        n = len(pos_err); x = np.arange(n)
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax0.plot(x, pos_err * 100, color="tab:red", lw=1.2, label="pos_err (cm)")
        ax0.axhline(5.0, color="gray", ls="--", lw=0.8, label="old drop thresh 5cm")
        ax0.set_ylabel("position error (cm)"); ax0.legend(loc="upper right", fontsize=8)
        ax1.plot(x, np.degrees(ori_err), color="tab:blue", lw=1.2, label="ori_err (deg)")
        ax1.set_ylabel("orientation error (deg)"); ax1.set_xlabel("frame")
        ax1.legend(loc="upper right", fontsize=8)
        # phase shading
        shade = {1: ("orange", 0.18), 2: ("gray", 0.10), 3: ("purple", 0.18)}
        for ph, (col, a) in shade.items():
            seg = np.asarray(phase) == ph
            t = 0
            while t < n:
                if seg[t]:
                    s = t
                    while t < n and seg[t]:
                        t += 1
                    for ax in (ax0, ax1):
                        ax.axvspan(s, t - 1, color=col, alpha=a)
                else:
                    t += 1
        ax0.set_title(
            f"Stage B tracking — feasible-by-construction (no dropped frames) | "
            f"jerk_rms={float(result['jerk_rms']):.4f} success={result['success']}"
        )
        fig.tight_layout()
        try:
            fig.savefig(str(paths.stageb_diagnostic), dpi=120, bbox_inches="tight")
        except Exception as e:  # pragma: no cover
            logger.warning("[stageb] failed to write diagnostic: %s", e)
        finally:
            plt.close(fig)
