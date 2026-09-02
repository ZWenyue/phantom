"""
Retarget Inpainting Processor (contact-grounded retargeting, Stage C).

This is the Stage C counterpart of :class:`RobotInpaintProcessor` for the
contact-grounded pipeline (intent -> stageb -> retarget_inpaint). Instead of
solving per-frame IK/OSC to human EE targets and dropping frames whose tracking
error is large, it:

1. Loads the whole-trajectory joint solution ``q_{1:T}`` from Stage B
   (``q_trajectory.npz``) — feasible + smooth by construction.
2. Renders the robot by **directly injecting** ``q_t`` into the *same* single-arm
   MuJoCo env whose FK Stage B optimized against (``env_name="Phantom"``), so the
   rendered pixels are exactly ``FK(q_t)`` — observation/action consistency by
   construction (no OSC residual).
3. Flips the labels relative to the human-target convention: ``joint_pos`` stores
   ``q_t`` and the task-space ``action`` stores ``FK(q_t)`` (robot frame), i.e. the
   labels describe *what is rendered*, not the (possibly infeasible) human target.
4. Replaces per-frame ``TRACKING_ERROR_THRESHOLD`` frame dropping with a
   **trajectory-level quality gate** (Stage B already guarantees no IK failures).

Rendering requires an EGL/offscreen GL context (see ``b/run_process.sh`` lines
66-69: ``MUJOCO_GL=egl`` etc.).
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapy as media
from scipy.spatial.transform import Rotation

from phantom.processors.phantom_data import TrainingData, TrainingDataSequence
from phantom.processors.paths import Paths
from phantom.processors.robotinpaint_processor import RobotInpaintProcessor
from phantom.twin_robot import TwinRobot

logger = logging.getLogger(__name__)


def crop_square_if_needed(frame: np.ndarray, square: bool) -> np.ndarray:
    """Center-crop a wide frame to square, matching ``_load_background``."""
    if not square:
        return frame
    h, w = frame.shape[:2]
    if w == h:
        return frame
    d = (w - h) // 2
    if d <= 0:
        return frame
    return frame[:, d:w - d]


def prepare_scene_depth_for_overlay(depth: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resize metric depth to the sim mask size; invalid pixels never occlude the robot."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    height, width = hw
    if d.shape[:2] != (height, width):
        d = cv2.resize(d, (width, height), interpolation=cv2.INTER_NEAREST)
    out = np.array(d, dtype=np.float32, copy=True)
    out[~np.isfinite(out) | (out <= 1e-6)] = np.inf
    return out


def _log_depth_overlay_stats(results: Dict[str, np.ndarray], real_d: np.ndarray) -> None:
    mask = results["robot_mask"] == 1
    sim = np.squeeze(results["depth_img"])
    if sim.shape != real_d.shape:
        logger.info("[retarget] depth overlay shape sim=%s real=%s (will resize in overlay)",
                    sim.shape, real_d.shape)
        return
    if not np.any(mask):
        return
    sim_m = sim[mask]
    real_m = real_d[mask]
    real_ok = np.isfinite(real_m)
    real_med = float(np.median(real_m[real_ok])) if np.any(real_ok) else float("nan")
    logger.info(
        "[retarget] depth overlay (robot pixels): sim median=%.3fm  real median=%.3fm  real_valid=%.1f%%",
        float(np.median(sim_m)), real_med, 100.0 * float(np.mean(real_ok)),
    )


class RetargetInpaintProcessor(RobotInpaintProcessor):
    """Render Stage B ``q_{1:T}`` onto the demo video and emit consistent labels."""

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        if self.bimanual_setup != "single_arm" and not self.contact_bimanual():
            logger.warning(
                "[retarget] only single_arm / contact_bimanual is implemented (got %s)",
                self.bimanual_setup,
            )
        # Depth-aware occlusion: robot pixels cover the background only when they
        # are closer than the (aligned) DA3 metric depth map.
        self.use_depth = bool(getattr(self.cfg, "retarget_use_depth", False))
        self.distal_only = bool(getattr(self.cfg, "retarget_distal_only", False))
        distal_bodies = getattr(
            self.cfg,
            "retarget_distal_body_tokens",
            ["link7", "eef", "gripper", "finger", "knuckle"],
        )
        self.distal_body_tokens = tuple(str(x).lower() for x in distal_bodies)
        self._distal_geom_ids_cache: Dict[int, np.ndarray] = {}
        self._distal_filter_warned = False
        self._configure_distal_render_geometry()

        # Trajectory-level quality gate thresholds (replace per-frame drop).
        self.key_pos_thresh = float(getattr(self.cfg, "retarget_key_pos_thresh", 0.03))
        self.vel_thresh = float(getattr(self.cfg, "retarget_vel_thresh", 0.5))
        self.jerk_thresh = float(getattr(self.cfg, "retarget_jerk_thresh", 0.05))
        self.frame_pos_thresh = float(getattr(self.cfg, "retarget_frame_pos_thresh", 0.05))

        self._grip_addrs: Optional[list] = None
        self._grip_ranges: Optional[list] = None
        self._grip_by_robot: Dict[int, Tuple[list, list]] = {}

    def _initialize_robot(self) -> None:
        if self.contact_bimanual():
            saved = self.bimanual_setup
            self.bimanual_setup = self.contact_bimanual_layout()
            try:
                RobotInpaintProcessor._initialize_robot(self)
            finally:
                self.bimanual_setup = saved
            if hasattr(self, "_distal_geom_ids_cache"):
                self._distal_geom_ids_cache.clear()
            self._configure_distal_render_geometry()
            return
        super()._initialize_robot()
        if hasattr(self, "_distal_geom_ids_cache"):
            self._distal_geom_ids_cache.clear()
        self._configure_distal_render_geometry()

    def _configure_distal_render_geometry(self) -> None:
        """Hide proximal robot geoms so they cannot occlude retained distal links."""
        if not getattr(self, "distal_only", False) or not hasattr(self, "twin_robot"):
            return
        sim = self.twin_robot.env.env.sim
        hidden = []
        for geom_id in range(sim.model.ngeom):
            body_id = int(sim.model.geom_bodyid[geom_id])
            body_name = sim.model.body_id2name(body_id) or ""
            body_lower = body_name.lower()
            is_robot = "robot" in body_lower or "gripper" in body_lower
            is_distal = any(token in body_lower for token in self.distal_body_tokens)
            if is_robot and not is_distal:
                hidden.append(geom_id)
        if hidden:
            sim.model.geom_rgba[np.asarray(hidden, dtype=np.int32), 3] = 0.0
            sim.forward()
        logger.info(
            "[retarget] distal-only hid %d proximal robot geoms before rendering",
            len(hidden),
        )

    # ------------------------------------------------------------------
    def process_one_demo(self, data_sub_folder: str) -> None:
        save_folder = self.get_save_folder(data_sub_folder)
        paths = self.get_paths(save_folder)

        if not os.path.exists(paths.joint_trajectory) and not (
            os.path.exists(paths.joint_trajectory_right) or os.path.exists(paths.joint_trajectory_left)
        ):
            logger.warning("[retarget] no q_trajectory.npz at %s; run mode=stageb first. Skipping.",
                           paths.joint_trajectory)
            return

        # Clean env state per demo (rebuilds TwinRobot single-arm).
        self.__del__()
        self._initialize_robot()

        traj = self._load_arm_trajectories(paths)
        if not traj:
            logger.warning("[retarget] no q_trajectory at %s; run mode=stageb first. Skipping.",
                           paths.joint_trajectory)
            return
        n = max(len(np.asarray(qd["q"])) for qd in traj.values())
        phase = np.asarray(next(iter(traj.values()))["phase"])
        pos_err = np.asarray(next(iter(traj.values()))["pos_err"], dtype=float)

        accept, reasons, quality = self._quality_gate_arms(traj)
        self._save_quality_report(paths, quality, accept, reasons)
        if not accept:
            logger.warning("[retarget] demo %s PRUNED by trajectory quality gate: %s",
                           data_sub_folder, "; ".join(reasons))
            return
        logger.info("[retarget] demo %s accepted (key_pos=%.1fcm vmax=%.2f jerk=%.3f arms=%s)",
                    data_sub_folder, quality["key_pos"] * 100, quality["vmax"], quality["jerk"],
                    ",".join(sorted(traj)))

        T_c2r_seq = self._load_T_cam2robot_seq(paths, n)
        background = self._load_background(paths, n)
        scene_depth = self._load_scene_depth(paths, n) if self.use_depth else None
        if self.use_depth and scene_depth is None:
            logger.warning(
                "[retarget] retarget_use_depth=true but no usable depth.npy; falling back to silhouette overlay"
            )

        sequence, img_overlay = self._render_trajectory(
            traj, background, paths, T_c2r_seq=T_c2r_seq, scene_depth=scene_depth,
        )
        self._save_results_retarget(paths, sequence, img_overlay, background, phase, pos_err)
        logger.info("[retarget] done demo=%s -> %s", data_sub_folder, paths.retarget_video_overlay)

    # ------------------------------------------------------------------
    def _load_arm_trajectories(self, paths: Paths) -> Dict[str, Any]:
        traj: Dict[str, Any] = {}
        sides = self.intent_sides() if self.contact_bimanual() else (
            [self.target_hand] if self.target_hand in ("left", "right") else ["right"]
        )
        for side in sides:
            hp = paths.for_hand(side) if self.contact_bimanual() else paths
            p = hp.joint_trajectory
            if not os.path.exists(p) and os.path.exists(paths.joint_trajectory) and len(sides) == 1:
                p = paths.joint_trajectory
            if os.path.exists(p):
                traj[side] = np.load(p, allow_pickle=True)
        if not traj and os.path.exists(paths.joint_trajectory):
            side = self.target_hand if self.target_hand in ("left", "right") else "right"
            traj[side] = np.load(paths.joint_trajectory, allow_pickle=True)
        return traj

    def _quality_gate_arms(self, traj: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        reasons: List[str] = []
        qualities = {}
        active = []
        for side, qd in traj.items():
            parked = bool(np.asarray(qd["parked"]).reshape(-1)[0]) if "parked" in qd.files else False
            if parked:
                qualities[side] = dict(parked=True, key_pos=0.0, vmax=0.0, jerk=0.0, viol=0)
                continue
            accept, why, q = self._quality_gate(qd)
            qualities[side] = dict(q, parked=False)
            active.append(side)
            if not accept:
                reasons.extend(f"{side}: {r}" for r in why)
        if not active:
            reasons.append("no active (non-parked) arm")
        key_pos = max((qualities[s]["key_pos"] for s in active), default=0.0)
        vmax = max((qualities[s]["vmax"] for s in active), default=0.0)
        jerk = max((qualities[s]["jerk"] for s in active), default=0.0)
        quality = dict(
            key_pos=key_pos, vmax=vmax, jerk=jerk,
            viol=int(sum(qualities[s].get("viol", 0) for s in active)),
            frame_ok=qualities[active[0]]["frame_ok"] if active else np.array([], dtype=bool),
            pos_err=qualities[active[0]]["pos_err"] if active else np.array([]),
            arms=qualities,
        )
        return len(reasons) == 0, reasons, quality

    def _open_width_for(self, paths: Paths, side: str) -> float:
        cands = [paths.intent]
        if self.contact_bimanual():
            cands.insert(0, paths.for_hand(side).intent)
        for p in cands:
            if os.path.exists(p):
                intent = np.load(p, allow_pickle=True)
                return float(np.asarray(intent.get("gripper_open_width", 0.08)))
        return 0.08

    def _pad_traj(self, arr: np.ndarray, n: int) -> np.ndarray:
        a = np.asarray(arr)
        if len(a) >= n:
            return a[:n]
        pad = np.repeat(a[-1:], n - len(a), axis=0)
        return np.concatenate([a, pad], axis=0)

    def _render_trajectory(self, traj, background, paths, T_c2r_seq=None, scene_depth=None):
        from tqdm import tqdm
        sequence = TrainingDataSequence()
        img_overlay: List[np.ndarray] = []
        n = len(background)
        logged_depth_stats = False

        def pack(side, key, default):
            if side not in traj:
                return default
            return self._pad_traj(traj[side][key], n)

        q_r = pack("right", "q", np.zeros((n, 7)))
        q_l = pack("left", "q", np.zeros((n, 7)))
        ee_r = pack("right", "ee_pos_robot", np.zeros((n, 3)))
        ee_l = pack("left", "ee_pos_robot", np.zeros((n, 3)))
        Rr = pack("right", "ee_R_robot", np.tile(np.eye(3), (n, 1, 1)))
        Rl = pack("left", "ee_R_robot", np.tile(np.eye(3), (n, 1, 1)))
        gw_r = pack("right", "gripper_width", np.zeros(n))
        gw_l = pack("left", "gripper_width", np.zeros(n))
        open_r = self._open_width_for(paths, "right")
        open_l = self._open_width_for(paths, "left")
        ga_r, gwv_r = self._compute_gripper_actions(np.asarray(gw_r, dtype=float).copy())
        ga_l, gwv_l = self._compute_gripper_actions(np.asarray(gw_l, dtype=float).copy())
        zeros_j = np.zeros(7)

        for idx in tqdm(range(n), desc="Retarget render"):
            if T_c2r_seq is not None:
                self._set_render_camera_T_c2r(T_c2r_seq[idx])
            if self.contact_bimanual():
                results = self._render_joint_positions_bimanual(
                    q_r[idx], q_l[idx], float(gw_r[idx]), float(gw_l[idx]), open_r, open_l,
                )
            else:
                side = "left" if "left" in traj and "right" not in traj else "right"
                q = q_l[idx] if side == "left" else q_r[idx]
                w = float(gw_l[idx] if side == "left" else gw_r[idx])
                results = self._render_joint_positions(q, w, open_r)
            use_depth_overlay = (
                self.use_depth
                and scene_depth is not None
                and "depth_img" in results
            )
            if use_depth_overlay:
                robot_hw = results["robot_mask"].shape[:2]
                real_d = prepare_scene_depth_for_overlay(scene_depth[idx], robot_hw)
                if not logged_depth_stats:
                    _log_depth_overlay_stats(results, real_d)
                    logged_depth_stats = True
                hand_mask = np.zeros(robot_hw, dtype=np.uint8)
                overlay = self._process_robot_overlay_with_depth(
                    background[idx], hand_mask, real_d, results,
                )
            else:
                overlay = self._process_robot_overlay(background[idx], results)
            img_overlay.append(overlay)

            quat_r = Rotation.from_matrix(Rr[idx]).as_quat()
            quat_l = Rotation.from_matrix(Rl[idx]).as_quat()
            has_r, has_l = "right" in traj, "left" in traj
            sequence.add_frame(TrainingData(
                frame_idx=idx,
                valid=True,
                action_pos_left=ee_l[idx] if has_l else np.zeros(3),
                action_orixyzw_left=quat_l if has_l else np.zeros(4),
                action_pos_right=ee_r[idx] if has_r else np.zeros(3),
                action_orixyzw_right=quat_r if has_r else np.zeros(4),
                action_gripper_left=float(ga_l[idx]) if has_l else 0.0,
                action_gripper_right=float(ga_r[idx]) if has_r else 0.0,
                gripper_width_left=float(gwv_l[idx]) if has_l else 0.0,
                gripper_width_right=float(gwv_r[idx]) if has_r else 0.0,
                joint_pos_left=q_l[idx] if has_l else zeros_j,
                joint_pos_right=q_r[idx] if has_r else zeros_j,
            ))
        return sequence, img_overlay

    # ------------------------------------------------------------------
    def _render_joint_positions(self, q: np.ndarray, width: float, open_width: float) -> Dict[str, np.ndarray]:
        """Inject ``q`` (+ gripper) into the render env and return overlay inputs.

        Renders the exact ``FK(q)`` — no controller stepping — so the rendered
        arm matches the recorded ``joint_pos`` label.
        """
        env = self.twin_robot.env.env
        sim = env.sim
        robot = env.robots[0]
        sim.data.qpos[robot.joint_indexes] = q
        self._set_gripper_qpos(sim, width, open_width)
        sim.forward()

        obs = env._get_observations(force_update=True)
        cam = self.twin_robot.camera_name
        rgb = np.asarray(obs[f"{cam}_image"])                       # (H, W, 3) uint8
        seg = np.asarray(obs[f"{cam}_segmentation_instance"])[..., 0]
        element_seg = obs.get(f"{cam}_segmentation_element")
        if element_seg is not None:
            element_seg = np.asarray(element_seg)[..., 0]

        depth = None
        if self.use_depth:
            from robosuite.utils.camera_utils import get_real_depth_map
            depth = np.squeeze(get_real_depth_map(sim=sim, depth_map=obs[f"{cam}_depth"]))

        if self.square:
            H, W = rgb.shape[:2]
            nrm = (W - H) // 2
            if nrm > 0:
                rgb = rgb[:, nrm:W - nrm]
                seg = seg[:, nrm:W - nrm]
                if element_seg is not None:
                    element_seg = element_seg[:, nrm:W - nrm]
                if depth is not None:
                    depth = depth[:, nrm:W - nrm]

        robot_mask = self._robot_render_mask(seg, element_seg, sim)
        results: Dict[str, np.ndarray] = {
            "rgb_img": rgb.astype(np.float32) / 255.0,
            "robot_mask": robot_mask,
            "gripper_mask": np.zeros_like(robot_mask),
        }
        if depth is not None:
            results["depth_img"] = depth
        return results

    def _obs_camera_rgb_seg_depth(self, env, sim):
        obs = env._get_observations(force_update=True)
        cam = self.twin_robot.camera_name
        if f"{cam}_image" not in obs:
            cam = getattr(getattr(self.twin_robot, "camera_params", None), "name", cam)
        rgb = np.asarray(obs[f"{cam}_image"])
        seg = np.asarray(obs[f"{cam}_segmentation_instance"])[..., 0]
        element_seg = obs.get(f"{cam}_segmentation_element")
        if element_seg is not None:
            element_seg = np.asarray(element_seg)[..., 0]
        depth = None
        if self.use_depth:
            from robosuite.utils.camera_utils import get_real_depth_map
            depth = np.squeeze(get_real_depth_map(sim=sim, depth_map=obs[f"{cam}_depth"]))
        if self.square:
            H, W = rgb.shape[:2]
            nrm = (W - H) // 2
            if nrm > 0:
                rgb = rgb[:, nrm:W - nrm]
                seg = seg[:, nrm:W - nrm]
                if element_seg is not None:
                    element_seg = element_seg[:, nrm:W - nrm]
                if depth is not None:
                    depth = depth[:, nrm:W - nrm]
        return rgb, seg, element_seg, depth

    def _robot_render_mask(self, instance_seg, element_seg, sim) -> np.ndarray:
        """Return the full robot mask or a geom-filtered distal-link mask."""
        full_mask = np.asarray(instance_seg) > 0
        if not self.distal_only:
            return full_mask.astype(np.uint8)
        if element_seg is None:
            if not self._distal_filter_warned:
                logger.warning(
                    "[retarget] distal-only requested but element segmentation is unavailable; "
                    "using the full robot silhouette"
                )
                self._distal_filter_warned = True
            return full_mask.astype(np.uint8)

        model_key = id(sim.model)
        allowed_ids = self._distal_geom_ids_cache.get(model_key)
        if allowed_ids is None:
            ids = []
            selected_bodies = set()
            for geom_id in range(sim.model.ngeom):
                body_id = int(sim.model.geom_bodyid[geom_id])
                body_name = sim.model.body_id2name(body_id) or ""
                body_lower = body_name.lower()
                is_robot = "robot" in body_lower or "gripper" in body_lower
                if is_robot and any(token in body_lower for token in self.distal_body_tokens):
                    ids.append(geom_id)
                    selected_bodies.add(body_name)
            allowed_ids = np.asarray(ids, dtype=np.int32)
            self._distal_geom_ids_cache[model_key] = allowed_ids
            logger.info(
                "[retarget] distal-only keeps %d geoms on bodies=%s",
                len(allowed_ids),
                ",".join(sorted(selected_bodies)),
            )
        if allowed_ids.size == 0:
            if not self._distal_filter_warned:
                logger.warning(
                    "[retarget] distal-only selected no robot geoms; using the full robot silhouette"
                )
                self._distal_filter_warned = True
            return full_mask.astype(np.uint8)
        return np.isin(np.asarray(element_seg), allowed_ids).astype(np.uint8)

    def _render_joint_positions_bimanual(
        self, q_right, q_left, width_r, width_l, open_r, open_l,
    ) -> Dict[str, np.ndarray]:
        env = self.twin_robot.env.env
        sim = env.sim
        sim.data.qpos[env.robots[0].joint_indexes] = q_right
        sim.data.qpos[env.robots[1].joint_indexes] = q_left
        self._set_gripper_qpos(sim, width_r, open_r, robot_idx=0)
        self._set_gripper_qpos(sim, width_l, open_l, robot_idx=1)
        sim.forward()
        rgb, seg, element_seg, depth = self._obs_camera_rgb_seg_depth(env, sim)
        robot_mask = self._robot_render_mask(seg, element_seg, sim)
        results: Dict[str, np.ndarray] = {
            "rgb_img": rgb.astype(np.float32) / 255.0,
            "robot_mask": robot_mask,
            "gripper_mask": np.zeros_like(robot_mask),
        }
        if depth is not None:
            results["depth_img"] = depth
        return results

    def _load_T_cam2robot_seq(self, paths: Paths, n: int) -> Optional[np.ndarray]:
        """Per-frame camera-to-robot from Stage A (T_place @ T_c2w). None → JSON calib."""
        cands = [getattr(paths, "object_pcd", None)]
        if self.contact_bimanual():
            cands.extend([paths.object_pcd_right, paths.object_pcd_left])
        pcd_path = next((p for p in cands if p is not None and os.path.exists(p)), None)
        if pcd_path is None:
            return None
        pcd = np.load(pcd_path, allow_pickle=True)
        if "T_cam2robot_seq" not in pcd.files:
            return None
        T = np.asarray(pcd["T_cam2robot_seq"], dtype=np.float64)
        if T.ndim != 3 or T.shape[-2:] != (4, 4):
            logger.warning("[retarget] ignore bad T_cam2robot_seq shape %s", T.shape)
            return None
        if len(T) < n:
            pad = np.repeat(T[-1:], n - len(T), axis=0)
            T = np.concatenate([T, pad], axis=0)
        elif len(T) > n:
            T = T[:n]
        logger.info("[retarget] using per-frame T_cam2robot_seq (%d frames) for overlay camera", n)
        return T

    def _workspace_RT(self) -> Tuple[np.ndarray, np.ndarray]:
        """MuJoCo pose of the shared workspace origin.

        Bimanual contact-grounded: torso (not robot0), matching Stage B so
        moving the shoulders actually changes ego occlusion.
        """
        if self.contact_bimanual():
            from phantom.processors.stageb_processor import bimanual_torso_RT
            return bimanual_torso_RT(self.twin_robot.env)
        env = self.twin_robot.env.env
        sim = env.sim
        try:
            bid = sim.model.body_name2id("robot0_base")
            t = np.asarray(sim.data.body_xpos[bid], dtype=np.float64).copy()
            R = np.asarray(sim.data.body_xmat[bid], dtype=np.float64).reshape(3, 3).copy()
            return R, t
        except Exception:  # noqa: BLE001
            return np.eye(3), TwinRobot.DEFAULT_ROBOT_BASE_POS.astype(np.float64)

    def _set_render_camera_T_c2r(self, T_c2r: np.ndarray) -> None:
        """Move the render camera to camera-to-workspace ``T_c2r`` in MuJoCo world."""
        T = np.asarray(T_c2r, dtype=np.float64)
        Rw, tw = self._workspace_RT()
        R = Rw @ T[:3, :3]
        ori = self._convert_real_camera_ori_to_mujoco(R)
        pos_world = Rw @ T[:3, 3] + tw
        env = self.twin_robot.env.env
        sim = env.sim
        cam = self.twin_robot.camera_name
        if f"{cam}_image" not in getattr(env, "observation_names", []):
            cam_alt = getattr(getattr(self.twin_robot, "camera_params", None), "name", None)
            if cam_alt:
                cam = cam_alt
        if hasattr(sim.model, "camera_name2id"):
            cid = sim.model.camera_name2id(cam)
        else:
            import mujoco
            cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        sim.model.cam_pos[cid] = pos_world
        sim.model.cam_quat[cid] = ori

    def _set_gripper_qpos(self, sim, width: float, open_width: float, robot_idx: int = 0) -> None:
        """Set gripper finger qpos to a plausible opening from the intended width."""
        if robot_idx not in self._grip_by_robot:
            addrs, ranges = [], []
            robots = self.twin_robot.env.env.robots
            robot = robots[min(robot_idx, len(robots) - 1)]
            gripper = getattr(robot, "gripper", None)
            names = list(getattr(gripper, "joints", []) or [])
            for nm in names:
                try:
                    addr = sim.model.get_joint_qpos_addr(nm)
                    if isinstance(addr, tuple):
                        addr = addr[0]
                    jid = sim.model.joint_name2id(nm)
                    lo, hi = sim.model.jnt_range[jid]
                    addrs.append(int(addr))
                    ranges.append((float(lo), float(hi)))
                except Exception:  # noqa: BLE001 - best-effort gripper posing
                    continue
            self._grip_by_robot[robot_idx] = (addrs, ranges)
            logger.info("[retarget] gripper joints robot%d: %d", robot_idx, len(addrs))
        addrs, ranges = self._grip_by_robot[robot_idx]
        ow = open_width if open_width > 1e-6 else 0.08
        closure = float(np.clip((ow - width) / ow, 0.0, 1.0))
        for addr, (lo, hi) in zip(addrs, ranges):
            sim.data.qpos[addr] = lo + closure * (hi - lo)

    # ------------------------------------------------------------------
    def _align_spatial(self, frame: np.ndarray, interpolation: int) -> np.ndarray:
        """Center-crop (if square) and resize to the overlay output resolution."""
        im = crop_square_if_needed(frame, self.square)
        H, W = im.shape[:2]
        if self.square:
            size = (self.output_resolution, self.output_resolution)
        else:
            size = (int(W / H * self.output_resolution), self.output_resolution)
        return cv2.resize(im, size, interpolation=interpolation)

    def _load_scene_depth(self, paths: Paths, n: int) -> Optional[np.ndarray]:
        """Load DA3 metric ``depth.npy`` and align it like the RGB background.

        Overlay compares this (meters) against MuJoCo ``get_real_depth_map``
        (also meters). Invalid pixels are later treated as "do not occlude".
        """
        if not os.path.exists(paths.depth):
            logger.warning("[retarget] depth.npy missing at %s", paths.depth)
            return None
        depth = np.load(paths.depth)
        if depth.ndim == 4:
            depth = depth[..., 0]
        if depth.ndim != 3:
            logger.warning("[retarget] ignore depth.npy with shape %s", depth.shape)
            return None
        depth = np.asarray(depth, dtype=np.float32)
        if len(depth) != n:
            m = min(len(depth), n)
            logger.warning("[retarget] depth len %d != T %d; using first %d", len(depth), n, m)
            depth = depth[:m]
        aligned = np.stack(
            [self._align_spatial(depth[i], cv2.INTER_NEAREST) for i in range(len(depth))],
            axis=0,
        )
        finite = aligned[np.isfinite(aligned) & (aligned > 1e-6)]
        if finite.size == 0:
            logger.warning("[retarget] depth.npy has no valid metric values")
            return None
        logger.info(
            "[retarget] scene depth aligned %s, median=%.3fm p95=%.3fm",
            aligned.shape, float(np.median(finite)), float(np.percentile(finite, 95)),
        )
        return aligned

    def _load_background(self, paths: Paths, n: int) -> np.ndarray:
        """Background frames to composite the robot onto (output resolution).

        Prefers the human-inpainted video (hands removed) when it is aligned to the
        Stage A/B frame count; otherwise falls back to the raw original frames.
        """
        src: Optional[np.ndarray] = None
        if os.path.exists(paths.video_human_inpaint):
            vid = np.array(media.read_video(paths.video_human_inpaint))
            if len(vid) == n:
                src = vid
                logger.info("[retarget] background = human-inpaint video (hands removed)")
            else:
                logger.info("[retarget] human-inpaint video len %d != T %d; using original frames",
                            len(vid), n)
        if src is None:
            folder = paths.original_images_folder
            files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")],
                           key=lambda x: int(os.path.splitext(x)[0]))
            src = np.stack([cv2.cvtColor(cv2.imread(os.path.join(folder, f)), cv2.COLOR_BGR2RGB)
                            for f in files], axis=0)
            logger.info("[retarget] background = original frames (hands still visible)")

        if len(src) != n:
            m = min(len(src), n)
            logger.warning("[retarget] background len %d != T %d; using first %d", len(src), n, m)
            src = src[:m]

        out = [self._align_spatial(im, cv2.INTER_LINEAR) for im in src]
        return np.stack(out, axis=0)

    # ------------------------------------------------------------------
    def _quality_gate(self, qd) -> Tuple[bool, List[str], Dict[str, Any]]:
        q = np.asarray(qd["q"], dtype=float)
        pos_err = np.asarray(qd["pos_err"], dtype=float)
        phase = np.asarray(qd["phase"])
        jerk = float(qd["jerk_rms"])
        lim = np.asarray(qd["joint_limits"], dtype=float)
        T = len(q)

        dq = np.diff(q, axis=0) if T > 1 else np.zeros((1, q.shape[1]))
        vmax = float(np.abs(dq).max()) if len(dq) else 0.0
        key = np.isin(phase, [1, 3])  # grasp / release
        key_pos = float(pos_err[key].max()) if key.any() else float(pos_err.max())
        viol = int(((q < lim[:, 0] - 1e-6) | (q > lim[:, 1] + 1e-6)).sum())

        reasons: List[str] = []
        if key_pos > self.key_pos_thresh:
            reasons.append(f"key_pos {key_pos*100:.1f}cm > {self.key_pos_thresh*100:.0f}cm")
        if vmax > self.vel_thresh:
            reasons.append(f"vmax {vmax:.2f} > {self.vel_thresh:.2f} rad/frame")
        if jerk > self.jerk_thresh:
            reasons.append(f"jerk {jerk:.3f} > {self.jerk_thresh:.3f}")
        if viol > 0:
            reasons.append(f"joint_viol {viol}")

        quality = dict(
            key_pos=key_pos, vmax=vmax, jerk=jerk, viol=viol,
            frame_ok=(pos_err <= self.frame_pos_thresh),
            pos_err=pos_err,
        )
        return len(reasons) == 0, reasons, quality

    def _save_quality_report(self, paths: Paths, quality: Dict[str, Any], accept: bool,
                             reasons: List[str]) -> None:
        os.makedirs(paths.retarget_processor, exist_ok=True)
        np.savez(
            paths.retarget_quality,
            accept=bool(accept),
            reasons=np.array(reasons, dtype=object),
            key_pos=float(quality["key_pos"]),
            vmax=float(quality["vmax"]),
            jerk=float(quality["jerk"]),
            viol=int(quality["viol"]),
            frame_ok=quality["frame_ok"],
            pos_err=quality["pos_err"],
        )

    def _save_results_retarget(self, paths: Paths, sequence: TrainingDataSequence,
                               img_overlay: List[np.ndarray], background: np.ndarray,
                               phase: np.ndarray, pos_err: np.ndarray) -> None:
        os.makedirs(paths.retarget_processor, exist_ok=True)
        if len(img_overlay) == 0:
            logger.warning("[retarget] no overlay frames; skipping save")
            return
        self._save_video(str(paths.retarget_video_overlay), img_overlay)
        sequence.save(str(paths.retarget_training_data))
        self._save_diagnostic_montage(paths, background, img_overlay, phase)

    def _save_diagnostic_montage(self, paths: Paths, background: np.ndarray,
                                 img_overlay: List[np.ndarray], phase: np.ndarray) -> None:
        """Side-by-side (raw | overlay) montage at a few keyframes for visual QA."""
        n = len(img_overlay)
        grasp = np.where(np.asarray(phase) == 1)[0]
        release = np.where(np.asarray(phase) == 3)[0]
        picks = [0]
        if len(grasp):
            picks.append(int(grasp[0]))
        picks.append(n // 2)
        if len(release):
            picks.append(int(release[-1]))
        picks.append(n - 1)
        picks = sorted(set(int(np.clip(p, 0, n - 1)) for p in picks))

        rows = []
        for p in picks:
            pair = np.concatenate([background[p], img_overlay[p]], axis=1)
            label = {0: "free", 1: "grasp", 2: "transport", 3: "release"}.get(int(phase[p]), "?")
            cv2.putText(pair, f"f{p} {label}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1, cv2.LINE_AA)
            rows.append(pair)
        montage = np.concatenate(rows, axis=0)
        cv2.imwrite(str(paths.retarget_diagnostic), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
