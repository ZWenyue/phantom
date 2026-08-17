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


class RetargetInpaintProcessor(RobotInpaintProcessor):
    """Render Stage B ``q_{1:T}`` onto the demo video and emit consistent labels."""

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        if self.bimanual_setup != "single_arm":
            logger.warning(
                "[retarget] only single_arm is implemented (got %s); the contact-grounded "
                "pipeline is single-arm for now.", self.bimanual_setup,
            )
        # Simple silhouette overlay by default; depth-aware occlusion is opt-in and
        # requires the human depth map to be aligned to the render resolution.
        self.use_depth = bool(getattr(self.cfg, "retarget_use_depth", False))

        # Trajectory-level quality gate thresholds (replace per-frame drop).
        self.key_pos_thresh = float(getattr(self.cfg, "retarget_key_pos_thresh", 0.03))
        self.vel_thresh = float(getattr(self.cfg, "retarget_vel_thresh", 0.5))
        self.jerk_thresh = float(getattr(self.cfg, "retarget_jerk_thresh", 0.05))
        self.frame_pos_thresh = float(getattr(self.cfg, "retarget_frame_pos_thresh", 0.05))

        self._grip_addrs: Optional[list] = None
        self._grip_ranges: Optional[list] = None

    # ------------------------------------------------------------------
    def process_one_demo(self, data_sub_folder: str) -> None:
        save_folder = self.get_save_folder(data_sub_folder)
        paths = self.get_paths(save_folder)

        if not os.path.exists(paths.joint_trajectory):
            logger.warning("[retarget] no q_trajectory.npz at %s; run mode=stageb first. Skipping.",
                           paths.joint_trajectory)
            return

        # Clean env state per demo (rebuilds TwinRobot single-arm).
        self.__del__()
        self._initialize_robot()

        qd = np.load(paths.joint_trajectory, allow_pickle=True)
        q_all = np.asarray(qd["q"], dtype=float)               # (T, 7)
        ee_pos_robot = np.asarray(qd["ee_pos_robot"], dtype=float)  # (T, 3) robot frame
        ee_R_robot = np.asarray(qd["ee_R_robot"], dtype=float)     # (T, 3, 3) robot frame
        gripper_width = np.asarray(qd["gripper_width"], dtype=float)
        phase = np.asarray(qd["phase"])
        pos_err = np.asarray(qd["pos_err"], dtype=float)
        n = len(q_all)

        open_width = 0.08
        if os.path.exists(paths.intent):
            intent = np.load(paths.intent, allow_pickle=True)
            open_width = float(np.asarray(intent.get("gripper_open_width", 0.08)))

        accept, reasons, quality = self._quality_gate(qd)
        self._save_quality_report(paths, quality, accept, reasons)
        if not accept:
            logger.warning("[retarget] demo %s PRUNED by trajectory quality gate: %s",
                           data_sub_folder, "; ".join(reasons))
            return
        logger.info("[retarget] demo %s accepted (key_pos=%.1fcm vmax=%.2f jerk=%.3f)",
                    data_sub_folder, quality["key_pos"] * 100, quality["vmax"], quality["jerk"])

        T_c2r_seq = self._load_T_cam2robot_seq(paths, n)
        background = self._load_background(paths, n)
        gripper_actions, gripper_widths = self._compute_gripper_actions(gripper_width.copy())

        sequence, img_overlay = self._render_trajectory(
            q_all, ee_pos_robot, ee_R_robot, gripper_width, open_width,
            gripper_actions, gripper_widths, background, T_c2r_seq=T_c2r_seq,
        )
        self._save_results_retarget(paths, sequence, img_overlay, background, phase, pos_err)
        logger.info("[retarget] done demo=%s -> %s", data_sub_folder, paths.retarget_video_overlay)

    # ------------------------------------------------------------------
    def _render_trajectory(self, q_all, ee_pos_robot, ee_R_robot, gripper_width, open_width,
                           gripper_actions, gripper_widths, background, T_c2r_seq=None):
        from tqdm import tqdm
        sequence = TrainingDataSequence()
        img_overlay: List[np.ndarray] = []
        is_left = self.target_hand == "left"
        n_joints = q_all.shape[1]
        zeros_j = np.zeros(n_joints)

        for idx in tqdm(range(len(q_all)), desc="Retarget render"):
            if T_c2r_seq is not None:
                self._set_render_camera_T_c2r(T_c2r_seq[idx])
            results = self._render_joint_positions(q_all[idx], gripper_width[idx], open_width)
            if self.use_depth and "imgs_depth" in results:
                overlay = self._process_robot_overlay_with_depth(
                    background[idx], np.zeros(background[idx].shape[:2], np.uint8),
                    results["imgs_depth"], results,
                )
            else:
                overlay = self._process_robot_overlay(background[idx], results)
            img_overlay.append(overlay)

            quat = Rotation.from_matrix(ee_R_robot[idx]).as_quat()  # xyzw
            if is_left:
                ap_l, ao_l, jp_l = ee_pos_robot[idx], quat, q_all[idx]
                ap_r, ao_r, jp_r = np.zeros(3), np.zeros(4), zeros_j
                ga_l, ga_r = gripper_actions[idx], 0.0
                gw_l, gw_r = gripper_widths[idx], 0.0
            else:
                ap_r, ao_r, jp_r = ee_pos_robot[idx], quat, q_all[idx]
                ap_l, ao_l, jp_l = np.zeros(3), np.zeros(4), zeros_j
                ga_r, ga_l = gripper_actions[idx], 0.0
                gw_r, gw_l = gripper_widths[idx], 0.0

            sequence.add_frame(TrainingData(
                frame_idx=idx,
                valid=True,
                action_pos_left=ap_l,
                action_orixyzw_left=ao_l,
                action_pos_right=ap_r,
                action_orixyzw_right=ao_r,
                action_gripper_left=ga_l,
                action_gripper_right=ga_r,
                gripper_width_left=gw_l,
                gripper_width_right=gw_r,
                joint_pos_left=jp_l,
                joint_pos_right=jp_r,
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
                if depth is not None:
                    depth = depth[:, nrm:W - nrm]

        robot_mask = (seg > 0).astype(np.uint8)  # single-arm scene: any instance = robot
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
        pcd_path = getattr(paths, "object_pcd", None)
        if pcd_path is None or not os.path.exists(pcd_path):
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

    def _set_render_camera_T_c2r(self, T_c2r: np.ndarray) -> None:
        """Move the MuJoCo frontview to camera-to-robot ``T_c2r`` (robot frame)."""
        T = np.asarray(T_c2r, dtype=np.float64)
        R = np.array(T[:3, :3], dtype=np.float64, copy=True)
        ori = self._convert_real_camera_ori_to_mujoco(R)
        pos_world = T[:3, 3] + TwinRobot.DEFAULT_ROBOT_BASE_POS
        env = self.twin_robot.env.env
        sim = env.sim
        cam = self.twin_robot.camera_name
        if hasattr(sim.model, "camera_name2id"):
            cid = sim.model.camera_name2id(cam)
        else:
            import mujoco
            cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        sim.model.cam_pos[cid] = pos_world
        sim.model.cam_quat[cid] = ori

    def _set_gripper_qpos(self, sim, width: float, open_width: float) -> None:
        """Set gripper finger qpos to a plausible opening from the intended width."""
        if self._grip_addrs is None:
            self._grip_addrs, self._grip_ranges = [], []
            robot = self.twin_robot.env.env.robots[0]
            gripper = getattr(robot, "gripper", None)
            names = list(getattr(gripper, "joints", []) or [])
            for nm in names:
                try:
                    addr = sim.model.get_joint_qpos_addr(nm)
                    if isinstance(addr, tuple):
                        addr = addr[0]
                    jid = sim.model.joint_name2id(nm)
                    lo, hi = sim.model.jnt_range[jid]
                    self._grip_addrs.append(int(addr))
                    self._grip_ranges.append((float(lo), float(hi)))
                except Exception:  # noqa: BLE001 - best-effort gripper posing
                    continue
            logger.info("[retarget] gripper joints for rendering: %d", len(self._grip_addrs))
        ow = open_width if open_width > 1e-6 else 0.08
        closure = float(np.clip((ow - width) / ow, 0.0, 1.0))  # 0 = open, 1 = closed
        for addr, (lo, hi) in zip(self._grip_addrs, self._grip_ranges):
            sim.data.qpos[addr] = lo + closure * (hi - lo)

    # ------------------------------------------------------------------
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

        out = []
        for im in src:
            H, W = im.shape[:2]
            if self.square:
                if W != H:
                    d = (W - H) // 2
                    im = im[:, d:W - d]
                im = cv2.resize(im, (self.output_resolution, self.output_resolution))
            else:
                im = cv2.resize(im, (int(W / H * self.output_resolution), self.output_resolution))
            out.append(im)
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
