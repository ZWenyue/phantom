"""
Intent Processor Module (Contact-Grounded Retargeting — Stage A)

This is the front end of the contact-grounded retargeting pipeline described in
``b/doc/research/contact_grounded_retargeting.md``. Its eventual job is to turn a
human video into a task-space *intent* (per-frame EE targets, contact/grasp
events, object-anchored grasp poses) that the Stage B trajectory optimizer
consumes.

Implemented scope:

    1. Determine the task object prompt (per-demo ``objects.json`` or config).
    2. Detect the object with Grounding-DINO and propagate the mask through the
       whole clip with SAM2 (forward + reverse from the best-confidence frame).
    3. Back-project the per-frame object mask with the metric depth map to build a
       per-frame object point cloud (camera frame + robot frame).
    4. Detect contact / segment free->grasp->transport->release phases
       (``_detect_contacts``): fingertip-object distance cue, with an
       object-motion fallback when no hand keypoints are available.
    5. Synthesize an object-anchored antipodal grasp ``G*`` (``_synthesize_grasp``):
       closing axis from the human thumb-index axis (or object PCA), approach from
       the human approach direction (or top-down), antipodal contacts from the
       object cloud, estimated on a pre-contact (least-occluded) frame.
    6. Integrate everything into a unified per-frame *intent* (``_integrate_intent``):
       EE position/orientation targets ``p_t*`` / ``R_t*`` (object-relative during
       the grasp, hand-following otherwise), gripper command ``g_t``, phase labels
       and phase-dependent cost weights ``w_p`` / ``w_r`` (design §2.4) — the
       ``intent.npz`` schema consumed by the Stage B trajectory optimizer.
    7. Save masks, point clouds, contact events, grasp pose, intent and debug
       visualizations for verification.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from phantom.processors.base_processor import BaseProcessor
from phantom.processors.paths import Paths
from phantom.processors.phantom_data import HandSequence
from phantom.utils.pcd_utils import get_point_cloud_of_segmask
from phantom.utils.transform_utils import transform_pts

logger = logging.getLogger(__name__)

# MediaPipe / HaMeR 21-keypoint convention: wrist=0, thumb tip=4, index tip=8,
# middle tip=12, ring tip=16, pinky tip=20.
FINGERTIP_IDXS = [4, 8, 12]  # thumb / index / middle tips
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8

# Phase codes.
PHASE_FREE, PHASE_GRASP, PHASE_TRANSPORT, PHASE_RELEASE = 0, 1, 2, 3
PHASE_NAMES = {
    PHASE_FREE: "free",
    PHASE_GRASP: "grasp",
    PHASE_TRANSPORT: "transport",
    PHASE_RELEASE: "release",
}


class IntentProcessor(BaseProcessor):
    """Stage A: perception + intent extraction (object seg + point cloud).

    Config keys (all optional, sensible defaults applied):
        object_prompt (str):      fallback object noun for DINO if no objects.json.
        intent_dino_threshold (float): DINO confidence threshold (default 0.25).
        intent_seed_stride (int): stride for the DINO seed search (default 3).
        intent_depth_max (float): max valid depth in meters (default 3.0).
        intent_min_object_pts (int): min points to consider a frame's cloud valid.
    """

    def __init__(self, args):
        super().__init__(args)
        self.dino_threshold = float(getattr(self.cfg, "intent_dino_threshold", 0.25))
        self.seed_stride = int(getattr(self.cfg, "intent_seed_stride", 3))
        self.seed_min_score = float(getattr(self.cfg, "intent_seed_min_score", 0.35))
        self.depth_max = float(getattr(self.cfg, "intent_depth_max", 3.0))
        self.min_object_pts = int(getattr(self.cfg, "intent_min_object_pts", 50))
        self.mask_erode = int(getattr(self.cfg, "intent_mask_erode", 2))
        self.outlier_nb = int(getattr(self.cfg, "intent_outlier_nb", 20))
        self.outlier_std = float(getattr(self.cfg, "intent_outlier_std", 2.0))

        # Contact detection params.
        self.contact_dist_in = float(getattr(self.cfg, "intent_contact_dist_in", 0.03))
        self.contact_dist_out = float(getattr(self.cfg, "intent_contact_dist_out", 0.05))
        self.contact_min_valid = int(getattr(self.cfg, "intent_contact_min_valid", 5))
        self.contact_min_run = int(getattr(self.cfg, "intent_contact_min_run", 3))
        self.contact_motion_thresh = float(getattr(self.cfg, "intent_contact_motion_thresh", 0.004))
        self.grasp_window = int(getattr(self.cfg, "intent_grasp_window", 3))
        self.release_window = int(getattr(self.cfg, "intent_release_window", 3))
        self.cfg_object_prompt = getattr(self.cfg, "object_prompt", None)

        # Grasp synthesis params (block 3).
        self.grasp_precontact_window = int(getattr(self.cfg, "intent_grasp_precontact_window", 8))
        self.grasp_approach_frames = int(getattr(self.cfg, "intent_grasp_approach_frames", 4))
        self.gripper_max_width = float(getattr(self.cfg, "intent_gripper_max_width", 0.08))
        self.grasp_antipodal_pct = float(getattr(self.cfg, "intent_grasp_antipodal_pct", 2.0))

        # Intent integration params (block 4): phase-dependent cost weights (design §2.4).
        self.wp_free = float(getattr(self.cfg, "intent_wp_free", 1.0))
        self.wr_free = float(getattr(self.cfg, "intent_wr_free", 0.1))
        self.wp_grasp = float(getattr(self.cfg, "intent_wp_grasp", 5.0))
        self.wr_grasp = float(getattr(self.cfg, "intent_wr_grasp", 5.0))

        # Lazily built to avoid loading heavy models when not needed.
        self._dino = None
        self._sam2 = None

    # ------------------------------------------------------------------
    # Detector lazy init
    # ------------------------------------------------------------------
    @property
    def dino(self):
        if self._dino is None:
            from phantom.detectors.detector_dino import DetectorDino
            self._dino = DetectorDino("IDEA-Research/grounding-dino-base")
        return self._dino

    @property
    def sam2(self):
        if self._sam2 is None:
            from phantom.detectors.detector_sam2 import DetectorSam2
            self._sam2 = DetectorSam2()
        return self._sam2

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def process_one_demo(self, data_sub_folder: str) -> None:
        save_folder = self.get_save_folder(data_sub_folder)
        paths = self.get_paths(save_folder)

        object_prompt = self._get_object_prompt(paths)
        logger.info("[intent] demo=%s object_prompt=%r", data_sub_folder, object_prompt)

        frames = self._load_frames(paths)  # (T, H, W, 3) RGB uint8
        n_frames = len(frames)
        logger.info("[intent] loaded %d frames of size %s", n_frames, frames.shape[1:3])

        # 1) object detection + mask propagation
        seed_idx, seed_bbox, seed_score = self._detect_seed(frames, object_prompt)
        if seed_idx is None:
            logger.warning(
                "[intent] object %r not detected in any frame (threshold=%.2f); "
                "skipping demo", object_prompt, self.dino_threshold,
            )
            return
        logger.info(
            "[intent] seed frame=%d score=%.3f bbox=%s", seed_idx, seed_score, seed_bbox.astype(int)
        )
        object_masks = self._propagate_mask(paths, seed_idx, seed_bbox, n_frames)

        # 2) object point clouds from depth back-projection
        depth = self._load_depth(paths, n_frames, frames.shape[1:3])
        pcd_result = self._build_object_pointclouds(frames, object_masks, depth)

        # 3) contact detection + phase segmentation
        hands = self._load_hand_fingertips(paths, len(object_masks))
        contact_result = self._detect_contacts(pcd_result, hands)

        # 4) hand->gripper antipodal grasp synthesis (object-anchored G*)
        grasp_result = self._synthesize_grasp(pcd_result, contact_result, hands)

        # 5) intent integration: unified per-frame task-space targets for Stage B
        intent_result = self._integrate_intent(pcd_result, contact_result, grasp_result, hands)

        # 6) save outputs + debug visualizations
        self._save_results(
            paths=paths,
            object_prompt=object_prompt,
            seed_idx=seed_idx,
            seed_score=seed_score,
            object_masks=object_masks,
            frames=frames,
            pcd_result=pcd_result,
            contact_result=contact_result,
            grasp_result=grasp_result,
            intent_result=intent_result,
        )

    # ------------------------------------------------------------------
    # Object prompt resolution
    # ------------------------------------------------------------------
    def _get_object_prompt(self, paths: Paths) -> str:
        """Resolve the object noun to feed Grounding-DINO.

        Priority: per-demo ``objects.json`` (written from EgoDex llm_objects) >
        config ``object_prompt``.
        """
        objects_json = paths.data_path / "objects.json"
        if objects_json.exists():
            try:
                with open(objects_json, "r") as f:
                    data = json.load(f)
                objs = data.get("objects", data) if isinstance(data, dict) else data
                if isinstance(objs, (list, tuple)) and len(objs) > 0:
                    return str(objs[0])
                if isinstance(objs, str) and objs.strip():
                    return objs.strip()
            except Exception as e:  # noqa: BLE001
                logger.warning("[intent] failed to read %s: %s", objects_json, e)

        if self.cfg_object_prompt:
            return str(self.cfg_object_prompt)

        raise ValueError(
            f"No object prompt available for {paths.data_path}. Provide a per-demo "
            f"objects.json or set `object_prompt` in the config."
        )

    # ------------------------------------------------------------------
    # Frame / depth loading
    # ------------------------------------------------------------------
    def _load_frames(self, paths: Paths) -> np.ndarray:
        """Ensure the square-cropped frames exist on disk and return them RGB.

        Frames are extracted with the same ``square`` convention as the rest of
        the pipeline so pixels line up with the (square) depth map.
        """
        folder = paths.original_images_folder
        if not os.path.exists(folder) or len([f for f in os.listdir(folder) if f.endswith(".jpg")]) == 0:
            self._extract_frames_cv2(paths.video_left, folder, square=self.square)

        frame_files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        frames = [cv2.cvtColor(cv2.imread(os.path.join(folder, f)), cv2.COLOR_BGR2RGB) for f in frame_files]
        return np.stack(frames, axis=0)

    @staticmethod
    def _extract_frames_cv2(video_path, folder, square: bool) -> None:
        """Extract video frames to JPGs with cv2 (no ffmpeg dependency)."""
        os.makedirs(folder, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if square and frame_bgr.shape[1] != frame_bgr.shape[0]:
                delta = (frame_bgr.shape[1] - frame_bgr.shape[0]) // 2
                frame_bgr = frame_bgr[:, delta:frame_bgr.shape[1] - delta, :]
            cv2.imwrite(os.path.join(folder, f"{idx:05d}.jpg"), frame_bgr)
            idx += 1
        cap.release()

    def _load_depth(self, paths: Paths, n_frames: int, frame_hw: Tuple[int, int]) -> np.ndarray:
        """Load the metric depth map and align it to the (square) frame size."""
        if not os.path.exists(paths.depth):
            raise FileNotFoundError(
                f"Depth map not found at {paths.depth}. The object point cloud step "
                f"requires an RGB-D demo."
            )
        depth = np.load(paths.depth)
        if depth.ndim == 4:
            depth = depth[..., 0]

        H, W = frame_hw
        # Square-crop depth to match square-cropped frames when needed.
        if self.square and depth.shape[1] != depth.shape[2]:
            delta = (depth.shape[2] - depth.shape[1]) // 2
            depth = depth[:, :, delta:depth.shape[2] - delta]
        if depth.shape[1:] != (H, W):
            depth = np.stack([
                cv2.resize(depth[i], (W, H), interpolation=cv2.INTER_NEAREST) for i in range(len(depth))
            ], axis=0)
        if len(depth) != n_frames:
            logger.warning(
                "[intent] depth frames (%d) != rgb frames (%d); truncating to min",
                len(depth), n_frames,
            )
        return depth

    # ------------------------------------------------------------------
    # Detection + mask propagation
    # ------------------------------------------------------------------
    def _detect_seed(
        self, frames: np.ndarray, object_prompt: str
    ) -> Tuple[Optional[int], Optional[np.ndarray], float]:
        """Pick the SAM2 seed detection.

        Prefer the *earliest* confident detection (>= ``seed_min_score``) rather
        than the global maximum: early frames typically show the object isolated,
        before manipulation/occlusion causes the box to drift onto the hand or
        nearby clutter. Falls back to the global best if nothing clears the floor.
        """
        best_idx: Optional[int] = None
        best_bbox: Optional[np.ndarray] = None
        best_score = -1.0
        early_idx: Optional[int] = None
        early_bbox: Optional[np.ndarray] = None
        early_score = -1.0
        for idx in range(0, len(frames), max(1, self.seed_stride)):
            bboxes, scores = self.dino.get_bboxes(
                frames[idx], object_prompt, threshold=self.dino_threshold
            )
            if len(bboxes) == 0:
                continue
            k = int(np.argmax(scores))
            score = float(scores[k])
            if score > best_score:
                best_score = score
                best_bbox = np.asarray(bboxes[k], dtype=np.float32)
                best_idx = idx
            if early_idx is None and score >= self.seed_min_score:
                early_idx = idx
                early_bbox = np.asarray(bboxes[k], dtype=np.float32)
                early_score = score
        if early_idx is not None:
            return early_idx, early_bbox, early_score
        return best_idx, best_bbox, best_score

    def _propagate_mask(
        self, paths: Paths, seed_idx: int, seed_bbox: np.ndarray, n_frames: int
    ) -> np.ndarray:
        """Propagate the seed detection to every frame with SAM2 (both directions)."""
        cx = float((seed_bbox[0] + seed_bbox[2]) / 2)
        cy = float((seed_bbox[1] + seed_bbox[3]) / 2)
        points = [np.array([[cx, cy]], dtype=np.float32)]
        indices = [int(seed_idx)]

        masks = np.zeros((n_frames,) + self._frame_hw(paths), dtype=np.uint8)

        for reverse in (False, True):
            video_segments, _ = self.sam2.segment_video(
                paths.original_images_folder,
                bbox=seed_bbox,
                points=points,
                indices=indices,
                reverse=reverse,
            )
            for fidx, obj_dict in video_segments.items():
                if fidx >= n_frames:
                    continue
                m = obj_dict.get(0)
                if m is None:
                    continue
                masks[fidx] = np.asarray(m[0], dtype=np.uint8)
        return masks

    def _frame_hw(self, paths: Paths) -> Tuple[int, int]:
        folder = paths.original_images_folder
        frame_files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        img = cv2.imread(os.path.join(folder, frame_files[0]))
        return img.shape[0], img.shape[1]

    # ------------------------------------------------------------------
    # Point cloud back-projection
    # ------------------------------------------------------------------
    def _build_object_pointclouds(
        self, frames: np.ndarray, object_masks: np.ndarray, depth: np.ndarray
    ) -> Dict[str, list]:
        """Back-project each frame's object mask into a metric point cloud."""
        n = min(len(frames), len(object_masks), len(depth))
        points_cam: List[np.ndarray] = []
        points_robot: List[np.ndarray] = []
        colors: List[np.ndarray] = []
        centroids_robot = np.full((n, 3), np.nan, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)

        erode_kernel = (
            np.ones((self.mask_erode * 2 + 1, self.mask_erode * 2 + 1), np.uint8)
            if self.mask_erode > 0 else None
        )
        for i in range(n):
            mask = object_masks[i].astype(np.uint8)
            # Erode the mask to drop boundary pixels that bleed onto the
            # background depth (floor/table) and create stray 3D points.
            if erode_kernel is not None:
                eroded = cv2.erode(mask, erode_kernel, iterations=1)
                if eroded.sum() >= self.min_object_pts:
                    mask = eroded
            mask = mask.astype(bool)
            depth_i = depth[i].astype(np.float32)
            depth_valid = np.isfinite(depth_i) & (depth_i > 1e-3) & (depth_i < self.depth_max)
            mask = mask & depth_valid
            if mask.sum() < self.min_object_pts:
                points_cam.append(np.empty((0, 3), np.float32))
                points_robot.append(np.empty((0, 3), np.float32))
                colors.append(np.empty((0, 3), np.float32))
                continue

            pcd = get_point_cloud_of_segmask(mask, depth_i, frames[i], self.intrinsics_dict)
            # Drop sparse depth-noise / mask-edge points far from the object body.
            if self.outlier_nb > 0 and len(pcd.points) > self.outlier_nb:
                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=self.outlier_nb, std_ratio=self.outlier_std
                )
            pts_cam = np.asarray(pcd.points, dtype=np.float32)
            cols = np.asarray(pcd.colors, dtype=np.float32)
            pts_rf = transform_pts(pts_cam, self.T_cam2robot).astype(np.float32)

            points_cam.append(pts_cam)
            points_robot.append(pts_rf)
            colors.append(cols)
            centroids_robot[i] = pts_rf.mean(axis=0)
            valid[i] = True

        logger.info(
            "[intent] built object point clouds: %d/%d frames valid", int(valid.sum()), n
        )
        return {
            "points_cam": points_cam,
            "points_robot": points_robot,
            "colors": colors,
            "centroids_robot": centroids_robot,
            "valid": valid,
        }

    # ------------------------------------------------------------------
    # Saving + visualization
    # ------------------------------------------------------------------
    def _save_results(
        self,
        paths: Paths,
        object_prompt: str,
        seed_idx: int,
        seed_score: float,
        object_masks: np.ndarray,
        frames: np.ndarray,
        pcd_result: Dict[str, list],
        contact_result: Optional[Dict[str, np.ndarray]] = None,
        grasp_result: Optional[Dict[str, object]] = None,
        intent_result: Optional[Dict[str, object]] = None,
    ) -> None:
        os.makedirs(paths.intent_processor, exist_ok=True)

        np.save(paths.object_masks, object_masks)

        np.savez(
            paths.object_pcd,
            points_cam=np.array(pcd_result["points_cam"], dtype=object),
            points_robot=np.array(pcd_result["points_robot"], dtype=object),
            colors=np.array(pcd_result["colors"], dtype=object),
            centroids_robot=pcd_result["centroids_robot"],
            valid=pcd_result["valid"],
            object_prompt=object_prompt,
            seed_idx=seed_idx,
            seed_score=seed_score,
            intrinsics=json.dumps(self.intrinsics_dict),
            T_cam2robot=self.T_cam2robot,
        )

        if contact_result is not None:
            np.savez(
                paths.contact_events,
                phase=contact_result["phase"],
                phase_names=contact_result["phase_names"],
                g_closed=contact_result["g_closed"],
                contact=contact_result["contact"],
                d_finger_obj=contact_result["d_finger_obj"],
                aperture=contact_result["aperture"],
                obj_speed=contact_result["obj_speed"],
                grasp_keyframe=contact_result["grasp_keyframe"],
                release_keyframe=contact_result["release_keyframe"],
                source=contact_result["source"],
            )
            self._save_contact_diagnostic(paths, contact_result)

        if grasp_result is not None and grasp_result.get("valid", False):
            np.savez(
                paths.grasp,
                grasp_frame=grasp_result["grasp_frame"],
                grasp_keyframe=grasp_result["grasp_keyframe"],
                G_center=grasp_result["G_center"],
                G_rot=grasp_result["G_rot"],
                G_rot_pipeline=grasp_result["G_rot_pipeline"],
                G_width=grasp_result["G_width"],
                closing_axis=grasp_result["closing_axis"],
                approach_axis=grasp_result["approach_axis"],
                contact_points=grasp_result["contact_points"],
                source=grasp_result["source"],
            )
            self._save_grasp_preview(paths, pcd_result, grasp_result)

        if intent_result is not None:
            self._save_intent(paths, intent_result)
            self._save_intent_preview(paths, intent_result)

        self._save_mask_overlay_video(paths, frames, object_masks)
        self._save_pcd_preview(paths, pcd_result, seed_idx)
        logger.info("[intent] saved results to %s", paths.intent_processor)

    def _save_intent(self, paths: Paths, intent_result: Dict[str, object]) -> None:
        """Persist the unified intent (Stage B consumes this)."""
        np.savez(
            paths.intent,
            p_target=intent_result["p_target"],
            R_target=intent_result["R_target"],
            p_valid=intent_result["p_valid"],
            phase=intent_result["phase"],
            phase_names=intent_result["phase_names"],
            g_closed=intent_result["g_closed"],
            gripper_width=intent_result["gripper_width"],
            w_p=intent_result["w_p"],
            w_r=intent_result["w_r"],
            object_centroid=intent_result["object_centroid"],
            p_source=intent_result["p_source"],
            grasp_frame=intent_result["grasp_frame"],
            grasp_keyframe=intent_result["grasp_keyframe"],
            release_keyframe=intent_result["release_keyframe"],
            G_center=intent_result["G_center"],
            G_rot=intent_result["G_rot"],
            G_width=intent_result["G_width"],
            gripper_open_width=intent_result["gripper_open_width"],
            grasp_valid=intent_result["grasp_valid"],
        )

    def _save_contact_diagnostic(self, paths: Paths, contact_result: Dict[str, np.ndarray]) -> None:
        """Plot the contact signals with phase shading + grasp/release lines."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            phase = contact_result["phase"]
            n = len(phase)
            x = np.arange(n)
            fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

            phase_colors = {
                PHASE_FREE: "#eeeeee", PHASE_GRASP: "#ffd27f",
                PHASE_TRANSPORT: "#9ecae1", PHASE_RELEASE: "#fca5a5",
            }
            for ax in axes:
                for p, color in phase_colors.items():
                    ax.fill_between(x, 0, 1, where=(phase == p), transform=ax.get_xaxis_transform(),
                                    color=color, alpha=0.6, step="mid", linewidth=0)

            ax0 = axes[0]
            d = contact_result["d_finger_obj"]
            if np.isfinite(d).any():
                ax0.plot(x, d, color="k", lw=1.2, label="fingertip-object dist (m)")
                ax0.axhline(self.contact_dist_in, color="g", ls="--", lw=0.8, label="dist_in")
                ax0.axhline(self.contact_dist_out, color="r", ls="--", lw=0.8, label="dist_out")
            ap = contact_result["aperture"]
            if np.isfinite(ap).any():
                ax0.plot(x, ap, color="purple", lw=1.0, alpha=0.7, label="grasp aperture (m)")
            ax0.set_ylabel("distance (m)")
            if ax0.get_legend_handles_labels()[1]:
                ax0.legend(loc="upper right", fontsize=8)
            ax0.set_title(f"contact detection (source={contact_result['source']})")

            ax1 = axes[1]
            ax1.plot(x, contact_result["obj_speed"], color="teal", lw=1.0, label="object speed (m/frame)")
            ax1.axhline(self.contact_motion_thresh, color="orange", ls="--", lw=0.8, label="motion_thresh")
            ax1.plot(x, contact_result["g_closed"].astype(float) * (np.nanmax(contact_result["obj_speed"]) or 0.01),
                     color="k", lw=0.8, alpha=0.5, label="gripper closed")
            ax1.set_ylabel("speed (m/frame)")
            ax1.set_xlabel("frame")
            ax1.legend(loc="upper right", fontsize=8)

            gk = int(contact_result["grasp_keyframe"]); rk = int(contact_result["release_keyframe"])
            for ax in axes:
                if gk >= 0:
                    ax.axvline(gk, color="darkgreen", lw=1.5)
                if rk >= 0:
                    ax.axvline(rk, color="darkred", lw=1.5)

            fig.tight_layout()
            fig.savefig(str(paths.contact_diagnostic), dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            logger.warning("[intent] failed to write contact diagnostic: %s", e)

    def _save_mask_overlay_video(
        self, paths: Paths, frames: np.ndarray, object_masks: np.ndarray
    ) -> None:
        n = min(len(frames), len(object_masks))
        red = np.array([255, 0, 0], dtype=np.uint8)
        H, W = frames.shape[1:3]
        writer = cv2.VideoWriter(
            str(paths.video_object_mask), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H)
        )
        ok_writer = writer.isOpened()
        for i in range(n):
            frame = frames[i].copy()
            m = object_masks[i].astype(bool)
            if m.any():
                frame[m] = (0.5 * frame[m] + 0.5 * red).astype(np.uint8)
            if ok_writer:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        if not ok_writer:
            logger.warning("[intent] cv2.VideoWriter unavailable; saving sample overlay frames")
            for i in np.linspace(0, n - 1, num=min(n, 6)).astype(int):
                frame = frames[i].copy()
                m = object_masks[i].astype(bool)
                if m.any():
                    frame[m] = (0.5 * frame[m] + 0.5 * red).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(paths.intent_processor, f"overlay_{i:05d}.jpg"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )

    def _save_pcd_preview(
        self, paths: Paths, pcd_result: Dict[str, list], seed_idx: int
    ) -> None:
        """Render a quick 3D scatter of the seed frame's object cloud."""
        valid = pcd_result["valid"]
        if not valid.any():
            return
        idx = seed_idx if seed_idx < len(valid) and valid[seed_idx] else int(np.argmax(valid))
        pts = pcd_result["points_robot"][idx]
        cols = pcd_result["colors"][idx]
        if len(pts) == 0:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=2)
            ax.set_title(f"object cloud (robot frame), frame {idx}, {len(pts)} pts")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            fig.savefig(str(paths.object_pcd_preview), dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            logger.warning("[intent] failed to write pcd preview: %s", e)

    def _save_grasp_preview(
        self, paths: Paths, pcd_result: Dict[str, list], grasp_result: Dict[str, object]
    ) -> None:
        """3D preview of the object cloud with the synthesized grasp frame + jaws."""
        gf = int(grasp_result["grasp_frame"])
        pts = np.asarray(pcd_result["points_robot"][gf], dtype=np.float32)
        cols = np.asarray(pcd_result["colors"][gf], dtype=np.float32)
        if len(pts) == 0:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            center = np.asarray(grasp_result["G_center"], dtype=np.float32)
            R = np.asarray(grasp_result["G_rot"], dtype=np.float32)
            width = float(grasp_result["G_width"])
            cps = np.asarray(grasp_result["contact_points"], dtype=np.float32)
            scale = max(width, 0.03)

            fig = plt.figure(figsize=(6.5, 6.5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=2, alpha=0.5)
            # Antipodal contacts + closing line (the parallel-jaw bite).
            ax.scatter(cps[:, 0], cps[:, 1], cps[:, 2], c="red", s=60, marker="X")
            ax.plot(cps[:, 0], cps[:, 1], cps[:, 2], c="red", lw=1.5)
            # Gripper frame axes: x=closing (red), y (green), z=approach (blue).
            for k, color in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
                v = R[:, k] * scale
                ax.quiver(center[0], center[1], center[2], v[0], v[1], v[2],
                          color=color, linewidth=2)
            ax.set_title(
                f"grasp G* frame {gf} ({grasp_result['source']}), width {width*100:.1f}cm"
            )
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            fig.savefig(str(paths.grasp_preview), dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            logger.warning("[intent] failed to write grasp preview: %s", e)

    # ------------------------------------------------------------------
    # Contact detection + phase segmentation
    # ------------------------------------------------------------------
    def _detect_contacts(
        self, pcd_result: Dict[str, list], hands: Dict[str, dict]
    ) -> Dict[str, np.ndarray]:
        """Detect grasp/release and segment free/grasp/transport/release phases.

        Primary cue (per design): min distance from the fingertips (thumb/index/
        middle tips) to the object point-cloud surface. When hand keypoints are
        unavailable, falls back to an object-motion cue (the object only moves
        while grasped), so the phase FSM still produces a usable segmentation.
        """
        n = len(pcd_result["valid"])
        points_robot = pcd_result["points_robot"]
        centroids = pcd_result["centroids_robot"]
        obj_valid = np.asarray(pcd_result["valid"], dtype=bool)

        # Per-frame signals.
        d_finger_obj = np.full(n, np.nan, dtype=np.float32)
        aperture = np.full(n, np.nan, dtype=np.float32)
        for t in range(n):
            if not obj_valid[t] or len(points_robot[t]) == 0:
                continue
            obj_pts = points_robot[t]
            best_d, best_ap = np.inf, np.nan
            for h in hands.values():
                if not h["detected"][t]:
                    continue
                fps = h["fingertips"][t]  # (len(FINGERTIP_IDXS), 3)
                dmin = min(
                    float(np.linalg.norm(obj_pts - fp, axis=1).min()) for fp in fps
                )
                if dmin < best_d:
                    best_d = dmin
                    best_ap = float(h["aperture"][t])
            if np.isfinite(best_d):
                d_finger_obj[t] = best_d
                aperture[t] = best_ap

        # Object centroid speed (m/frame).
        obj_speed = np.full(n, np.nan, dtype=np.float32)
        for t in range(1, n):
            if obj_valid[t] and obj_valid[t - 1]:
                obj_speed[t] = float(np.linalg.norm(centroids[t] - centroids[t - 1]))

        have_hands = int(np.isfinite(d_finger_obj).sum()) >= self.contact_min_valid
        if have_hands:
            contact = self._hysteresis_contact(
                d_finger_obj, self.contact_dist_in, self.contact_dist_out
            )
            source = "fingertip"
        else:
            contact = self._motion_contact(obj_speed, self.contact_motion_thresh)
            source = "object_motion"
            logger.warning(
                "[intent] no hand keypoints available (%s missing); using "
                "object-motion contact cue", "hand_data_*",
            )
        contact = self._filter_min_run(contact, self.contact_min_run)

        phase, g_state, grasp_kf, release_kf = self._segment_phases(contact)

        logger.info(
            "[intent] contact source=%s grasp_kf=%s release_kf=%s contact_frames=%d/%d",
            source, grasp_kf, release_kf, int(contact.sum()), n,
        )
        return {
            "phase": phase,
            "phase_names": np.array([PHASE_NAMES[int(p)] for p in phase], dtype=object),
            "g_closed": g_state,
            "contact": contact,
            "d_finger_obj": d_finger_obj,
            "aperture": aperture,
            "obj_speed": obj_speed,
            "grasp_keyframe": np.int64(grasp_kf if grasp_kf is not None else -1),
            "release_keyframe": np.int64(release_kf if release_kf is not None else -1),
            "source": source,
        }

    def _load_hand_fingertips(self, paths: Paths, n: int) -> Dict[str, dict]:
        """Load per-hand fingertip trajectories (robot frame) if available."""
        hands: Dict[str, dict] = {}
        for side in ("left", "right"):
            hand_path = getattr(paths, f"hand_data_{side}", None)
            if hand_path is None or not os.path.exists(hand_path):
                continue
            seq = HandSequence.load(hand_path)
            kpts_cam = seq.kpts_3d  # (M, 21, 3), camera frame
            detected = np.asarray(seq.hand_detected, dtype=bool)
            kpts_rf = self._to_robot_frame(kpts_cam)
            m = min(n, len(kpts_rf))

            fingertips = np.zeros((n, len(FINGERTIP_IDXS), 3), dtype=np.float32)
            aperture = np.full(n, np.nan, dtype=np.float32)
            det = np.zeros(n, dtype=bool)
            fingertips[:m] = kpts_rf[:m][:, FINGERTIP_IDXS, :]
            det[:m] = detected[:m]
            aperture[:m] = np.linalg.norm(
                kpts_rf[:m, THUMB_TIP_IDX] - kpts_rf[:m, INDEX_TIP_IDX], axis=1
            )
            hands[side] = {"fingertips": fingertips, "detected": det, "aperture": aperture}
            logger.info("[intent] loaded %s hand: %d/%d detected", side, int(det.sum()), n)
        return hands

    def _to_robot_frame(self, kpts_cam: np.ndarray) -> np.ndarray:
        """Transform (N,21,3) camera-frame keypoints to robot frame."""
        pts_h = np.concatenate(
            [kpts_cam, np.ones((*kpts_cam.shape[:2], 1))], axis=-1
        )
        return np.einsum("ij,bpj->bpi", self.T_cam2robot, pts_h)[..., :3]

    @staticmethod
    def _hysteresis_contact(dist: np.ndarray, thr_in: float, thr_out: float) -> np.ndarray:
        """Two-threshold contact detection on a distance signal."""
        n = len(dist)
        contact = np.zeros(n, dtype=bool)
        state = False
        for t in range(n):
            d = dist[t]
            if not np.isfinite(d):
                contact[t] = state
                continue
            if not state and d < thr_in:
                state = True
            elif state and d > thr_out:
                state = False
            contact[t] = state
        return contact

    @staticmethod
    def _motion_contact(speed: np.ndarray, thresh: float) -> np.ndarray:
        """Fallback contact: object is moving => grasped. Bridge brief pauses."""
        moving = np.isfinite(speed) & (speed > thresh)
        contact = moving.copy()
        idx = np.where(moving)[0]
        if len(idx) > 0:
            contact[idx[0]:idx[-1] + 1] = True  # bridge still moments within transport
        return contact

    @staticmethod
    def _filter_min_run(contact: np.ndarray, min_run: int) -> np.ndarray:
        """Remove contact runs shorter than min_run frames."""
        if min_run <= 1:
            return contact
        out = contact.copy()
        n = len(contact)
        t = 0
        while t < n:
            if out[t]:
                s = t
                while t < n and out[t]:
                    t += 1
                if (t - s) < min_run:
                    out[s:t] = False
            else:
                t += 1
        return out

    def _segment_phases(self, contact: np.ndarray):
        """Map a contact boolean to free/grasp/transport/release phases.

        Uses the longest contact run as the manipulation episode.
        """
        n = len(contact)
        phase = np.full(n, PHASE_FREE, dtype=np.int64)
        g_state = np.zeros(n, dtype=bool)

        # Find contact runs; pick the longest.
        runs = []
        t = 0
        while t < n:
            if contact[t]:
                s = t
                while t < n and contact[t]:
                    t += 1
                runs.append((s, t))  # [s, t)
            else:
                t += 1
        if not runs:
            return phase, g_state, None, None

        c0, c1 = max(runs, key=lambda r: r[1] - r[0])  # [c0, c1)
        grasp_kf = c0
        release_kf = c1 - 1

        g_state[c0:c1] = True

        grasp_end = min(c0 + self.grasp_window, c1)
        release_start = max(c1 - self.release_window, grasp_end)
        phase[c0:grasp_end] = PHASE_GRASP
        phase[grasp_end:release_start] = PHASE_TRANSPORT
        phase[release_start:c1] = PHASE_RELEASE
        return phase, g_state, grasp_kf, release_kf

    # ------------------------------------------------------------------
    # Hand -> gripper antipodal grasp synthesis (block 3)
    # ------------------------------------------------------------------
    def _synthesize_grasp(
        self,
        pcd_result: Dict[str, list],
        contact_result: Dict[str, np.ndarray],
        hands: Dict[str, dict],
    ) -> Dict[str, object]:
        """Synthesize an object-anchored antipodal grasp pose ``G*``.

        Per design (§3.4 / progress §5.1): at the grasp onset we do *not* replicate
        the five fingers. Instead we ground a parallel-jaw grasp in the object
        point cloud:

          * **closing axis** aligned with the human thumb-index axis (or, with no
            hand, the object's narrow principal axis);
          * **approach axis** aligned with the human approach direction (hand
            motion into the object; or top-down when no hand);
          * **antipodal contacts** = the extreme object points along the closing
            axis → grasp center + opening width grounded in real geometry.

        Occlusion mitigation (Risk #1): geometry is estimated from a *pre-contact*
        (least-occluded) frame near the grasp keyframe, then locked.

        Returns a dict (``valid=False`` when no grasp keyframe exists).
        """
        n = len(pcd_result["valid"])
        grasp_kf = int(contact_result["grasp_keyframe"])
        invalid = {
            "valid": False,
            "grasp_frame": np.int64(-1),
            "grasp_keyframe": np.int64(grasp_kf),
            "G_center": np.full(3, np.nan, np.float32),
            "G_rot": np.full((3, 3), np.nan, np.float32),
            "G_rot_pipeline": np.full((3, 3), np.nan, np.float32),
            "G_width": np.float32(np.nan),
            "closing_axis": np.full(3, np.nan, np.float32),
            "approach_axis": np.full(3, np.nan, np.float32),
            "contact_points": np.full((2, 3), np.nan, np.float32),
            "source": "none",
        }
        if grasp_kf < 0:
            logger.warning("[intent] no grasp keyframe; skipping grasp synthesis")
            return invalid

        # 1) pick the least-occluded pre-contact frame for stable geometry.
        gf = self._select_grasp_frame(pcd_result, grasp_kf)
        if gf is None:
            logger.warning("[intent] no valid object cloud near grasp keyframe %d", grasp_kf)
            return invalid
        P = np.asarray(pcd_result["points_robot"][gf], dtype=np.float32)
        obj_center = P.mean(axis=0)

        # 2) closing + approach axes: prefer human hand, else object PCA + top-down.
        axes = self._grasp_axes_from_hand(hands, gf, obj_center)
        if axes is not None:
            closing0, approach, source = axes
        else:
            closing0, approach = self._grasp_axes_from_pca(P)
            source = "object_pca"

        # 3) antipodal contacts along the closing axis ground the grasp in geometry.
        # The closing axis stays the intended jaw-travel direction; the width is
        # the object's extent projected onto it (contacts are the real extremes).
        closing = closing0
        p_neg, p_pos, width, grasp_center = self._antipodal_from_axis(
            P, obj_center, closing, self.grasp_antipodal_pct
        )

        if width > self.gripper_max_width:
            logger.warning(
                "[intent] antipodal width %.3fm exceeds gripper max %.3fm; "
                "object may be too wide along the closing axis",
                width, self.gripper_max_width,
            )

        # 4) assemble the gripper frame (x=closing/opening, z=approach).
        R = self._build_grasp_rotation(approach, closing)
        # Match the pipeline convention (HandModel: gripper_ori @ Rz(90deg)).
        from scipy.spatial.transform import Rotation
        rot_90 = Rotation.from_euler("Z", 90, degrees=True).as_matrix()
        R_pipeline = R @ rot_90

        logger.info(
            "[intent] grasp G*: frame=%d (kf=%d) source=%s center=%s width=%.3fm",
            gf, grasp_kf, source, np.round(grasp_center, 3).tolist(), float(width),
        )
        return {
            "valid": True,
            "grasp_frame": np.int64(gf),
            "grasp_keyframe": np.int64(grasp_kf),
            "G_center": grasp_center.astype(np.float32),
            "G_rot": R.astype(np.float32),
            "G_rot_pipeline": R_pipeline.astype(np.float32),
            "G_width": np.float32(width),
            "closing_axis": closing.astype(np.float32),
            "approach_axis": approach.astype(np.float32),
            "contact_points": np.stack([p_neg, p_pos]).astype(np.float32),
            "source": source,
        }

    def _select_grasp_frame(self, pcd_result: Dict[str, list], grasp_kf: int) -> Optional[int]:
        """Pick a pre-contact frame with the most object points (least occluded).

        Searches ``[grasp_kf - window, grasp_kf]`` so the geometry comes from
        before the hand occludes the object; falls back to the grasp keyframe or
        the globally densest valid cloud.
        """
        valid = np.asarray(pcd_result["valid"], dtype=bool)
        points = pcd_result["points_robot"]
        lo = max(0, grasp_kf - self.grasp_precontact_window)
        candidates = [t for t in range(lo, grasp_kf + 1) if valid[t] and len(points[t]) > 0]
        if candidates:
            return max(candidates, key=lambda t: len(points[t]))
        if valid[grasp_kf] and len(points[grasp_kf]) > 0:
            return grasp_kf
        all_valid = [t for t in range(len(valid)) if valid[t] and len(points[t]) > 0]
        if not all_valid:
            return None
        return max(all_valid, key=lambda t: len(points[t]))

    def _grasp_axes_from_hand(
        self, hands: Dict[str, dict], gf: int, obj_center: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        """Closing (thumb-index) + approach (hand motion) axes from the human hand.

        Returns ``None`` when no hand is detected at/near the grasp frame.
        """
        if not hands:
            return None
        # Pick the hand whose fingertips are closest to the object at gf.
        best_side, best_d = None, np.inf
        for side, h in hands.items():
            if gf < len(h["detected"]) and h["detected"][gf]:
                d = float(np.linalg.norm(h["fingertips"][gf].mean(axis=0) - obj_center))
                if d < best_d:
                    best_d, best_side = d, side
        if best_side is None:
            return None
        h = hands[best_side]
        fps = h["fingertips"][gf]  # (thumb, index, middle) tips
        closing = fps[0] - fps[1]  # thumb - index
        cnorm = np.linalg.norm(closing)
        if cnorm < 1e-6:
            return None
        closing = closing / cnorm

        # Approach = direction the hand travels into the object over the frames
        # leading up to the grasp; fall back to (object - hand) direction.
        cur = fps.mean(axis=0)
        prev_f = None
        for k in range(1, self.grasp_approach_frames + 1):
            j = gf - k
            if j >= 0 and h["detected"][j]:
                prev_f = h["fingertips"][j].mean(axis=0)
        approach = cur - prev_f if prev_f is not None else obj_center - cur
        anorm = np.linalg.norm(approach)
        approach = approach / anorm if anorm > 1e-6 else obj_center - cur
        anorm = np.linalg.norm(approach)
        if anorm < 1e-6:
            approach = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            approach = approach / anorm
        return closing.astype(np.float32), approach.astype(np.float32), "hand_anchored"

    def _grasp_axes_from_pca(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback axes from object geometry: top-down approach, narrow closing.

        Approach is top-down (robot -Z); of the two principal axes orthogonal to
        the approach, the closing axis is the one with the smaller object extent
        (a parallel jaw closes across the object's narrow dimension).
        """
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        Pc = P - P.mean(axis=0)
        cov = (Pc.T @ Pc) / max(len(Pc), 1)
        _, evecs = np.linalg.eigh(cov)  # columns = eigenvectors, ascending
        axes = evecs.T
        # Drop the axis most aligned with the approach direction.
        align = np.abs(axes @ approach)
        keep = [i for i in range(3) if i != int(np.argmax(align))]
        # Of the remaining two, closing = smaller object extent.
        def extent(ax):
            proj = Pc @ ax
            return float(proj.max() - proj.min())
        closing_ax = min(keep, key=lambda i: extent(axes[i]))
        closing = axes[closing_ax]
        return closing.astype(np.float32), approach

    @staticmethod
    def _antipodal_from_axis(
        P: np.ndarray, center: np.ndarray, axis: np.ndarray, pct: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Antipodal contacts = robust extreme object points along the closing axis.

        Uses the ``pct``/``100-pct`` percentiles of the projection rather than the
        absolute min/max so that a few stray outlier points can't inflate the
        grasp width; the contact points are the real object points nearest those
        percentile projections.
        """
        a = axis / max(np.linalg.norm(axis), 1e-9)
        proj = (P - center) @ a
        pct = float(np.clip(pct, 0.0, 49.0))
        lo, hi = np.percentile(proj, [pct, 100.0 - pct])
        i_min = int(np.argmin(np.abs(proj - lo)))
        i_max = int(np.argmin(np.abs(proj - hi)))
        p_neg, p_pos = P[i_min], P[i_max]
        width = float(hi - lo)
        grasp_center = 0.5 * (p_neg + p_pos)
        return p_neg.astype(np.float32), p_pos.astype(np.float32), width, grasp_center.astype(np.float32)

    @staticmethod
    def _build_grasp_rotation(approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
        """Right-handed gripper frame: x=closing (opening dir), z=approach."""
        z = approach / max(np.linalg.norm(approach), 1e-9)
        x = closing - (closing @ z) * z  # orthogonalize opening dir against approach
        if np.linalg.norm(x) < 1e-6:
            # Degenerate (closing || approach): pick any axis orthogonal to z.
            tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = tmp - (tmp @ z) * z
        x = x / max(np.linalg.norm(x), 1e-9)
        y = np.cross(z, x)
        y = y / max(np.linalg.norm(y), 1e-9)
        R = np.column_stack([x, y, z])
        if np.linalg.det(R) < 0:
            x = -x
            R = np.column_stack([x, y, z])
        return R

    # ------------------------------------------------------------------
    # Intent integration (block 4): unified per-frame targets for Stage B
    # ------------------------------------------------------------------
    def _integrate_intent(
        self,
        pcd_result: Dict[str, list],
        contact_result: Dict[str, np.ndarray],
        grasp_result: Dict[str, object],
        hands: Dict[str, dict],
    ) -> Dict[str, object]:
        """Fuse perception outputs into the unified per-frame *intent*.

        This is the Stage A -> Stage B hand-off (design §2.2/§2.3). For each frame
        it emits the EE targets the trajectory optimizer approximates:

          * ``p_target`` — EE position target. During the grasp episode
            ``[grasp_kf, release_kf]`` it is expressed **object-relative**: the
            grasp offset ``G_center - centroid[grasp_frame]`` is locked at grasp
            onset and re-applied to each frame's object centroid, so the EE carries
            the object even where the human hand is occluded. Outside the episode
            it follows the human hand (thumb-index midpoint) when available, else
            holds the nearest defined target.
          * ``R_target`` — EE orientation target. Held at the object-anchored grasp
            rotation ``G_rot`` throughout (rigid parallel-jaw grasp; object motion
            is tracked translation-only — a v1 assumption, see progress §5.2).
          * ``g_closed`` / ``gripper_width`` — discrete gripper command from the
            contact FSM (open = ``gripper_max_width``, closed = grasp width ``G*``).
          * ``w_p`` / ``w_r`` — phase-dependent position/orientation cost weights
            (design §2.4): fidelity budget concentrated on grasp/release.

        Degrades gracefully when no valid grasp exists: falls back to hand-follow /
        centroid targets with identity orientation and uniform (free) weights.
        """
        n = len(pcd_result["valid"])
        centroids = np.asarray(pcd_result["centroids_robot"], dtype=np.float32)  # (n,3)
        obj_valid = np.asarray(pcd_result["valid"], dtype=bool)
        phase = np.asarray(contact_result["phase"])
        g_closed = np.asarray(contact_result["g_closed"], dtype=bool)
        grasp_kf = int(contact_result["grasp_keyframe"])
        release_kf = int(contact_result["release_keyframe"])

        grasp_valid = bool(grasp_result.get("valid", False))
        if grasp_valid:
            G_center = np.asarray(grasp_result["G_center"], dtype=np.float32)
            G_rot = np.asarray(grasp_result["G_rot"], dtype=np.float32)
            G_width = float(grasp_result["G_width"])
            grasp_frame = int(grasp_result["grasp_frame"])
        else:
            G_center = np.full(3, np.nan, np.float32)
            G_rot = np.eye(3, dtype=np.float32)
            G_width = float(self.gripper_max_width)
            grasp_frame = -1

        p_target = np.full((n, 3), np.nan, dtype=np.float32)
        R_target = np.tile(G_rot, (n, 1, 1)).astype(np.float32)
        p_valid = np.zeros(n, dtype=bool)
        p_source = np.empty(n, dtype=object)

        # Object-relative offset locked at the grasp frame (translation tracking).
        offset = None
        if grasp_valid and 0 <= grasp_frame < n and obj_valid[grasp_frame]:
            offset = G_center - centroids[grasp_frame]

        ep_lo = grasp_kf if grasp_kf >= 0 else n
        ep_hi = release_kf if release_kf >= 0 else -1
        for t in range(n):
            in_episode = grasp_valid and offset is not None and ep_lo <= t <= ep_hi
            if in_episode and obj_valid[t]:
                p_target[t] = centroids[t] + offset
                p_valid[t] = True
                p_source[t] = "object_relative"
                continue
            # Outside the grasp episode (or missing object): follow the human hand.
            hand_pt = self._ee_target_from_hand(hands, t)
            if hand_pt is not None:
                p_target[t] = hand_pt
                p_valid[t] = True
                p_source[t] = "hand"
            elif grasp_valid:
                p_target[t] = G_center  # hold the grasp pose as a static target
                p_valid[t] = True
                p_source[t] = "hold_grasp"
            elif obj_valid[t]:
                p_target[t] = centroids[t]
                p_valid[t] = True
                p_source[t] = "object_centroid"
            else:
                p_source[t] = "none"

        # Fill any remaining undefined targets by nearest-valid hold (temporal).
        p_target, p_valid = self._fill_targets(p_target, p_valid)

        # Gripper command + phase-dependent cost weights (design §2.4).
        gripper_width = np.where(g_closed, np.float32(G_width), np.float32(self.gripper_max_width))
        is_key = np.isin(phase, [PHASE_GRASP, PHASE_RELEASE])
        w_p = np.where(is_key, np.float32(self.wp_grasp), np.float32(self.wp_free)).astype(np.float32)
        w_r = np.where(is_key, np.float32(self.wr_grasp), np.float32(self.wr_free)).astype(np.float32)
        if not grasp_valid:
            # No object-anchored orientation -> don't over-trust orientation targets.
            w_r[:] = np.float32(self.wr_free)

        logger.info(
            "[intent] integrated intent: %d/%d frames with EE target, grasp_valid=%s "
            "episode=[%s,%s]",
            int(p_valid.sum()), n, grasp_valid, grasp_kf, release_kf,
        )
        return {
            "p_target": p_target,
            "R_target": R_target,
            "p_valid": p_valid,
            "p_source": p_source,
            "phase": phase,
            "phase_names": np.asarray(contact_result["phase_names"], dtype=object),
            "g_closed": g_closed,
            "gripper_width": gripper_width.astype(np.float32),
            "w_p": w_p,
            "w_r": w_r,
            "object_centroid": centroids,
            "grasp_frame": np.int64(grasp_frame),
            "grasp_keyframe": np.int64(grasp_kf),
            "release_keyframe": np.int64(release_kf),
            "G_center": G_center,
            "G_rot": G_rot,
            "G_width": np.float32(G_width),
            "gripper_open_width": np.float32(self.gripper_max_width),
            "grasp_valid": bool(grasp_valid),
        }

    @staticmethod
    def _ee_target_from_hand(hands: Dict[str, dict], t: int) -> Optional[np.ndarray]:
        """EE position proxy from the human hand: thumb-index midpoint at frame t.

        Picks the detected hand with the smaller aperture (the one actively pinching).
        Returns ``None`` when no hand is detected at ``t``.
        """
        best_pt, best_ap = None, np.inf
        for h in hands.values():
            det = h["detected"]
            if t >= len(det) or not det[t]:
                continue
            fps = h["fingertips"][t]  # (thumb, index, middle) tips
            mid = 0.5 * (fps[0] + fps[1])  # thumb-index midpoint ~ gripper center
            ap = float(h["aperture"][t]) if np.isfinite(h["aperture"][t]) else 0.0
            if ap < best_ap:
                best_ap, best_pt = ap, mid.astype(np.float32)
        return best_pt

    @staticmethod
    def _fill_targets(p_target: np.ndarray, p_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fill undefined EE targets by holding the nearest valid frame in time.

        Keeps the target trajectory gap-free for Stage B without inventing motion:
        leading gaps hold the first valid target, trailing gaps hold the last, and
        interior gaps hold the previous valid target.
        """
        n = len(p_valid)
        if not p_valid.any():
            return p_target, p_valid
        valid_idx = np.where(p_valid)[0]
        first, last = valid_idx[0], valid_idx[-1]
        out = p_target.copy()
        for t in range(first):
            out[t] = p_target[first]
        last_seen = p_target[first]
        for t in range(first, n):
            if p_valid[t]:
                last_seen = p_target[t]
            else:
                out[t] = last_seen
        filled_valid = np.ones(n, dtype=bool)
        return out.astype(np.float32), filled_valid

    def _save_intent_preview(self, paths: Paths, intent_result: Dict[str, object]) -> None:
        """Plot the intent EE-target trajectory (x/y/z) with phase shading + gripper."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover - viz is best-effort
            logger.warning("[intent] matplotlib unavailable, skipping intent preview: %s", e)
            return
        p = np.asarray(intent_result["p_target"], dtype=np.float32)
        phase = np.asarray(intent_result["phase"])
        g_closed = np.asarray(intent_result["g_closed"], dtype=bool)
        n = len(p)
        x = np.arange(n)
        fig, ax = plt.subplots(figsize=(11, 4))
        for label, c in (("x", "tab:red"), ("y", "tab:green"), ("z", "tab:blue")):
            ax.plot(x, p[:, "xyz".index(label)], color=c, lw=1.2, label=f"p_target.{label} (m)")
        # Phase shading.
        shade = {PHASE_GRASP: ("orange", 0.18), PHASE_TRANSPORT: ("gray", 0.10),
                 PHASE_RELEASE: ("purple", 0.18)}
        for ph, (col, a) in shade.items():
            seg = phase == ph
            t = 0
            while t < n:
                if seg[t]:
                    s = t
                    while t < n and seg[t]:
                        t += 1
                    ax.axvspan(s, t - 1, color=col, alpha=a)
                else:
                    t += 1
        ax.plot(x, g_closed.astype(float) * (np.nanmax(p) if np.isfinite(p).any() else 1.0) * 0.1,
                color="black", lw=0.8, alpha=0.6, label="gripper closed")
        ax.set_xlabel("frame")
        ax.set_ylabel("robot-frame target (m)")
        ax.set_title(
            f"intent EE target — grasp_valid={intent_result['grasp_valid']} "
            f"kf={int(intent_result['grasp_keyframe'])}/{int(intent_result['release_keyframe'])}"
        )
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.tight_layout()
        try:
            fig.savefig(str(paths.intent_preview), dpi=120, bbox_inches="tight")
        except Exception as e:  # pragma: no cover
            logger.warning("[intent] failed to write intent preview: %s", e)
        finally:
            plt.close(fig)
