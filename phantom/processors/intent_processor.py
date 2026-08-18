"""
Intent Processor Module (Contact-Grounded Retargeting — Stage A)

This is the front end of the contact-grounded retargeting pipeline described in
``b/doc/research/contact_grounded_retargeting.md``. Its eventual job is to turn a
human video into a task-space *intent* (per-frame EE targets, contact/grasp
events, object-anchored grasp poses) that the Stage B trajectory optimizer
consumes.

Implemented scope:

    1. Determine the task object nouns (per-demo ``objects.json`` or config).
       Used as an open-vocab seed prior when ``intent_seed_grounding=auto``;
       unused by the geometric fallback.
    2. Track the manipulated object. Default ``hand_sam2``: global moving-hand
       candidates, VLM-grounded in-hand box (YOLO-World / OWLv2) with geometric
       fallback, SAM2 image + hand-region overlap veto, SAM2 video bidirectional
       propagate, motion-consistency gate. Fallback ``yolo``: YOLO-World +
       per-frame SAM2 with hold-on-miss.
    3. Back-project the per-frame object mask with the metric depth map to build a
       per-frame object point cloud (camera frame + robot frame).
    4. Detect contact / segment free->grasp->transport->release phases
       (``_detect_contacts``): fingertip-object distance cue, with a wrap-aware
       lateral inflation so occluded contact faces still register, plus an
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from phantom.processors.base_processor import BaseProcessor
from phantom.processors.paths import Paths
from phantom.processors.phantom_data import HandSequence
from phantom.utils.pcd_utils import get_point_cloud_of_segmask
from phantom.utils.transform_utils import transform_pts

logger = logging.getLogger(__name__)

# OpenGL/ARKit (x right, y up, z back) → OpenCV (x right, y down, z forward).
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)

# MediaPipe / HaMeR 21-keypoint convention: wrist=0, thumb tip=4, index tip=8,
# middle tip=12, ring tip=16, pinky tip=20.
FINGERTIP_IDXS = [4, 8, 12]  # thumb / index / middle tips
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8
# Wrist + finger MCPs — on the palm, not on the in-hand object. Used as SAM negatives.
PALM_IDXS = [0, 1, 5, 9, 13, 17]
# All five tips — excluded from SAM negatives (they sit on / past the object).
TIP_IDXS = {4, 8, 12, 16, 20}
# MediaPipe 21-keypoint bones. Drawn as polylines then dilated to make a hand
# region that follows the fingers without filling the pinch gap (a convex hull
# of all 21 points would cover the in-hand object).
HAND_BONES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

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
        object_prompt (str):      fallback object noun if no objects.json.
        intent_dino_threshold (float): detector confidence threshold (default 0.01 for YOLO-World).
        intent_yolo_model (str):  ultralytics YOLO-World weight (default yolov8x-worldv2.pt).
        intent_seed_detector (str): open-vocab seed backend ``yolo`` (default) or ``owlv2``.
        intent_seed_grounding (str): ``auto`` (VLM then geom), ``vlm``, or ``geom``.
        intent_seed_stride (int): stride for the seed search (default 3).
        intent_seed_min_score (float): earliest-confident seed floor (default 0.01).
        intent_depth_max (float): max valid depth in meters (default 3.0).
        intent_min_object_pts (int): min points to consider a frame's cloud valid.
    """

    def __init__(self, args):
        super().__init__(args)
        # YOLO-World open-vocab scores are typically ~0.01-0.05, not DINO's ~0.3+.
        self.dino_threshold = float(getattr(self.cfg, "intent_dino_threshold", 0.01))
        self.yolo_model = str(getattr(self.cfg, "intent_yolo_model", "yolov8x-worldv2.pt"))
        self.owlv2_model = str(getattr(
            self.cfg, "intent_owlv2_model", "google/owlv2-base-patch16-ensemble",
        ))
        seed_det = str(getattr(self.cfg, "intent_seed_detector", "yolo")).strip().lower()
        if seed_det not in ("yolo", "owlv2"):
            logger.warning("[intent] unknown intent_seed_detector=%r; using yolo", seed_det)
            seed_det = "yolo"
        self.seed_detector = seed_det
        grounding = str(getattr(self.cfg, "intent_seed_grounding", "auto")).strip().lower()
        if grounding == "hand":
            grounding = "geom"
        if grounding not in ("auto", "vlm", "geom"):
            logger.warning("[intent] unknown intent_seed_grounding=%r; using auto", grounding)
            grounding = "auto"
        self.seed_grounding = grounding
        self.seed_stride = int(getattr(self.cfg, "intent_seed_stride", 3))
        self.seed_min_score = float(getattr(self.cfg, "intent_seed_min_score", 0.01))
        # When the prompt says "black"/"dark", rank YOLO seed boxes by dark-pixel
        # fraction so a grey hammer cannot beat a black stapler on score alone.
        self.seed_prefer_dark = str(getattr(self.cfg, "intent_seed_prefer_dark", "auto"))
        self.seed_dark_luma = float(getattr(self.cfg, "intent_seed_dark_luma", 60.0))
        self.seed_score_keep = float(getattr(self.cfg, "intent_seed_score_keep", 0.2))
        self.seed_tighten_dark = bool(getattr(self.cfg, "intent_seed_tighten_dark", True))
        self.depth_max = float(getattr(self.cfg, "intent_depth_max", 3.0))
        self.min_object_pts = int(getattr(self.cfg, "intent_min_object_pts", 50))
        self.mask_erode = int(getattr(self.cfg, "intent_mask_erode", 2))
        # Tracking backend: hand_sam2 (default, category-agnostic) or yolo.
        self.track_backend = str(getattr(self.cfg, "intent_track_backend", "hand_sam2")).strip().lower()
        self.seed_approach_window = int(getattr(self.cfg, "intent_seed_approach_window", 8))
        self.seed_hold_window = int(getattr(self.cfg, "intent_seed_hold_window", 8))
        self.seed_move_vel = float(getattr(self.cfg, "intent_seed_move_vel", 0.01))
        self.seed_move_run = int(getattr(self.cfg, "intent_seed_move_run", 4))
        self.seed_cand_k = int(getattr(self.cfg, "intent_seed_cand_k", 5))
        self.seed_probe_win = int(getattr(self.cfg, "intent_seed_probe_win", 8))
        self.seed_global_stride = int(getattr(self.cfg, "intent_seed_global_stride", 5))
        self.hand_dilate_px = int(getattr(self.cfg, "intent_hand_dilate_px", 25))
        self.track_hand_overlap_max = float(getattr(self.cfg, "intent_track_hand_overlap_max", 0.5))
        self.track_static_bonus_w = float(getattr(self.cfg, "intent_track_static_bonus_w", 0.3))
        self.track_min_disp = float(getattr(self.cfg, "intent_track_min_disp", 0.05))
        self._seed_low_conf = False
        self._seed_hand_overlap = 0.0
        self._seed_source = ""
        self._mask_stack_cache: Dict[str, Optional[np.ndarray]] = {}
        self.track_max_area_frac = float(getattr(self.cfg, "intent_track_max_area_frac", 0.20))
        self.track_neg_exclude_px = float(getattr(self.cfg, "intent_track_neg_exclude_px", 28.0))
        self.track_reseed_max = int(getattr(self.cfg, "intent_track_reseed_max", 2))
        self.track_motion_corr_min = float(getattr(self.cfg, "intent_track_motion_corr_min", 0.3))
        self.track_static_vel_max = float(getattr(self.cfg, "intent_track_static_vel_max", 0.02))
        self.track_attach_err_max = float(getattr(self.cfg, "intent_track_attach_err_max", 0.06))
        self._last_track_qa: Optional[dict] = None
        # Per-frame (or every-N) YOLO + SAM2-image tracking (yolo backend).
        self.track_stride = int(getattr(self.cfg, "intent_track_stride", 1))
        self.track_iou = float(getattr(self.cfg, "intent_track_iou", 0.15))
        # Pixel radius around the target-hand fingertips that can steal the
        # track onto an in-hand box even when a table remnant still overlaps
        # the previous mask. 0 disables the prior.
        self.track_hand_px = float(getattr(self.cfg, "intent_track_hand_px", 160.0))
        self.track_max_area_ratio = float(getattr(self.cfg, "intent_track_max_area_ratio", 3.5))
        # Max previous-mask-centroid → box-centre distance (px) for a no-overlap
        # re-id. Far jumps snap back onto table remnants; 0 disables the gate.
        self.track_jump_px = float(getattr(self.cfg, "intent_track_jump_px", 120.0))
        # EgoDex per-frame camera-to-world (HDF5 transforms/camera). Composes
        # T_cam2robot(t) = T_w2r @ T_c2w(t) so a static world point stays still
        # in robot frame despite head motion. Default T_w2r is the shoulder
        # calib at t_ref; ``intent_place_workspace`` replaces it with a rigid
        # placement that puts the grasp in front of the Panda.
        self.use_T_camera = str(getattr(self.cfg, "intent_use_T_camera", "auto"))
        self.T_camera_ref = int(getattr(self.cfg, "intent_T_camera_ref", 0))
        self.T_camera_convention = str(getattr(self.cfg, "intent_T_camera_convention", "opencv"))
        self.place_workspace = str(getattr(self.cfg, "intent_place_workspace", "auto"))
        target = getattr(self.cfg, "intent_place_target", [0.50, 0.0, 0.05])
        self.place_target = np.asarray(target, dtype=np.float64).reshape(3)
        self.grasp_offset_from = str(getattr(self.cfg, "intent_grasp_offset", "hand"))
        self.reuse_masks = bool(getattr(self.cfg, "intent_reuse_masks", False))
        self._T_c2w: Optional[np.ndarray] = None  # (T,4,4) EgoDex camera-to-world
        self._T_c2r: Optional[np.ndarray] = None  # (T,4,4) or None
        self._T_place: Optional[np.ndarray] = None  # (4,4) world-to-robot after freeze
        self.outlier_nb = int(getattr(self.cfg, "intent_outlier_nb", 20))
        self.outlier_std = float(getattr(self.cfg, "intent_outlier_std", 2.0))
        # Temporal centroid smoothing (small objects: per-frame depth-mean of a
        # tiny mask jitters in the camera-Z direction; the raw target drives
        # Stage B key_pos spikes at grasp/release). Median outlier rejection +
        # centered moving-average over the *valid* centroid track only.
        self.centroid_smooth_win = int(getattr(self.cfg, "intent_centroid_smooth_win", 5))
        self.centroid_reject_m = float(getattr(self.cfg, "intent_centroid_reject_m", 0.05))

        # Contact detection params.
        self.contact_dist_in = float(getattr(self.cfg, "intent_contact_dist_in", 0.03))
        self.contact_dist_out = float(getattr(self.cfg, "intent_contact_dist_out", 0.05))
        # Lateral (camera-XY) inflation for wrap grasps: if the nearest visible
        # surface is within this radius and |Δz| is small, treat |Δz| as the
        # contact score. 0 disables wrap and keeps pure Euclidean.
        self.contact_xy_inflate = float(getattr(self.cfg, "intent_contact_xy_inflate", 0.06))
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
        self._detector = None
        self._sam2 = None

    # ------------------------------------------------------------------
    # Detector lazy init
    # ------------------------------------------------------------------
    @property
    def detector(self):
        if self._detector is None:
            kind = str(getattr(self, "seed_detector", "yolo")).strip().lower()
            if kind == "owlv2":
                from phantom.detectors.detector_owlv2 import DetectorOwlv2
                model_id = str(getattr(
                    self, "owlv2_model", "google/owlv2-base-patch16-ensemble",
                ))
                logger.info("[intent] seed detector=owlv2 model=%s", model_id)
                self._detector = DetectorOwlv2(model_id)
            else:
                from phantom.detectors.detector_yolo_world import DetectorYoloWorld
                logger.info("[intent] seed detector=yolo model=%s", self.yolo_model)
                self._detector = DetectorYoloWorld(self.yolo_model)
        return self._detector

    # Backward-compatible alias used by older call sites / notebooks.
    dino = detector

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

        need_prompt = self.track_backend != "hand_sam2"
        object_prompt = self._get_object_prompt(paths, required=need_prompt)
        logger.info(
            "[intent] demo=%s backend=%s seed_detector=%s grounding=%s object_prompt=%r",
            data_sub_folder, self.track_backend, self.seed_detector,
            self.seed_grounding, object_prompt,
        )

        frames = self._load_frames(paths)  # (T, H, W, 3) RGB uint8
        n_frames = len(frames)
        logger.info("[intent] loaded %d frames of size %s", n_frames, frames.shape[1:3])
        self._T_c2r = self._load_T_cam2robot_seq(paths, n_frames)

        # Depth is needed by the hand_sam2 motion gate as well as the point cloud.
        depth = self._load_depth(paths, n_frames, frames.shape[1:3])

        # 1) object detection + mask tracking (or reuse a previous SAM run)
        object_masks, seed_idx, seed_bbox, seed_score = self._resolve_masks(
            paths, frames, object_prompt, n_frames, depth=depth,
        )
        if object_masks is None:
            return

        # 2) object point clouds from depth back-projection
        pcd_result = self._build_object_pointclouds(frames, object_masks, depth)

        # 3) contact detection + phase segmentation
        hands = self._load_hand_fingertips(paths, len(object_masks))
        contact_result = self._detect_contacts(pcd_result, hands)
        self._last_track_qa = self._track_quality_gate(
            object_masks, pcd_result, contact_result, hands, seed_idx,
        )
        self._save_track_quality(paths, self._last_track_qa)

        # 4) hand->gripper antipodal grasp synthesis (object-anchored G*)
        grasp_result = self._synthesize_grasp(pcd_result, contact_result, hands)

        # 4b) rigid-place the frozen EgoDex world into the Panda workspace so
        # G*/hands sit in front of the base instead of behind it.
        if self._want_place_workspace():
            pcd_result, hands, grasp_result = self._place_into_panda_workspace(
                pcd_result, hands, grasp_result, contact_result
            )

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
    # Object prompt / noun resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_object_nouns(data) -> List[str]:
        """Parse an ``objects.json`` payload into an ordered noun list."""
        if data is None:
            return []
        if isinstance(data, str):
            s = data.strip()
            return [s] if s else []
        if isinstance(data, (list, tuple)):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict):
            if "objects" in data:
                return IntentProcessor._parse_object_nouns(data.get("objects"))
            if "prompt" in data:
                return IntentProcessor._parse_object_nouns(data.get("prompt"))
            return []
        return []

    def _get_object_nouns(self, paths: Paths) -> List[str]:
        """Ordered object nouns: ``objects.json`` then config ``object_prompt``.

        ``objects[0]`` is the manipulated object; later entries are fallbacks
        if the first noun has no in-hand box on a candidate frame.
        """
        objects_json = Path(getattr(paths, "data_path", paths)) / "objects.json"
        if objects_json.exists():
            try:
                with open(objects_json, "r") as f:
                    data = json.load(f)
                nouns = self._parse_object_nouns(data)
                if nouns:
                    return nouns
            except Exception as e:  # noqa: BLE001
                logger.warning("[intent] failed to read %s: %s", objects_json, e)

        fallback = getattr(self, "cfg_object_prompt", None)
        if fallback:
            s = str(fallback).strip()
            if s:
                return [s]
        return []

    def _get_object_prompt(self, paths: Paths, required: bool = True) -> str:
        """Resolve a single object noun (first of ``_get_object_nouns``).

        Empty when optional (hand_sam2 backend with no objects.json / config).
        """
        nouns = self._get_object_nouns(paths)
        if nouns:
            return nouns[0]
        if not required:
            return ""
        raise ValueError(
            f"No object prompt available for {paths.data_path}. Provide a per-demo "
            f"objects.json or set `object_prompt` in the config."
        )

    @staticmethod
    def _pinch_uv(kpts_2d_t: np.ndarray) -> Optional[np.ndarray]:
        """Thumb–index midpoint; fall back to mean of valid fingertips."""
        k = np.asarray(kpts_2d_t, dtype=np.float32)
        if k.ndim == 1:
            k = k.reshape(-1, 2)
        else:
            k = k.reshape(-1, k.shape[-1])[:, :2]
        if k.shape[0] > max(THUMB_TIP_IDX, INDEX_TIP_IDX):
            thumb, index = k[THUMB_TIP_IDX], k[INDEX_TIP_IDX]
            if np.isfinite(thumb).all() and np.isfinite(index).all():
                return (0.5 * (thumb + index)).astype(np.float32)
        tips = []
        for i in FINGERTIP_IDXS:
            if i < len(k) and np.isfinite(k[i]).all():
                tips.append(k[i])
        if tips:
            return np.mean(np.stack(tips, axis=0), axis=0).astype(np.float32)
        valid = k[np.isfinite(k).all(axis=1)]
        if len(valid) == 0:
            return None
        return valid.mean(axis=0).astype(np.float32)

    @staticmethod
    def _pick_inhand_box(
        bboxes: np.ndarray,
        scores: np.ndarray,
        hand_uv: Optional[np.ndarray],
        hand_px: float,
        max_area: float,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Keep the nearest in-hand box; reject whole-table / far boxes.

        Returns ``(xyxy, score)`` or ``None`` if nothing is near the hand and
        below ``max_area``. Ties (same distance) break by higher score.
        """
        if hand_uv is None:
            return None
        uv = np.asarray(hand_uv, dtype=np.float32).reshape(-1)
        if uv.size < 2 or not np.isfinite(uv[:2]).all():
            return None
        b = np.asarray(bboxes, dtype=np.float32)
        if b.size == 0:
            return None
        b = b.reshape(-1, 4)
        s = np.asarray(scores, dtype=np.float32).reshape(-1)
        n = min(len(b), len(s))
        if n == 0:
            return None
        b, s = b[:n], s[:n]
        cx = 0.5 * (b[:, 0] + b[:, 2])
        cy = 0.5 * (b[:, 1] + b[:, 3])
        areas = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
        dist = np.hypot(cx - float(uv[0]), cy - float(uv[1]))
        keep = np.ones(n, dtype=bool)
        if float(hand_px) > 0:
            keep &= dist <= float(hand_px)
        if float(max_area) > 0:
            keep &= areas <= float(max_area)
        keep &= areas > 0
        if not keep.any():
            return None
        idx = np.flatnonzero(keep)
        order = np.lexsort((-s[idx], dist[idx]))
        i = int(idx[order[0]])
        return b[i].astype(np.float32), float(s[i])

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
    def _prefer_dark(self, prompt: str) -> bool:
        mode = str(self.seed_prefer_dark).strip().lower()
        if mode in ("0", "false", "no", "off"):
            return False
        if mode in ("1", "true", "yes", "on"):
            return True
        p = (prompt or "").lower()
        return any(w in p for w in ("black", "dark"))

    @staticmethod
    def _box_luma_stats(frame: np.ndarray, box: np.ndarray) -> Tuple[float, float]:
        """Return (mean luma, fraction of pixels below the caller’s threshold later)."""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return 255.0, 0.0
        patch = frame[y0:y1, x0:x1]
        if patch.ndim == 3:
            luma = (
                0.299 * patch[..., 0].astype(np.float32)
                + 0.587 * patch[..., 1].astype(np.float32)
                + 0.114 * patch[..., 2].astype(np.float32)
            )
        else:
            luma = patch.astype(np.float32)
        return float(luma.mean()), luma

    @staticmethod
    def _box_dark_frac(frame: np.ndarray, box: np.ndarray, luma_thr: float) -> float:
        mean, luma = IntentProcessor._box_luma_stats(frame, box)
        if not isinstance(luma, np.ndarray):
            return 0.0
        return float((luma < float(luma_thr)).mean())

    @staticmethod
    def _tighten_dark_box(
        frame: np.ndarray,
        box: np.ndarray,
        luma_thr: float = 60.0,
        min_core_frac: float = 0.05,
        pad: int = 8,
    ) -> np.ndarray:
        """Shrink ``box`` to the largest dark connected component inside it."""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        orig = np.array([x0, y0, x1, y1], dtype=np.float32)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return orig
        patch = frame[y0:y1, x0:x1]
        if patch.ndim == 3:
            luma = (
                0.299 * patch[..., 0].astype(np.float32)
                + 0.587 * patch[..., 1].astype(np.float32)
                + 0.114 * patch[..., 2].astype(np.float32)
            )
        else:
            luma = patch.astype(np.float32)
        bin_ = (luma < float(luma_thr)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(bin_, 8)
        if n <= 1:
            return orig
        areas = stats[1:, cv2.CC_STAT_AREA]
        k = int(np.argmax(areas)) + 1
        core_area = float(stats[k, cv2.CC_STAT_AREA])
        box_area = float(max((x1 - x0) * (y1 - y0), 1))
        if core_area < float(min_core_frac) * box_area:
            return orig
        cx, cy, cw, ch = stats[k, :4]
        nx0 = max(0, x0 + int(cx) - pad)
        ny0 = max(0, y0 + int(cy) - pad)
        nx1 = min(w, x0 + int(cx) + int(cw) + pad)
        ny1 = min(h, y0 + int(cy) + int(ch) + pad)
        if nx1 - nx0 < 4 or ny1 - ny0 < 4:
            return orig
        return np.array([nx0, ny0, nx1, ny1], dtype=np.float32)

    @staticmethod
    def _select_yolo_box(
        frame: np.ndarray,
        bboxes: np.ndarray,
        scores: np.ndarray,
        prefer_dark: bool = False,
        dark_luma: float = 60.0,
        score_keep: float = 0.2,
    ) -> Tuple[Optional[np.ndarray], float, str]:
        """Pick a YOLO box. ``prefer_dark`` uses dark-pixel fraction among
        boxes that keep at least ``score_keep`` of the top score (hammer vs stapler)."""
        if bboxes is None or len(bboxes) == 0:
            return None, 0.0, "empty"
        bboxes = np.asarray(bboxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(1, 4)
        k_sc = int(np.argmax(scores))
        if not prefer_dark:
            return np.asarray(bboxes[k_sc], dtype=np.float32), float(scores[k_sc]), "score"
        smax = float(scores.max())
        keep = scores >= max(float(score_keep) * smax, 1e-6)
        if not bool(keep.any()):
            keep = np.ones(len(bboxes), dtype=bool)
        dark = np.array(
            [IntentProcessor._box_dark_frac(frame, b, dark_luma) for b in bboxes],
            dtype=np.float32,
        )
        k_cands = np.flatnonzero(keep)
        k = int(k_cands[int(np.argmax(dark[k_cands]))])
        return np.asarray(bboxes[k], dtype=np.float32), float(scores[k]), "dark"

    def _resolve_masks(
        self,
        paths: Paths,
        frames: np.ndarray,
        object_prompt: str,
        n_frames: int,
        depth: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], int, Optional[np.ndarray], float]:
        """Track or reuse previously saved object masks."""
        mask_path = str(paths.object_masks)
        if self.reuse_masks and os.path.exists(mask_path):
            object_masks = np.load(mask_path)
            if len(object_masks) == n_frames:
                seed_idx, seed_score = 0, 0.0
                if os.path.exists(paths.object_pcd):
                    prev = np.load(paths.object_pcd, allow_pickle=True)
                    seed_idx = int(prev["seed_idx"]) if "seed_idx" in prev.files else 0
                    seed_score = float(prev["seed_score"]) if "seed_score" in prev.files else 0.0
                logger.info(
                    "[intent] reusing saved masks %s  seed=%d", mask_path, seed_idx
                )
                return object_masks, seed_idx, None, seed_score
            logger.warning(
                "[intent] saved masks T=%d != frames T=%d; re-tracking",
                len(object_masks), n_frames,
            )

        if self.track_backend == "hand_sam2":
            object_masks, seed_idx, seed_bbox, seed_score = self._track_masks_hand_sam2(
                paths, frames, depth,
            )
            if object_masks is not None:
                return object_masks, seed_idx, seed_bbox, seed_score
            logger.warning("[intent] hand_sam2 produced no masks; falling back to yolo")
            if not object_prompt:
                logger.warning("[intent] no object prompt for yolo fallback; skipping demo")
                return None, 0, None, 0.0

        seed_idx, seed_bbox, seed_score = self._detect_seed(frames, object_prompt)
        if seed_idx is None:
            logger.warning(
                "[intent] object %r not detected in any frame (threshold=%.2f); "
                "skipping demo", object_prompt, self.dino_threshold,
            )
            return None, 0, None, 0.0
        logger.info(
            "[intent] seed frame=%d score=%.3f bbox=%s how=%s",
            seed_idx, seed_score, seed_bbox.astype(int), getattr(self, "_last_seed_how", "?"),
        )
        hand_uv = self._load_target_hand_uv(paths, n_frames)
        object_masks = self._track_masks(
            frames, object_prompt, seed_idx, seed_bbox, hand_uv=hand_uv,
        )
        return object_masks, seed_idx, seed_bbox, seed_score

    @staticmethod
    def _compose_T_cam2robot_seq(
        T_c2w: np.ndarray, T_calib: np.ndarray, ref_idx: int = 0,
    ) -> np.ndarray:
        """T_cam2robot(t) = T_calib @ T_c2w(ref)^{-1} @ T_c2w(t)."""
        T_c2w = np.asarray(T_c2w, dtype=np.float64)
        T_calib = np.asarray(T_calib, dtype=np.float64)
        n = len(T_c2w)
        ref = int(np.clip(ref_idx, 0, n - 1))
        T_w2r = T_calib @ np.linalg.inv(T_c2w[ref])
        return np.einsum("ij,njk->nik", T_w2r, T_c2w)

    def _want_place_workspace(self) -> bool:
        mode = str(self.place_workspace).strip().lower()
        if mode in ("0", "false", "no", "off"):
            return False
        if mode in ("1", "true", "yes", "on"):
            return self._T_c2w is not None
        return self._T_c2w is not None  # auto: place when we have T_camera

    @staticmethod
    def _place_T_w2r(
        p_world_ref: np.ndarray,
        look_world: np.ndarray,
        p_star: np.ndarray,
        up_world: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """SE(3) mapping EgoDex world into the Panda robot frame.

        Robot +Z aligns with world-up (ARKit +Y by default); robot +X is the
        camera look projected onto the table so the arm faces the scene.
        ``p_world_ref`` maps exactly onto ``p_star``.
        """
        p_world_ref = np.asarray(p_world_ref, dtype=np.float64).reshape(3)
        look = np.asarray(look_world, dtype=np.float64).reshape(3)
        p_star = np.asarray(p_star, dtype=np.float64).reshape(3)
        if up_world is None:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up = np.asarray(up_world, dtype=np.float64).reshape(3)
            nrm = np.linalg.norm(up)
            up = up / nrm if nrm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_fwd = look - np.dot(look, up) * up
        n = np.linalg.norm(x_fwd)
        if n < 1e-6:
            helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
            x_fwd = np.cross(up, helper)
            n = np.linalg.norm(x_fwd)
        x_fwd = x_fwd / n
        y_axis = np.cross(up, x_fwd)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_fwd, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        R = np.stack([x_fwd, y_axis, z_axis], axis=0)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = p_star - R @ p_world_ref
        return T

    @staticmethod
    def _apply_T_xyz(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Apply a 4x4 transform to an array of xyz points, leaving NaNs in place."""
        xyz = np.asarray(xyz, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        out_shape = xyz.shape
        pts = xyz.reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1)
        out = pts.copy()
        if bool(valid.any()):
            out[valid] = transform_pts(pts[valid], T)
        return out.reshape(out_shape)

    def _place_into_panda_workspace(
        self,
        pcd_result: Dict[str, list],
        hands: Dict[str, dict],
        grasp_result: Dict[str, object],
        contact_result: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, list], Dict[str, dict], Dict[str, object]]:
        """Left-multiply a constant T_place after T_camera freeze.

        Intermediate robot frame is EgoDex world (T_c2r = T_c2w). After this
        call, robot frame is the Panda workspace and T_c2r = T_place @ T_c2w.
        """
        if self._T_c2w is None:
            return pcd_result, hands, grasp_result
        T_c2w = np.asarray(self._T_c2w, dtype=np.float64)
        n = len(T_c2w)
        ref = int(np.clip(self.T_camera_ref, 0, n - 1))
        grasp_kf = int(contact_result.get("grasp_keyframe", -1))
        p_ref = None
        if 0 <= grasp_kf < n:
            p_ref = self._ee_target_from_hand(self._hands_for_contact(hands), grasp_kf)
        if p_ref is None and bool(grasp_result.get("valid", False)):
            p_ref = np.asarray(grasp_result["G_center"], dtype=np.float64)
        if p_ref is None or not np.isfinite(np.asarray(p_ref)).all():
            cents = np.asarray(pcd_result["centroids_robot"], dtype=np.float64)
            ok = np.asarray(pcd_result["valid"], dtype=bool) & np.isfinite(cents).all(axis=1)
            if not bool(np.any(ok)):
                logger.warning("[intent] T_place skipped: no world-frame reference point")
                return pcd_result, hands, grasp_result
            p_ref = cents[ok].mean(axis=0)
        look = T_c2w[ref, :3, 2]
        T_place = self._place_T_w2r(p_ref, look, self.place_target)
        self._T_place = T_place
        self._T_c2r = np.einsum("ij,njk->nik", T_place, T_c2w)

        pts_rf = []
        for pts in pcd_result["points_robot"]:
            pts = np.asarray(pts, dtype=np.float32)
            pts_rf.append(
                self._apply_T_xyz(pts, T_place).astype(np.float32) if len(pts) else pts
            )
        pcd_result["points_robot"] = pts_rf
        pcd_result["centroids_robot"] = self._apply_T_xyz(
            pcd_result["centroids_robot"], T_place
        ).astype(np.float32)

        for h in hands.values():
            h["fingertips"] = self._apply_T_xyz(h["fingertips"], T_place).astype(np.float32)

        R = T_place[:3, :3]
        if bool(grasp_result.get("valid", False)):
            grasp_result["G_center"] = self._apply_T_xyz(
                np.asarray(grasp_result["G_center"]), T_place
            ).astype(np.float32)
            grasp_result["G_rot"] = (R @ np.asarray(grasp_result["G_rot"], dtype=np.float64)).astype(np.float32)
            if "G_rot_pipeline" in grasp_result:
                grasp_result["G_rot_pipeline"] = (
                    R @ np.asarray(grasp_result["G_rot_pipeline"], dtype=np.float64)
                ).astype(np.float32)
            for key in ("closing_axis", "approach_axis"):
                if key in grasp_result:
                    v = np.asarray(grasp_result[key], dtype=np.float64)
                    grasp_result[key] = (R @ v).astype(np.float32)
            if "contact_points" in grasp_result:
                grasp_result["contact_points"] = self._apply_T_xyz(
                    np.asarray(grasp_result["contact_points"]), T_place
                ).astype(np.float32)

        g_r = self._apply_T_xyz(np.asarray(p_ref, dtype=np.float64), T_place)
        logger.info(
            "[intent] T_place: world ref %s -> robot %s (target %s)  look=%s",
            np.round(np.asarray(p_ref, dtype=float), 3).tolist(),
            np.round(g_r, 3).tolist(),
            np.round(self.place_target, 3).tolist(),
            np.round(look, 3).tolist(),
        )
        return pcd_result, hands, grasp_result

    def _T_cam2robot_at(self, i: int) -> np.ndarray:
        if self._T_c2r is None:
            return np.asarray(self.T_cam2robot, dtype=np.float64)
        return self._T_c2r[min(max(int(i), 0), len(self._T_c2r) - 1)]

    def _load_T_c2w(self, paths: Paths, n: int) -> Optional[np.ndarray]:
        mode = str(self.use_T_camera).strip().lower()
        if mode in ("0", "false", "no", "off"):
            return None
        cands = [
            Path(paths.data_path) / "T_camera_c2w.npy",
            Path(getattr(paths, "T_camera_c2w", Path(paths.data_path) / "T_camera_c2w.npy")),
        ]
        cfg_path = getattr(self.cfg, "intent_T_camera_path", None)
        if cfg_path:
            cands.insert(0, Path(str(cfg_path)))
        T = None
        src = None
        for p in cands:
            if p is None:
                continue
            p = Path(p)
            if p.is_file():
                T = np.load(p)
                src = str(p)
                break
        if T is None:
            hdf5 = self._resolve_egodex_hdf5(paths)
            if hdf5 is not None:
                import h5py
                with h5py.File(hdf5, "r") as f:
                    T = np.asarray(f["transforms/camera"][:], dtype=np.float64)
                src = str(hdf5)
                out = Path(paths.data_path) / "T_camera_c2w.npy"
                try:
                    np.save(out, T)
                    logger.info("[intent] cached T_c2w -> %s", out)
                except OSError:
                    pass
        if T is None:
            if mode in ("1", "true", "yes", "on"):
                logger.warning("[intent] intent_use_T_camera=%s but no T_c2w found", mode)
            return None
        T = np.asarray(T, dtype=np.float64)
        if T.ndim != 3 or T.shape[1:] != (4, 4):
            logger.warning("[intent] bad T_c2w shape %s from %s", T.shape, src)
            return None
        if self.T_camera_convention == "opengl":
            T = T @ GL_TO_CV[None]
        if len(T) != n:
            logger.warning("[intent] T_c2w T=%d != frames T=%d; trunc/pad", len(T), n)
            if len(T) >= n:
                T = T[:n]
            else:
                pad = np.repeat(T[-1:], n - len(T), axis=0)
                T = np.concatenate([T, pad], axis=0)
        logger.info("[intent] loaded T_c2w (%d frames) from %s  convention=%s", n, src, self.T_camera_convention)
        return T

    def _resolve_egodex_hdf5(self, paths: Paths) -> Optional[Path]:
        cfg_h5 = getattr(self.cfg, "egodex_hdf5", None)
        if cfg_h5 and Path(str(cfg_h5)).is_file():
            return Path(str(cfg_h5))
        root = Path(str(getattr(self.cfg, "egodex_root", "/home/a26160/DATA/test")))
        demo_dir = Path(paths.data_path)
        demo = demo_dir.name
        parent = demo_dir.parent.name
        task = getattr(self.cfg, "egodex_task", None)
        if not task:
            task = parent[7:] if parent.startswith("egodex_") else parent
        cand = root / str(task) / f"{demo}.hdf5"
        return cand if cand.is_file() else None

    def _load_T_cam2robot_seq(self, paths: Paths, n: int) -> Optional[np.ndarray]:
        T_c2w = self._load_T_c2w(paths, n)
        self._T_c2w = T_c2w
        if T_c2w is None:
            logger.info("[intent] robot-frame pose: constant T_cam2robot (no per-frame T_camera)")
            return None
        cam_d = np.linalg.norm(np.diff(T_c2w[:, :3, 3], axis=0), axis=1)
        if self._want_place_workspace():
            logger.info(
                "[intent] freeze world via T_c2w (robot:=world until T_place)  "
                "ref=%d  cam translation Δ mean=%.3f max=%.3f m",
                int(np.clip(self.T_camera_ref, 0, n - 1)),
                float(cam_d.mean()) if len(cam_d) else 0.0,
                float(cam_d.max()) if len(cam_d) else 0.0,
            )
            return np.asarray(T_c2w, dtype=np.float64)
        T_seq = self._compose_T_cam2robot_seq(T_c2w, self.T_cam2robot, self.T_camera_ref)
        logger.info(
            "[intent] using per-frame T_cam2robot from T_c2w  ref=%d  "
            "cam translation Δ mean=%.3f max=%.3f m",
            int(np.clip(self.T_camera_ref, 0, n - 1)),
            float(cam_d.mean()) if len(cam_d) else 0.0,
            float(cam_d.max()) if len(cam_d) else 0.0,
        )
        return T_seq

    def _detect_seed(
        self, frames: np.ndarray, object_prompt: str
    ) -> Tuple[Optional[int], Optional[np.ndarray], float]:
        """Pick the SAM2 seed detection.

        Prefer the *earliest* confident detection (>= ``seed_min_score``) rather
        than the global maximum: early frames typically show the object isolated,
        before manipulation/occlusion causes the box to drift onto the hand or
        nearby clutter. Falls back to the global best if nothing clears the floor.

        For dark objects (prompt contains black/dark), re-rank same-frame YOLO
        boxes by the fraction of dark pixels so a high-score grey distractor
        cannot beat the true black instance.
        """
        prefer_dark = self._prefer_dark(object_prompt)
        best_idx: Optional[int] = None
        best_bbox: Optional[np.ndarray] = None
        best_score = -1.0
        early_idx: Optional[int] = None
        early_bbox: Optional[np.ndarray] = None
        early_score = -1.0
        early_how = "score"
        best_how = "score"
        for idx in range(0, len(frames), max(1, self.seed_stride)):
            bboxes, scores = self.detector.get_bboxes(
                frames[idx], object_prompt, threshold=self.dino_threshold
            )
            if len(bboxes) == 0:
                continue
            bbox, score, how = self._select_yolo_box(
                frames[idx], bboxes, scores,
                prefer_dark=prefer_dark,
                dark_luma=self.seed_dark_luma,
                score_keep=self.seed_score_keep,
            )
            if bbox is None:
                continue
            if prefer_dark and self.seed_tighten_dark:
                bbox = self._tighten_dark_box(
                    frames[idx], bbox, luma_thr=self.seed_dark_luma,
                )
            if score > best_score:
                best_score = score
                best_bbox = np.asarray(bbox, dtype=np.float32)
                best_idx = idx
                best_how = how
            if early_idx is None and score >= self.seed_min_score:
                early_idx = idx
                early_bbox = np.asarray(bbox, dtype=np.float32)
                early_score = score
                early_how = how
        if early_idx is not None:
            self._last_seed_how = early_how
            logger.info(
                "[intent] seed-select prefer_dark=%s how=%s dark_frac=%.2f",
                prefer_dark, early_how,
                self._box_dark_frac(frames[early_idx], early_bbox, self.seed_dark_luma)
                if early_bbox is not None else float("nan"),
            )
            return early_idx, early_bbox, early_score
        self._last_seed_how = best_how
        return best_idx, best_bbox, best_score

    @staticmethod
    def _box_mask_iou(mask: np.ndarray, box: np.ndarray) -> float:
        """IoU between a binary mask and an XYXY box (as a filled rectangle)."""
        h, w = mask.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        box_area = (x1 - x0) * (y1 - y0)
        inter = int(mask[y0:y1, x0:x1].astype(bool).sum())
        mask_area = int(mask.astype(bool).sum())
        union = mask_area + box_area - inter
        return float(inter) / float(union) if union > 0 else 0.0

    @staticmethod
    def _box_areas(bboxes: np.ndarray) -> np.ndarray:
        b = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        return np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])

    @staticmethod
    def _size_ok(
        bboxes: np.ndarray, prev_mask: np.ndarray, max_area_ratio: float,
    ) -> np.ndarray:
        n = len(np.asarray(bboxes).reshape(-1, 4)) if bboxes is not None else 0
        if n == 0 or max_area_ratio is None or float(max_area_ratio) <= 0:
            return np.ones(n, dtype=bool)
        prev_area = max(float(np.asarray(prev_mask).astype(bool).sum()), 1.0)
        return IntentProcessor._box_areas(bboxes) <= float(max_area_ratio) * prev_area

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(np.asarray(mask).astype(bool))
        if len(xs) == 0:
            return np.array([np.nan, np.nan], dtype=np.float32)
        return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)

    @staticmethod
    def _associate_yolo_box(
        prev_mask: np.ndarray,
        bboxes: np.ndarray,
        scores: np.ndarray,
        iou_min: float,
        hand_uv: Optional[np.ndarray] = None,
        hand_px: float = 0.0,
        max_area_ratio: float = 0.0,
        jump_px: float = 0.0,
    ) -> Tuple[Optional[np.ndarray], str, float]:
        """Pick a YOLO box given the previous mask.

        Order:

        1. Among boxes whose centre is within ``hand_px`` of the target-hand
           UV *and* that are not huge vs the previous mask, pick the highest
           score. Labelled ``hand`` when it does not overlap (in-hand lift).
        2. If some box overlaps the previous mask, keep the max-IoU one
           (same instance; table remnant wins over a higher-score distractor).
        3. Else re-identify with the top YOLO score among reasonably sized
           boxes that are within ``jump_px`` of the previous mask centroid
           (ego-motion). Far boxes would snap back to a table leftover — hold.
        4. Empty YOLO / too-far reid → hold.
        """
        if bboxes is None or len(bboxes) == 0:
            return None, "hold", 0.0
        bboxes = np.asarray(bboxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(1, 4)
        ious = np.array(
            [IntentProcessor._box_mask_iou(prev_mask, b) for b in bboxes],
            dtype=np.float32,
        )
        size_ok = IntentProcessor._size_ok(bboxes, prev_mask, max_area_ratio)

        if (
            hand_uv is not None
            and float(hand_px) > 0
            and np.shape(hand_uv)[-1] >= 2
            and np.isfinite(np.asarray(hand_uv, dtype=np.float32).reshape(-1)[:2]).all()
        ):
            uv = np.asarray(hand_uv, dtype=np.float32).reshape(-1)[:2]
            cx = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
            cy = 0.5 * (bboxes[:, 1] + bboxes[:, 3])
            dist = np.hypot(cx - uv[0], cy - uv[1])
            near = (dist <= float(hand_px)) & size_ok
            if bool(near.any()):
                k_cands = np.flatnonzero(near)
                k = int(k_cands[int(np.argmax(scores[k_cands]))])
                how = "assoc" if float(ious[k]) >= iou_min else "hand"
                return np.asarray(bboxes[k], dtype=np.float32), how, float(ious[k])

        overlap = (ious >= iou_min) & size_ok
        if bool(overlap.any()):
            k = int(np.flatnonzero(overlap)[int(np.argmax(ious[overlap]))])
            return np.asarray(bboxes[k], dtype=np.float32), "assoc", float(ious[k])

        if not bool(size_ok.any()):
            return None, "hold", 0.0
        k_cands = np.flatnonzero(size_ok)
        k = int(k_cands[int(np.argmax(scores[k_cands]))])
        if float(jump_px) > 0:
            pc = IntentProcessor._mask_centroid(prev_mask)
            bc = 0.5 * (bboxes[k, :2] + bboxes[k, 2:])
            if np.isfinite(pc).all() and float(np.hypot(*(bc - pc))) > float(jump_px):
                return None, "hold", float(ious[k])
        return np.asarray(bboxes[k], dtype=np.float32), "reid", float(ious[k])

    def _sam_box_mask(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        mask = self.sam2.segment_box(frame, bbox)
        if mask is None or int(mask.sum()) < self.min_object_pts:
            return None
        return mask.astype(np.uint8)

    def _refresh_mask(
        self,
        frame: np.ndarray,
        object_prompt: str,
        prev_mask: np.ndarray,
        hand_uv: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, str]:
        """YOLO → associate with ``prev_mask`` → single-frame SAM. Hold on miss."""
        bboxes, scores = self.detector.get_bboxes(
            frame, object_prompt, threshold=self.dino_threshold,
        )
        if prev_mask is None or int(prev_mask.sum()) == 0:
            if len(bboxes) == 0:
                return prev_mask, "hold"
            bbox = np.asarray(bboxes[int(np.argmax(scores))], dtype=np.float32)
            mask = self._sam_box_mask(frame, bbox)
            return (mask if mask is not None else prev_mask), "reid"
        bbox, how, iou = self._associate_yolo_box(
            prev_mask, bboxes, scores, self.track_iou,
            hand_uv=hand_uv, hand_px=self.track_hand_px,
            max_area_ratio=self.track_max_area_ratio,
            jump_px=self.track_jump_px,
        )
        if bbox is None:
            return prev_mask, "hold"
        mask = self._sam_box_mask(frame, bbox)
        if mask is None:
            return prev_mask, "hold"
        logger.debug(
            "[intent] track %s iou=%.3f box=%s n=%d",
            how, iou, bbox.astype(int), int(mask.sum()),
        )
        if how in ("hand", "reid"):
            logger.info(
                "[intent] track-%s iou=%.3f box=%s n=%d",
                how, iou, bbox.astype(int), int(mask.sum()),
            )
        return mask, how

    def _load_target_hand_uv(self, paths: Paths, n: int) -> np.ndarray:
        """Mean fingertip UV of ``target_hand`` (NaN when undetected)."""
        uv = np.full((n, 2), np.nan, dtype=np.float32)
        side = str(getattr(self, "target_hand", "") or "").lower()
        if side not in ("left", "right"):
            return uv
        hand_path = getattr(paths, f"hand_data_{side}", None)
        if hand_path is None or not os.path.exists(hand_path):
            logger.info("[intent] no %s hand_data; tracking without hand prior", side)
            return uv
        data = np.load(hand_path, allow_pickle=True)
        k2 = np.asarray(data["kpts_2d"], dtype=np.float32)
        det = np.asarray(data["hand_detected"], dtype=bool)
        if k2.ndim != 3 or k2.shape[1] < max(FINGERTIP_IDXS) + 1:
            logger.warning("[intent] unexpected kpts_2d shape %s; skip hand prior", k2.shape)
            return uv
        m = min(n, len(k2), len(det))
        tips = k2[:m, FINGERTIP_IDXS, :2]
        mean = tips.mean(axis=1)
        valid = det[:m] & np.isfinite(mean).all(axis=1)
        uv[np.flatnonzero(valid)] = mean[valid]
        logger.info("[intent] target-hand uv (%s): %d/%d frames", side, int(valid.sum()), n)
        return uv

    def _uv_crop_delta(self, paths: Paths, frame_hw: Tuple[int, int]) -> float:
        """X-offset applied to full-res keypoints when frames are square-cropped."""
        if not self.square:
            return 0.0
        video = getattr(paths, "video_left", None)
        if video is None or not os.path.exists(str(video)):
            return 0.0
        cap = cv2.VideoCapture(str(video))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if orig_w <= orig_h:
            return 0.0
        return float((orig_w - orig_h) // 2)

    def _adjust_kpts2d(
        self, kpts_2d: np.ndarray, paths: Paths, frame_hw: Tuple[int, int],
    ) -> np.ndarray:
        k = np.asarray(kpts_2d, dtype=np.float32).copy()
        delta = self._uv_crop_delta(paths, frame_hw)
        if delta:
            k[..., 0] -= delta
        return k

    def _load_target_hand_kpts(
        self, paths: Paths, n: int, frame_hw: Tuple[int, int],
    ) -> Optional[dict]:
        """Target-hand 2D/3D keypoints + aperture for hand-anchored tracking."""
        side = str(getattr(self, "target_hand", "") or "").lower()
        if side not in ("left", "right"):
            return None
        hand_path = getattr(paths, f"hand_data_{side}", None)
        if hand_path is None or not os.path.exists(hand_path):
            logger.info("[intent] no %s hand_data; cannot run hand_sam2", side)
            return None
        data = np.load(hand_path, allow_pickle=True)
        k2 = np.asarray(data["kpts_2d"], dtype=np.float32)
        k3 = np.asarray(data["kpts_3d"], dtype=np.float32)
        det = np.asarray(data["hand_detected"], dtype=bool)
        if k2.ndim != 3 or k3.ndim != 3 or k2.shape[1] < 21 or k3.shape[1] < 21:
            logger.warning("[intent] unexpected hand kpts shape k2=%s k3=%s", k2.shape, k3.shape)
            return None
        m = min(n, len(k2), len(k3), len(det))
        k2 = self._adjust_kpts2d(k2[:m], paths, frame_hw)
        k3_rf = self._to_robot_frame(k3[:m])
        detected = np.zeros(n, dtype=bool)
        detected[:m] = det[:m]
        kpts_2d = np.full((n, k2.shape[1], 2), np.nan, dtype=np.float32)
        kpts_2d[:m] = k2[..., :2]
        kpts_rf = np.full((n, k3_rf.shape[1], 3), np.nan, dtype=np.float32)
        kpts_rf[:m] = k3_rf
        aperture = np.full(n, np.nan, dtype=np.float32)
        aperture[:m] = np.linalg.norm(
            k3_rf[:, THUMB_TIP_IDX] - k3_rf[:, INDEX_TIP_IDX], axis=1
        )
        tips_rf = np.full((n, 3), np.nan, dtype=np.float32)
        tips_rf[:m] = k3_rf[:, FINGERTIP_IDXS, :].mean(axis=1)
        logger.info(
            "[intent] target-hand kpts (%s): %d/%d detected", side, int(detected.sum()), n,
        )
        return {
            "side": side,
            "kpts_2d": kpts_2d,
            "kpts_rf": kpts_rf,
            "detected": detected,
            "aperture": aperture,
            "tips_rf": tips_rf,
        }

    def _load_mask_stack(self, path) -> Optional[np.ndarray]:
        if path is None:
            return None
        key = str(path)
        cache = self._mask_stack_cache
        if key in cache:
            return cache[key]
        if not os.path.exists(key):
            cache[key] = None
            return None
        try:
            arr = np.load(key)
        except Exception as e:  # noqa: BLE001
            logger.warning("[intent] failed to load mask stack %s: %s", key, e)
            cache[key] = None
            return None
        cache[key] = arr
        return arr

    @staticmethod
    def _mask_frame(
        stack: Optional[np.ndarray], t: int, frame_hw: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if stack is None or t < 0 or t >= len(stack):
            return None
        m = np.asarray(stack[t])
        while m.ndim > 2:
            m = m[0]
        m = (m > 0).astype(np.uint8)
        h, w = frame_hw
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return m

    def _load_hand_mask(
        self, paths: Paths, side: str, t: int, frame_hw: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        mask_path = getattr(paths, f"masks_hand_{side}", None)
        return self._mask_frame(self._load_mask_stack(mask_path), t, frame_hw)

    @staticmethod
    def _hand_region_mask(
        kpts_2d_t: np.ndarray,
        frame_hw: Tuple[int, int],
        dilate_px: int = 25,
        extra_mask: Optional[np.ndarray] = None,
        pinch_punch_px: float = 28.0,
    ) -> Optional[np.ndarray]:
        """Hand region from GT 21 keypoints: palm hull + bone polylines, dilated.

        A filled convex hull of *all* 21 points would cover the pinch gap (the
        in-hand object). Bones + palm-MCP hull follow the hand without filling
        that gap. ``extra_mask`` (``masks_hand_*``) is unioned, then a circle
        around the pinch is punched out so a coarse extra mask cannot veto the
        object.
        """
        h, w = int(frame_hw[0]), int(frame_hw[1])
        k = np.asarray(kpts_2d_t, dtype=np.float32).reshape(-1, 2)
        region = np.zeros((h, w), dtype=np.uint8)
        valid = np.isfinite(k).all(axis=1)
        if int(valid.sum()) < 3:
            if extra_mask is None:
                return None
            region = (np.asarray(extra_mask) > 0).astype(np.uint8)
            if region.shape != (h, w):
                region = cv2.resize(region, (w, h), interpolation=cv2.INTER_NEAREST)
            return region

        palm = []
        for i in PALM_IDXS:
            if i < len(k) and valid[i]:
                palm.append([float(k[i, 0]), float(k[i, 1])])
        if len(palm) >= 3:
            hull = cv2.convexHull(np.asarray(palm, dtype=np.float32).reshape(-1, 1, 2))
            cv2.fillConvexPoly(region, hull.astype(np.int32), 1)

        thickness = max(2, int(round(float(dilate_px) / 3.0)))
        for a, b in HAND_BONES:
            if a < len(k) and b < len(k) and valid[a] and valid[b]:
                pa = (int(round(float(k[a, 0]))), int(round(float(k[a, 1]))))
                pb = (int(round(float(k[b, 0]))), int(round(float(k[b, 1]))))
                cv2.line(region, pa, pb, 1, thickness=thickness)

        kdil = max(3, int(dilate_px))
        if kdil % 2 == 0:
            kdil += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kdil, kdil))
        region = cv2.dilate(region, kernel)

        if extra_mask is not None:
            extra = (np.asarray(extra_mask) > 0).astype(np.uint8)
            if extra.shape != (h, w):
                extra = cv2.resize(extra, (w, h), interpolation=cv2.INTER_NEAREST)
            region = np.maximum(region, extra)
            if (
                THUMB_TIP_IDX < len(k) and INDEX_TIP_IDX < len(k)
                and valid[THUMB_TIP_IDX] and valid[INDEX_TIP_IDX]
                and float(pinch_punch_px) > 0
            ):
                mid = 0.5 * (k[THUMB_TIP_IDX] + k[INDEX_TIP_IDX])
                cv2.circle(
                    region,
                    (int(round(float(mid[0]))), int(round(float(mid[1])))),
                    max(1, int(round(float(pinch_punch_px)))),
                    0, -1,
                )
        return region

    def _hand_region_for_frame(
        self, paths: Paths, hand: dict, t: int, frame_hw: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        extra = self._load_hand_mask(paths, hand["side"], t, frame_hw)
        return self._hand_region_mask(
            hand["kpts_2d"][t],
            frame_hw,
            dilate_px=int(getattr(self, "hand_dilate_px", 25)),
            extra_mask=extra,
            pinch_punch_px=float(getattr(self, "track_neg_exclude_px", 28.0)),
        )

    @staticmethod
    def _mask_overlap_frac(mask: np.ndarray, region: Optional[np.ndarray]) -> float:
        m = np.asarray(mask).astype(bool)
        area = int(m.sum())
        if area == 0 or region is None:
            return 0.0
        r = np.asarray(region).astype(bool)
        if r.shape != m.shape:
            r = cv2.resize(
                r.astype(np.uint8), (m.shape[1], m.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return float(int((m & r).sum())) / float(area)

    @staticmethod
    def _apply_hand_overlap_veto(
        mask: np.ndarray,
        hand_region: Optional[np.ndarray],
        max_overlap: float = 0.5,
        min_pts: int = 50,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Hard-reject (or subtract) a seed mask that is mostly the hand.

        Returns ``(cleaned_or_None, overlap_before_subtract)``. ``None`` means
        the candidate is the hand (or leftover after subtract is too small).
        """
        m = np.asarray(mask)
        while m.ndim > 2:
            m = m[0]
        m = (m > 0).astype(np.uint8)
        if hand_region is None:
            return m, 0.0
        r = np.asarray(hand_region)
        while r.ndim > 2:
            r = r[0]
        r = (r > 0).astype(np.uint8)
        if r.shape != m.shape:
            r = cv2.resize(r, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        area = int(m.sum())
        if area == 0:
            return None, 1.0
        overlap = float(int((m.astype(bool) & r.astype(bool)).sum())) / float(area)
        cleaned = m.copy()
        cleaned[r.astype(bool)] = 0
        leftover = int(cleaned.sum())
        if overlap > float(max_overlap):
            if leftover >= int(min_pts):
                return cleaned, overlap
            return None, overlap
        if leftover >= int(min_pts):
            return cleaned, overlap
        return m, overlap

    @staticmethod
    def _select_seed_frame_from_hand(
        detected: np.ndarray,
        aperture: np.ndarray,
        tips_xyz: np.ndarray,
        approach_window: int = 8,
        hold_window: int = 8,
    ) -> Optional[int]:
        """Pick a grasp-like seed frame from hand kinematics only.

        Prefers a local aperture minimum whose fingertip speed drops from the
        approach window into the subsequent hold window. Falls back to the
        global minimum aperture among detected frames.
        """
        n = len(detected)
        det = np.asarray(detected, dtype=bool)
        ap = np.asarray(aperture, dtype=np.float32)
        tips = np.asarray(tips_xyz, dtype=np.float32)
        if n == 0 or not det.any():
            return None

        speed = np.full(n, np.nan, dtype=np.float32)
        for t in range(1, n):
            if (
                det[t] and det[t - 1]
                and np.isfinite(tips[t]).all()
                and np.isfinite(tips[t - 1]).all()
            ):
                speed[t] = float(np.linalg.norm(tips[t] - tips[t - 1]))

        lo = max(int(approach_window), 1)
        hi = n - max(int(hold_window), 1)
        cands: List[Tuple[float, float, int]] = []
        t_hi = max(hi, lo + 1)
        for t in range(lo, min(t_hi, n)):
            if not det[t] or not np.isfinite(ap[t]):
                continue
            w0, w1 = max(0, t - 3), min(n, t + 4)
            win = ap[w0:w1]
            if np.isfinite(win).any() and float(ap[t]) > float(np.nanmin(win)) + 1e-4:
                continue
            speed_before = float(np.nanmean(speed[max(0, t - lo):t]))
            speed_after = float(np.nanmean(speed[t:min(n, t + int(hold_window))]))
            if not np.isfinite(speed_before) or not np.isfinite(speed_after):
                continue
            score = (speed_before - speed_after) / (float(ap[t]) + 1e-3)
            cands.append((score, -float(ap[t]), t))
        if cands:
            cands.sort(reverse=True)
            return int(cands[0][2])

        sl0 = int(0.1 * n)
        sl1 = max(int(0.9 * n), sl0 + 1)
        valid = det & np.isfinite(ap)
        mid = valid.copy()
        mid[:sl0] = False
        mid[sl1:] = False
        pick = mid if mid.any() else valid
        if not pick.any():
            return None
        return int(np.flatnonzero(pick)[int(np.argmin(ap[pick]))])

    @staticmethod
    def _find_transport_window(
        detected: np.ndarray,
        tips_xyz: np.ndarray,
        move_vel: float = 0.01,
        move_run: int = 4,
        smooth: int = 3,
        min_net: float = 0.05,
    ) -> Optional[Tuple[int, int]]:
        """Hand-transport window: a long enough high-speed run with large net disp.

        Among runs of fingertip speed > ``move_vel`` lasting ≥ ``move_run``
        frames, pick the one with the largest 3D net displacement (the carry),
        not the longest fidget/approach. Returns inclusive ``(w0, w1)``.
        """
        n = len(detected)
        det = np.asarray(detected, dtype=bool)
        tips = np.asarray(tips_xyz, dtype=np.float32)
        if n == 0 or not det.any():
            return None
        speed = np.zeros(n, dtype=np.float32)
        for t in range(1, n):
            if (
                det[t] and det[t - 1]
                and np.isfinite(tips[t]).all()
                and np.isfinite(tips[t - 1]).all()
            ):
                speed[t] = float(np.linalg.norm(tips[t] - tips[t - 1]))
        k = max(int(smooth), 1)
        if k > 1:
            kernel = np.ones(k, dtype=np.float32) / float(k)
            speed_s = np.convolve(speed, kernel, mode="same")
        else:
            speed_s = speed
        moving = det & (speed_s > float(move_vel))
        best: Optional[Tuple[float, int, int, int]] = None  # net, run, start, end
        t = 0
        min_run = max(int(move_run), 1)
        while t < n:
            if not moving[t]:
                t += 1
                continue
            s = t
            while t < n and moving[t]:
                t += 1
            run = t - s
            e = t - 1
            if run < min_run:
                continue
            if not (np.isfinite(tips[s]).all() and np.isfinite(tips[e]).all()):
                continue
            net = float(np.linalg.norm(tips[e] - tips[s]))
            if net < float(min_net):
                continue
            if best is None or net > best[0] + 1e-6 or (abs(net - best[0]) <= 1e-6 and run > best[1]):
                best = (net, run, s, e)
        if best is None:
            return None
        return int(best[2]), int(best[3])

    @staticmethod
    def _moving_hand_frames(
        detected: np.ndarray,
        tips_xyz: np.ndarray,
        move_vel: float = 0.01,
        move_run: int = 4,
        smooth: int = 3,
    ) -> np.ndarray:
        """Boolean mask of frames that sit in *any* long-enough moving run.

        Unlike ``_find_transport_window`` this does not pick a single max-net
        window, so an empty-hand reach and a later carry are both kept.
        """
        n = len(detected)
        det = np.asarray(detected, dtype=bool)
        tips = np.asarray(tips_xyz, dtype=np.float32)
        out = np.zeros(n, dtype=bool)
        if n == 0 or not det.any():
            return out
        speed = np.zeros(n, dtype=np.float32)
        for t in range(1, n):
            if (
                det[t] and det[t - 1]
                and np.isfinite(tips[t]).all()
                and np.isfinite(tips[t - 1]).all()
            ):
                speed[t] = float(np.linalg.norm(tips[t] - tips[t - 1]))
        k = max(int(smooth), 1)
        if k > 1:
            kernel = np.ones(k, dtype=np.float32) / float(k)
            speed_s = np.convolve(speed, kernel, mode="same")
        else:
            speed_s = speed
        moving = det & (speed_s > float(move_vel))
        min_run = max(int(move_run), 1)
        t = 0
        while t < n:
            if not moving[t]:
                t += 1
                continue
            s = t
            while t < n and moving[t]:
                t += 1
            if t - s >= min_run:
                out[s:t] = True
        return out

    @staticmethod
    def _sample_global_candidates(
        detected: np.ndarray,
        moving: np.ndarray,
        stride: int = 5,
        max_n: int = 24,
    ) -> List[int]:
        """Stride through moving-hand frames (fallback: all detected frames)."""
        det = np.asarray(detected, dtype=bool)
        mov = np.asarray(moving, dtype=bool) & det
        src = mov if int(mov.sum()) >= 3 else det
        idx = np.flatnonzero(src)
        if len(idx) == 0:
            return []
        stride = max(int(stride), 1)
        cands = [int(i) for i in idx[::stride]]
        last = int(idx[-1])
        if last not in cands:
            cands.append(last)
        max_n = max(int(max_n), 1)
        if len(cands) > max_n:
            pick = np.unique(np.linspace(0, len(cands) - 1, num=max_n, dtype=int))
            cands = [cands[int(i)] for i in pick]
        return cands

    @staticmethod
    def _sample_candidate_frames(
        w0: int, w1: int, k: int, detected: np.ndarray,
    ) -> List[int]:
        """Equally spaced frames in ``[w0, w1]``, snapped onto detected frames."""
        w0, w1 = int(w0), int(w1)
        if w1 < w0:
            return []
        det = np.asarray(detected, dtype=bool)
        n = len(det)
        w0 = max(0, min(w0, n - 1))
        w1 = max(0, min(w1, n - 1))
        n_k = max(int(k), 1)
        raw = np.unique(np.linspace(w0, w1, num=n_k, dtype=int))
        out: List[int] = []
        det_idx = np.flatnonzero(det)
        for i in raw:
            i = int(i)
            if 0 <= i < n and det[i]:
                if i not in out:
                    out.append(i)
                continue
            if len(det_idx) == 0:
                continue
            j = int(det_idx[np.argmin(np.abs(det_idx - i))])
            if w0 <= j <= w1 and j not in out:
                out.append(j)
        return out

    @staticmethod
    def _score_probe_window(
        obj_xyz: np.ndarray,
        tips_xyz: np.ndarray,
        areas: Optional[np.ndarray] = None,
        frame_hw: Optional[Tuple[int, int]] = None,
        max_area_frac: float = 0.20,
        min_pts: int = 50,
        static_vel_max: float = 0.02,
        w_static: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Score a short probe: (score, vel_corr, nan_frac).

        ``score = corr − nan_frac − area_penalty + w_static · static_bonus``.
        ``static_bonus`` is +1 if the blob is still in part of the window and
        moving in another (picked-up object), −0.5 if it is mostly static
        (table distractor), 0 if it moves throughout (mid-carry; co-motion
        already ranks these).
        """
        obj = np.asarray(obj_xyz, dtype=np.float32)
        tips = np.asarray(tips_xyz, dtype=np.float32)
        n = min(len(obj), len(tips))
        if n < 3:
            return -1.0, float("nan"), 1.0
        obj, tips = obj[:n], tips[:n]
        obj_vel = np.full((n, 3), np.nan, dtype=np.float32)
        tip_vel = np.full((n, 3), np.nan, dtype=np.float32)
        for t in range(1, n):
            if np.isfinite(obj[t]).all() and np.isfinite(obj[t - 1]).all():
                obj_vel[t] = obj[t] - obj[t - 1]
            if np.isfinite(tips[t]).all() and np.isfinite(tips[t - 1]).all():
                tip_vel[t] = tips[t] - tips[t - 1]
        corr = IntentProcessor._vel_corr(obj_vel[1:], tip_vel[1:])
        nan_frac = float(1.0 - np.isfinite(obj).all(axis=1).mean())
        score = (float(corr) if np.isfinite(corr) else -1.0) - nan_frac
        if areas is not None and len(areas) >= n:
            a = np.asarray(areas[:n], dtype=np.float32)
            pos = a[a >= float(min_pts)]
            med = float(np.median(pos)) if len(pos) else 0.0
            if med < float(min_pts):
                score -= 1.0
            if frame_hw is not None:
                h, w = frame_hw
                if med > float(max_area_frac) * float(h) * float(w):
                    score -= 1.0
        if float(w_static) != 0.0:
            sp = np.linalg.norm(obj_vel[1:], axis=1)
            finite = np.isfinite(sp)
            if int(finite.sum()) >= 3:
                n_static = int((sp[finite] < float(static_vel_max)).sum())
                n_move = int(finite.sum()) - n_static
                if n_move >= 2 and n_static >= 2:
                    static_bonus = 1.0
                elif n_static > n_move:
                    static_bonus = -0.5
                else:
                    static_bonus = 0.0
                score += float(w_static) * static_bonus
        return float(score), float(corr) if np.isfinite(corr) else float("nan"), nan_frac

    def _try_hand_seed(
        self,
        paths: Paths,
        frames: np.ndarray,
        hand: dict,
        t: int,
    ) -> Tuple[Optional[np.ndarray], float, float]:
        """SAM seed at frame ``t`` with hand-region negatives + overlap veto.

        Returns ``(mask, sam_score, overlap)``. ``mask`` is None if empty or
        the blob is the hand.
        """
        h, w = frames.shape[1:3]
        region = self._hand_region_for_frame(paths, hand, t, (h, w))
        mask, _, score = self._seed_mask_hand_conditioned(
            frames[t], hand["kpts_2d"][t], region,
        )
        if mask is None:
            return None, 0.0, 1.0
        ov = self._mask_overlap_frac(mask, region)
        return mask, float(score), float(ov)

    def _vlm_grounded_mask(
        self,
        frame: np.ndarray,
        nouns: List[str],
        hand_uv: Optional[np.ndarray],
        hw: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
        """Open-vocab box near the pinch → SAM2 image mask.

        Tries ``nouns`` in order (manipulated object first). Returns
        ``(mask, det_score, noun)`` or ``(None, 0, None)`` if no in-hand box.
        """
        h, w = int(hw[0]), int(hw[1])
        max_area = float(self.track_max_area_frac) * float(h) * float(w)
        for noun in nouns:
            name = str(noun).strip()
            if not name:
                continue
            bboxes, scores = self.detector.get_bboxes(
                frame, name, threshold=self.dino_threshold,
            )
            picked = self._pick_inhand_box(
                bboxes, scores, hand_uv, self.track_hand_px, max_area,
            )
            if picked is None:
                continue
            bbox, det_score = picked
            mask = self._sam_box_mask(frame, bbox)
            if mask is None:
                continue
            if int(mask.sum()) > max_area:
                logger.info(
                    "[intent] vlm %r in-hand mask area=%d too large; skip",
                    name, int(mask.sum()),
                )
                continue
            return mask.astype(np.uint8), float(det_score), name
        return None, 0.0, None

    def _candidate_seed_mask(
        self,
        paths: Paths,
        frames: np.ndarray,
        hand: dict,
        t: int,
        nouns: Optional[List[str]] = None,
    ) -> Tuple[Optional[np.ndarray], float, float, str]:
        """VLM-grounded seed, then geometric hand-conditioned fallback.

        Returns ``(mask, score, overlap, source)`` where ``source`` is
        ``vlm:<noun>`` or ``geom`` (empty if both failed).
        """
        h, w = frames.shape[1:3]
        region = self._hand_region_for_frame(paths, hand, t, (h, w))
        grounding = str(getattr(self, "seed_grounding", "auto")).strip().lower()
        if nouns is None:
            nouns = self._get_object_nouns(paths)

        if grounding in ("auto", "vlm") and nouns:
            pinch = self._pinch_uv(hand["kpts_2d"][t])
            mask, score, noun = self._vlm_grounded_mask(
                frames[t], nouns, pinch, (h, w),
            )
            if mask is not None:
                cleaned, ov = self._apply_hand_overlap_veto(
                    mask, region,
                    max_overlap=float(getattr(self, "track_hand_overlap_max", 0.5)),
                    min_pts=self.min_object_pts,
                )
                if cleaned is not None:
                    return cleaned, float(score), float(ov), f"vlm:{noun}"
                logger.info(
                    "[intent] vlm t=%d noun=%s vetoed (hand overlap=%.2f)",
                    t, noun, ov,
                )

        if grounding in ("auto", "geom"):
            mask, score, ov = self._try_hand_seed(paths, frames, hand, t)
            if mask is not None:
                return mask, float(score), float(ov), "geom"
        return None, 0.0, 1.0, ""

    def _select_seed_by_verify(
        self,
        paths: Paths,
        frames: np.ndarray,
        depth: Optional[np.ndarray],
        hand: dict,
        video_dir: str,
    ) -> Tuple[Optional[int], Optional[np.ndarray], float, bool]:
        """Pick a seed by probing candidates across all moving-hand segments (v4).

        Each candidate tries a VLM in-hand box first (``intent_seed_grounding=auto``),
        then the geometric hand-conditioned mask. Returns
        ``(seed_idx, seed_mask, sam_score, low_conf)``. ``low_conf`` is True
        when we fell back to the v1 aperture heuristic.
        """
        n, h, w = frames.shape[:3]
        moving = self._moving_hand_frames(
            hand["detected"], hand["tips_rf"],
            move_vel=self.seed_move_vel, move_run=self.seed_move_run,
        )
        cands = self._sample_global_candidates(
            hand["detected"], moving, stride=self.seed_global_stride, max_n=24,
        )
        n_move = int(moving.sum())
        nouns = self._get_object_nouns(paths)
        logger.info(
            "[intent] hand_sam2 moving_frames=%d/%d stride=%d grounding=%s "
            "detector=%s nouns=%s candidates=%s",
            n_move, n, self.seed_global_stride, self.seed_grounding,
            self.seed_detector, nouns, cands,
        )
        if not cands:
            logger.info("[intent] hand_sam2: no global candidates; aperture fallback")
            return self._aperture_fallback_seed(paths, frames, hand, nouns=nouns)

        # Track VLM-grounded and geometric winners separately. A VLM candidate
        # is anchored on the *named* object; the geometric fallback re-segments
        # "what's in the hand" and, on reach frames, can lock onto fingertips
        # whose co-motion score beats the true (lower-corr) object box. So the
        # geometric winner is only used when NO candidate produced a VLM box.
        best_vlm: Optional[Tuple[float, int, np.ndarray, float, float, str]] = None
        best_geom: Optional[Tuple[float, int, np.ndarray, float, float, str]] = None
        win = max(int(self.seed_probe_win), 1)
        n_reject = 0
        for t in cands:
            mask, sam_score, ov, source = self._candidate_seed_mask(
                paths, frames, hand, t, nouns=nouns,
            )
            if mask is None:
                n_reject += 1
                logger.info(
                    "[intent] hand_sam2 candidate t=%d rejected (empty or hand overlap)", t,
                )
                continue
            lo, hi = max(0, t - win), min(n - 1, t + win)
            segs = self.sam2.propagate_mask_window(video_dir, mask, t, win)
            obj = np.full((n, 3), np.nan, dtype=np.float32)
            areas = np.zeros(n, dtype=np.float32)
            for i in range(lo, hi + 1):
                mi = segs.get(i, mask if i == t else None)
                if mi is None:
                    continue
                sm = self._squeeze_video_mask(mi)
                if sm.shape != (h, w):
                    sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
                areas[i] = float(sm.sum())
                if depth is not None:
                    obj[i] = self._mask_centroid_robot(sm, depth[min(i, len(depth) - 1)], i)
            score, corr, nan_frac = self._score_probe_window(
                obj[lo:hi + 1],
                hand["tips_rf"][lo:hi + 1],
                areas=areas[lo:hi + 1],
                frame_hw=(h, w),
                max_area_frac=self.track_max_area_frac,
                min_pts=self.min_object_pts,
                static_vel_max=self.track_static_vel_max,
                w_static=self.track_static_bonus_w,
            )
            logger.info(
                "[intent] hand_sam2 candidate t=%d source=%s score=%.3f corr=%s "
                "nan=%.2f area=%d hand_ov=%.2f",
                t, source, score,
                f"{corr:.2f}" if np.isfinite(corr) else "nan",
                nan_frac, int(mask.sum()), ov,
            )
            entry = (score, int(t), mask, float(sam_score), float(ov), source)
            if str(source).startswith("vlm"):
                if best_vlm is None or score > best_vlm[0]:
                    best_vlm = entry
            else:
                if best_geom is None or score > best_geom[0]:
                    best_geom = entry

        try:
            self.sam2.clear_video_state()
        except Exception:  # noqa: BLE001
            pass

        best = best_vlm if best_vlm is not None else best_geom
        if best is None:
            logger.warning(
                "[intent] hand_sam2: all %d candidates failed (%d hand/empty); aperture fallback",
                len(cands), n_reject,
            )
            return self._aperture_fallback_seed(paths, frames, hand, nouns=nouns)
        if best_vlm is not None and best_geom is not None and best_geom[0] > best_vlm[0]:
            logger.info(
                "[intent] hand_sam2 prefer VLM seed t=%d (%.3f) over higher-score "
                "geom t=%d (%.3f): geom fallback only when no VLM box",
                best_vlm[1], best_vlm[0], best_geom[1], best_geom[0],
            )

        score, t, mask, sam_score, ov, source = best
        self._seed_hand_overlap = float(ov)
        self._seed_source = str(source)
        logger.info(
            "[intent] hand_sam2 verify-picked t=%d source=%s probe_score=%.3f hand_ov=%.2f",
            t, source, score, ov,
        )
        return t, mask, sam_score, False

    def _aperture_fallback_seed(
        self, paths: Paths, frames: np.ndarray, hand: dict,
        nouns: Optional[List[str]] = None,
    ) -> Tuple[Optional[int], Optional[np.ndarray], float, bool]:
        seed_idx = self._select_seed_frame_from_hand(
            hand["detected"], hand["aperture"], hand["tips_rf"],
            approach_window=self.seed_approach_window,
            hold_window=self.seed_hold_window,
        )
        if seed_idx is None:
            self._seed_hand_overlap = 1.0
            self._seed_source = ""
            return None, None, 0.0, True
        mask, score, ov, source = self._candidate_seed_mask(
            paths, frames, hand, int(seed_idx), nouns=nouns,
        )
        self._seed_hand_overlap = float(ov) if mask is not None else 1.0
        self._seed_source = str(source) if mask is not None else ""
        return int(seed_idx), mask, float(score), True

    @staticmethod
    def _inhand_from_aperture(
        aperture: np.ndarray,
        detected: np.ndarray,
        seed_idx: int,
        slack: float = 0.03,
    ) -> np.ndarray:
        """Boolean in-hand window: the aperture run that contains ``seed_idx``."""
        n = len(aperture)
        inhand = np.zeros(n, dtype=bool)
        if seed_idx < 0 or seed_idx >= n:
            return inhand
        a_seed = float(aperture[seed_idx]) if np.isfinite(aperture[seed_idx]) else np.nan
        if not np.isfinite(a_seed):
            inhand[seed_idx] = True
            return inhand
        thr = a_seed + float(slack)
        ok = np.asarray(detected, dtype=bool) & np.isfinite(aperture) & (aperture <= thr)
        left = int(seed_idx)
        while left > 0 and ok[left - 1]:
            left -= 1
        right = int(seed_idx)
        while right < n - 1 and ok[right + 1]:
            right += 1
        inhand[left:right + 1] = True
        return inhand

    @staticmethod
    def _clip_uv(uv: np.ndarray, h: int, w: int) -> np.ndarray:
        pts = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        ok = (
            np.isfinite(pts).all(axis=1)
            & (pts[:, 0] >= 0) & (pts[:, 0] < w)
            & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        )
        return pts[ok]

    @staticmethod
    def _sample_mask_uv(
        mask: np.ndarray,
        n_pts: int,
        exclude_uv: Optional[np.ndarray] = None,
        exclude_r: float = 28.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        ys, xs = np.nonzero(np.asarray(mask).astype(bool))
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if exclude_uv is not None and len(np.asarray(exclude_uv).reshape(-1, 2)) and exclude_r > 0:
            ex = np.asarray(exclude_uv, dtype=np.float32).reshape(-1, 2)
            d2 = (xs[:, None] - ex[None, :, 0]) ** 2 + (ys[:, None] - ex[None, :, 1]) ** 2
            keep = d2.min(axis=1) > float(exclude_r) ** 2
            xs, ys = xs[keep], ys[keep]
            if len(xs) == 0:
                return np.zeros((0, 2), dtype=np.float32)
        rng = rng or np.random.default_rng(0)
        k = min(int(n_pts), len(xs))
        idx = rng.choice(len(xs), size=k, replace=False)
        return np.stack([xs[idx].astype(np.float32), ys[idx].astype(np.float32)], axis=1)

    def _seed_points_hand_conditioned(
        self,
        kpts_2d_t: np.ndarray,
        hand_mask: Optional[np.ndarray],
        frame_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Positive pinch points + negative palm / hand-mask points."""
        h, w = frame_hw
        k = np.asarray(kpts_2d_t, dtype=np.float32)
        thumb = k[THUMB_TIP_IDX, :2]
        index = k[INDEX_TIP_IDX, :2]
        pos = []
        if np.isfinite(thumb).all() and np.isfinite(index).all():
            mid = 0.5 * (thumb + index)
            pos.extend([mid, thumb * 0.35 + index * 0.65, thumb * 0.65 + index * 0.35])
        pos_uv = self._clip_uv(np.asarray(pos, dtype=np.float32), h, w) if pos else np.zeros((0, 2), np.float32)

        neg = []
        for i in range(len(k)):
            if i in TIP_IDXS:
                continue
            if np.isfinite(k[i, :2]).all():
                neg.append(k[i, :2])
        neg_uv = self._clip_uv(np.asarray(neg, dtype=np.float32), h, w) if neg else np.zeros((0, 2), np.float32)
        if pos_uv.size and neg_uv.size and self.track_neg_exclude_px > 0:
            d = np.linalg.norm(neg_uv[:, None, :] - pos_uv[None, :, :], axis=2).min(axis=1)
            neg_uv = neg_uv[d > float(self.track_neg_exclude_px)]
        if hand_mask is not None:
            extra = self._sample_mask_uv(
                hand_mask, n_pts=16, exclude_uv=pos_uv, exclude_r=self.track_neg_exclude_px,
            )
            if extra.size:
                neg_uv = np.concatenate([neg_uv, extra], axis=0) if neg_uv.size else extra
        return pos_uv, neg_uv

    def _filter_seed_mask(
        self, mask: np.ndarray, pos_uv: np.ndarray, frame_hw: Tuple[int, int], score: float,
    ) -> Optional[np.ndarray]:
        h, w = frame_hw
        m = np.asarray(mask)
        while m.ndim > 2:
            m = m[0]
        m = (m > 0).astype(np.uint8)
        area = int(m.sum())
        if area < self.min_object_pts:
            return None
        if area > float(self.track_max_area_frac) * h * w:
            logger.info("[intent] seed mask area=%d too large (frac>%.2f); reject", area, self.track_max_area_frac)
            return None
        if pos_uv.size:
            px, py = int(round(float(pos_uv[0, 0]))), int(round(float(pos_uv[0, 1])))
            if 0 <= py < h and 0 <= px < w and m[py, px] == 0:
                logger.info("[intent] seed mask misses pinch centre; reject (score=%.3f)", score)
                return None
        return m

    def _seed_mask_hand_conditioned(
        self,
        frame: np.ndarray,
        kpts_2d_t: np.ndarray,
        hand_mask: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
        """SAM2 image mask of 'what's in the hand' at one frame.

        Returns ``(mask, pinch_uv, score)``. ``pinch_uv`` is the 2D grasp centre
        used for logging / fallback box.
        """
        h, w = frame.shape[:2]
        pos_uv, neg_uv = self._seed_points_hand_conditioned(kpts_2d_t, hand_mask, (h, w))
        pinch = pos_uv[:1] if len(pos_uv) else np.zeros((0, 2), dtype=np.float32)
        if len(pos_uv) == 0:
            return None, pinch, 0.0
        mask, score = self.sam2.segment_points(frame, pos_uv, neg_uv, multimask_output=True)
        mask = self._filter_seed_mask(mask, pos_uv, (h, w), score)
        if mask is not None:
            mask, _ov = self._apply_hand_overlap_veto(
                mask, hand_mask,
                max_overlap=float(getattr(self, "track_hand_overlap_max", 0.5)),
                min_pts=self.min_object_pts,
            )
            if mask is not None:
                return mask, pinch, score
            logger.info("[intent] seed mask vetoed: overlaps hand region")

        # Small box around the pinch — still hand-anchored, not YOLO.
        c = pos_uv[0]
        r = 80.0
        box = np.array([c[0] - r, c[1] - r, c[0] + r, c[1] + r], dtype=np.float32)
        box[0] = max(0.0, box[0]); box[1] = max(0.0, box[1])
        box[2] = min(float(w - 1), box[2]); box[3] = min(float(h - 1), box[3])
        box_mask = self._sam_box_mask(frame, box)
        if box_mask is None:
            return None, pinch, 0.0
        box_mask = self._filter_seed_mask(box_mask, pos_uv, (h, w), 0.0)
        if box_mask is None:
            return None, pinch, 0.0
        box_mask, _ov = self._apply_hand_overlap_veto(
            box_mask, hand_mask,
            max_overlap=float(getattr(self, "track_hand_overlap_max", 0.5)),
            min_pts=self.min_object_pts,
        )
        return box_mask, pinch, 0.0

    @staticmethod
    def _squeeze_video_mask(mask: np.ndarray) -> np.ndarray:
        m = np.asarray(mask)
        while m.ndim > 2:
            m = m[0]
        return (m > 0).astype(np.uint8)

    def _propagate_video_masks(
        self,
        video_dir: str,
        pairs: List[Tuple[np.ndarray, int]],
        n: int,
        hw: Tuple[int, int],
    ) -> np.ndarray:
        h, w = hw
        out = np.zeros((n, h, w), dtype=np.uint8)
        for mask, idx in pairs:
            if 0 <= int(idx) < n:
                out[int(idx)] = self._squeeze_video_mask(mask)
        _, fwd = self.sam2.segment_video_from_masks(video_dir, pairs, reverse=False)
        _, bwd = self.sam2.segment_video_from_masks(video_dir, pairs, reverse=True)
        for segs in (fwd, bwd):
            for idx, obj in segs.items():
                if int(idx) < 0 or int(idx) >= n:
                    continue
                m = obj.get(0, None)
                if m is None:
                    continue
                sm = self._squeeze_video_mask(m)
                if sm.shape != (h, w):
                    sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
                out[int(idx)] = sm
        return out

    def _mask_centroid_robot(
        self, mask: np.ndarray, depth: np.ndarray, t: int,
    ) -> np.ndarray:
        """Median-depth backprojection of a 2D mask centroid, robot frame."""
        nan3 = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        m = np.asarray(mask).astype(bool)
        if int(m.sum()) < self.min_object_pts:
            return nan3
        depth_i = np.asarray(depth, dtype=np.float32)
        valid = m & np.isfinite(depth_i) & (depth_i > 1e-3) & (depth_i < self.depth_max)
        ys, xs = np.nonzero(valid)
        if len(xs) < self.min_object_pts:
            return nan3
        z = float(np.median(depth_i[ys, xs]))
        u, v = float(xs.mean()), float(ys.mean())
        fx = float(self.intrinsics_dict["fx"])
        fy = float(self.intrinsics_dict["fy"])
        cx = float(self.intrinsics_dict["cx"])
        cy = float(self.intrinsics_dict["cy"])
        pt_cam = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z], dtype=np.float32).reshape(1, 3)
        return transform_pts(pt_cam, self._T_cam2robot_at(t))[0].astype(np.float32)

    def _motion_gate_failed(
        self,
        masks: np.ndarray,
        depth: np.ndarray,
        tips_rf: np.ndarray,
        inhand: np.ndarray,
    ) -> np.ndarray:
        """Per-frame fail flags: empty, not attached in-hand, or drifting when free."""
        n = len(masks)
        failed = np.zeros(n, dtype=bool)
        obj = np.full((n, 3), np.nan, dtype=np.float32)
        for t in range(n):
            if int(masks[t].sum()) < self.min_object_pts:
                failed[t] = True
                continue
            obj[t] = self._mask_centroid_robot(masks[t], depth[min(t, len(depth) - 1)], t)

        attach = float(self.track_attach_err_max)
        static_v = float(self.track_static_vel_max)
        for t in range(1, n):
            if failed[t] or not np.isfinite(obj[t]).all() or not np.isfinite(obj[t - 1]).all():
                continue
            d_obj = obj[t] - obj[t - 1]
            if bool(inhand[t]) and np.isfinite(tips_rf[t]).all() and np.isfinite(tips_rf[t - 1]).all():
                pred = obj[t - 1] + (tips_rf[t] - tips_rf[t - 1])
                if float(np.linalg.norm(obj[t] - pred)) > attach:
                    failed[t] = True
            elif not bool(inhand[t]) and float(np.linalg.norm(d_obj)) > static_v:
                failed[t] = True
        return failed

    def _largest_fail_seed(
        self,
        failed: np.ndarray,
        detected: np.ndarray,
        used: set,
        inhand: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """First detected frame of the longest consecutive fail run not already used.

        Prefers fail runs that overlap the in-hand window so we don't reseed
        an open-hand approach frame with 'what's in the hand'.
        """
        n = len(failed)
        inhand_b = np.ones(n, dtype=bool) if inhand is None else np.asarray(inhand, dtype=bool)

        def _search(require_inhand: bool) -> Optional[int]:
            best_len, best_t = 0, None
            t = 0
            while t < n:
                if not failed[t]:
                    t += 1
                    continue
                s = t
                while t < n and failed[t]:
                    t += 1
                run = np.arange(s, t)
                det_run = [
                    int(i) for i in run
                    if bool(detected[i]) and int(i) not in used
                    and (not require_inhand or bool(inhand_b[i]))
                ]
                if det_run and (t - s) > best_len:
                    best_len, best_t = t - s, det_run[0]
            return best_t

        return _search(True)

    def _track_masks_hand_sam2(
        self,
        paths: Paths,
        frames: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], int, Optional[np.ndarray], float]:
        """Hand-conditioned SAM2 video tracking (category-agnostic)."""
        n, h, w = frames.shape[:3]
        self._mask_stack_cache = {}
        self._seed_hand_overlap = 0.0
        self._seed_source = ""
        hand = self._load_target_hand_kpts(paths, n, (h, w))
        if hand is None:
            return None, 0, None, 0.0

        folder = paths.original_images_folder
        if not os.path.exists(folder) or len([f for f in os.listdir(folder) if f.endswith(".jpg")]) == 0:
            self._extract_frames_cv2(paths.video_left, folder, square=self.square)
        video_dir = str(folder)
        nouns = self._get_object_nouns(paths)

        seed_idx, seed_mask, seed_score, low_conf = self._select_seed_by_verify(
            paths, frames, depth, hand, video_dir,
        )
        self._seed_low_conf = bool(low_conf)
        if seed_idx is None or seed_mask is None:
            logger.warning("[intent] hand_sam2: no seed frame from verify/fallback")
            return None, 0, None, 0.0
        logger.info(
            "[intent] hand_sam2 seed t=%d source=%s sam=%.3f area=%d low_conf=%s",
            seed_idx, getattr(self, "_seed_source", ""), seed_score,
            int(seed_mask.sum()), low_conf,
        )
        pairs: List[Tuple[np.ndarray, int]] = [(seed_mask, int(seed_idx))]
        used = {int(seed_idx)}
        masks = self._propagate_video_masks(video_dir, pairs, n, (h, w))

        inhand = self._inhand_from_aperture(hand["aperture"], hand["detected"], seed_idx)
        if depth is not None:
            for _ in range(max(0, int(self.track_reseed_max))):
                failed = self._motion_gate_failed(masks, depth, hand["tips_rf"], inhand)
                n_fail = int(failed.sum())
                if n_fail == 0:
                    break
                reseed_t = self._largest_fail_seed(
                    failed, hand["detected"], used, inhand=inhand,
                )
                if reseed_t is None:
                    logger.info("[intent] hand_sam2 gate fail=%d/%d; no unused hand frame", n_fail, n)
                    break
                rmask, rscore, rov, rsrc = self._candidate_seed_mask(
                    paths, frames, hand, int(reseed_t), nouns=nouns,
                )
                if rmask is None:
                    used.add(int(reseed_t))
                    continue
                logger.info(
                    "[intent] hand_sam2 reseed t=%d source=%s score=%.3f area=%d "
                    "hand_ov=%.2f (gate fail=%d)",
                    reseed_t, rsrc, rscore, int(rmask.sum()), rov, n_fail,
                )
                pairs.append((rmask, int(reseed_t)))
                used.add(int(reseed_t))
                masks = self._propagate_video_masks(video_dir, pairs, n, (h, w))

            failed = self._motion_gate_failed(masks, depth, hand["tips_rf"], inhand)
            if failed.any():
                n_zero = 0
                for t in np.flatnonzero(failed):
                    if int(masks[t].sum()) < self.min_object_pts:
                        continue
                    # Drop drifted frames rather than holding a table remnant.
                    masks[t] = 0
                    n_zero += 1
                logger.info("[intent] hand_sam2 zeroed %d drifted frames after gate", n_zero)

        n_valid = int((masks.reshape(n, -1).sum(axis=1) >= self.min_object_pts).sum())
        bbox = None
        if int(seed_mask.sum()) > 0:
            ys, xs = np.nonzero(seed_mask)
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
        logger.info(
            "[intent] hand_sam2 tracked %d/%d valid  seeds=%s",
            n_valid, n, [p[1] for p in pairs],
        )
        if n_valid == 0:
            return None, int(seed_idx), bbox, float(seed_score)
        return masks, int(seed_idx), bbox, float(seed_score)

    @staticmethod
    def _vel_corr(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two (N, 3) velocity sequences (flattened).

        Pearson is undefined when either trajectory is constant (e.g. a
        steady in-hand carry). Cosine still reports 1 when the object
        translates with the fingertips.
        """
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        m = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        if int(m.sum()) < 5:
            return float("nan")
        aa = a[m].reshape(-1)
        bb = b[m].reshape(-1)
        den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if den < 1e-12:
            return float("nan")
        return float(np.dot(aa, bb) / den)

    def _track_quality_gate(
        self,
        object_masks: np.ndarray,
        pcd_result: Dict[str, list],
        contact_result: Dict[str, np.ndarray],
        hands: Dict[str, dict],
        seed_idx: int,
    ) -> dict:
        """Automatic track QA (design §4). Does not abort the demo."""
        n = len(object_masks)
        areas = object_masks.reshape(n, -1).sum(axis=1).astype(np.int64)
        post = areas[max(int(seed_idx), 0):]
        lock_frac = 0.0
        if len(post) > 10:
            _, counts = np.unique(post, return_counts=True)
            lock_frac = float(counts.max()) / float(len(post))

        centroids = np.asarray(pcd_result["centroids_robot"], dtype=np.float32)
        valid = np.asarray(pcd_result["valid"], dtype=bool)
        obj_vel = np.full((n, 3), np.nan, dtype=np.float32)
        for t in range(1, n):
            if valid[t] and valid[t - 1]:
                obj_vel[t] = centroids[t] - centroids[t - 1]

        use = self._hands_for_contact(hands)
        tip_vel = np.full((n, 3), np.nan, dtype=np.float32)
        for t in range(1, n):
            best = None
            for h in use.values():
                if h["detected"][t] and h["detected"][t - 1]:
                    d = h["fingertips"][t].mean(axis=0) - h["fingertips"][t - 1].mean(axis=0)
                    if best is None or float(np.linalg.norm(d)) > float(np.linalg.norm(best)):
                        best = d
            if best is not None:
                tip_vel[t] = best

        phase = np.asarray(contact_result["phase"])
        attached = np.isin(phase, [PHASE_GRASP, PHASE_TRANSPORT])
        corr = self._vel_corr(obj_vel[attached], tip_vel[attached]) if attached.any() else float("nan")

        grasp_disp = float("nan")
        if int((attached & valid).sum()) >= 2:
            c = centroids[attached & valid]
            grasp_disp = float(np.nanmax(np.linalg.norm(c - c[0], axis=1)))

        free = ~attached & valid
        static_disp = float("nan")
        static_vmax = float("nan")
        if int(free.sum()) >= 3:
            cfree = centroids[free]
            med = np.nanmedian(cfree, axis=0)
            static_disp = float(np.nanmax(np.linalg.norm(cfree - med, axis=1)))
            sp = np.linalg.norm(obj_vel[free], axis=1)
            if np.isfinite(sp).any():
                static_vmax = float(np.nanmax(sp))

        n_valid = int((areas >= self.min_object_pts).sum())
        reasons: List[str] = []
        if n_valid < max(8, n // 10):
            reasons.append(f"valid_masks {n_valid}/{n}")
        if lock_frac >= 0.8:
            reasons.append(f"area_lock {lock_frac:.2f}")
        if np.isfinite(corr) and corr < float(self.track_motion_corr_min):
            reasons.append(f"inhand_corr {corr:.2f} < {self.track_motion_corr_min:.2f}")
        if np.isfinite(static_vmax) and static_vmax > 5.0 * float(self.track_static_vel_max):
            reasons.append(f"free_speed {static_vmax:.3f} m/frame")
        min_disp = float(getattr(self, "track_min_disp", 0.05))
        if not np.isfinite(grasp_disp) or grasp_disp < min_disp:
            reasons.append(
                f"grasp_disp {grasp_disp if np.isfinite(grasp_disp) else float('nan'):.3f} < {min_disp:.3f}"
            )
        seed_hand_ov = float(getattr(self, "_seed_hand_overlap", 0.0))
        max_ov = float(getattr(self, "track_hand_overlap_max", 0.5))
        if seed_hand_ov > max_ov:
            reasons.append(f"seed_hand_overlap {seed_hand_ov:.2f} > {max_ov:.2f}")

        qa = {
            "pass": len(reasons) == 0,
            "reasons": reasons,
            "n_valid": n_valid,
            "n": n,
            "seed_idx": int(seed_idx),
            "lock_frac": lock_frac,
            "inhand_corr": corr,
            "static_disp": static_disp,
            "static_vmax": static_vmax,
            "grasp_disp": grasp_disp,
            "seed_hand_overlap": seed_hand_ov,
            "seed_low_conf": bool(getattr(self, "_seed_low_conf", False)),
            "seed_source": str(getattr(self, "_seed_source", "")),
            "backend": self.track_backend,
        }
        if qa["pass"]:
            logger.info(
                "[intent] track QA PASS  valid=%d/%d lock=%.2f corr=%s grasp_disp=%s "
                "static_vmax=%s source=%s",
                n_valid, n, lock_frac,
                f"{corr:.2f}" if np.isfinite(corr) else "nan",
                f"{grasp_disp:.3f}" if np.isfinite(grasp_disp) else "nan",
                f"{static_vmax:.3f}" if np.isfinite(static_vmax) else "nan",
                qa.get("seed_source", ""),
            )
        else:
            logger.warning("[intent] track QA FAIL: %s", "; ".join(reasons))
        return qa

    def _save_track_quality(self, paths: Paths, qa: dict) -> None:
        os.makedirs(paths.intent_processor, exist_ok=True)
        np.savez(
            paths.track_quality,
            accept=bool(qa["pass"]),
            reasons=np.array(qa["reasons"], dtype=object),
            n_valid=int(qa["n_valid"]),
            n=int(qa["n"]),
            seed_idx=int(qa["seed_idx"]),
            lock_frac=float(qa["lock_frac"]),
            inhand_corr=float(qa["inhand_corr"]) if np.isfinite(qa["inhand_corr"]) else np.nan,
            static_disp=float(qa["static_disp"]) if np.isfinite(qa["static_disp"]) else np.nan,
            static_vmax=float(qa["static_vmax"]) if np.isfinite(qa["static_vmax"]) else np.nan,
            grasp_disp=float(qa.get("grasp_disp", np.nan)) if np.isfinite(qa.get("grasp_disp", np.nan)) else np.nan,
            seed_hand_overlap=float(qa.get("seed_hand_overlap", 0.0)),
            seed_low_conf=bool(qa.get("seed_low_conf", False)),
            seed_source=str(qa.get("seed_source", "")),
            backend=str(qa["backend"]),
        )

    def _track_masks(
        self,
        frames: np.ndarray,
        object_prompt: str,
        seed_idx: int,
        seed_bbox: np.ndarray,
        hand_uv: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Track the object with per-frame YOLO + single-frame SAM2 (``yolo`` backend).

        Seed frame is segmented from the YOLO box. Every ``track_stride``
        frames, YOLO is re-run and associated with the previous mask.
        Misses copy the last mask. The ``hand_sam2`` backend does not use
        this path (see ``_track_masks_hand_sam2``).
        """
        t_all, h, w = frames.shape[:3]
        stride = max(1, int(self.track_stride))
        masks = np.zeros((t_all, h, w), dtype=np.uint8)
        seed_mask = self._sam_box_mask(frames[seed_idx], seed_bbox)
        if seed_mask is None:
            logger.warning("[intent] SAM2 produced an empty seed mask; tracking will hold zeros")
            return masks
        masks[seed_idx] = seed_mask
        counts = {"assoc": 0, "reid": 0, "hold": 0, "hand": 0}

        def _walk(indices):
            prev = masks[seed_idx]
            step = 0
            for t in indices:
                step += 1
                if step % stride == 0:
                    uv_t = None if hand_uv is None else hand_uv[t]
                    prev, how = self._refresh_mask(
                        frames[t], object_prompt, prev, hand_uv=uv_t,
                    )
                    counts[how] = counts.get(how, 0) + 1
                masks[t] = prev

        _walk(range(seed_idx + 1, t_all))
        _walk(range(seed_idx - 1, -1, -1))
        n_valid = int((masks.reshape(t_all, -1).sum(axis=1) >= self.min_object_pts).sum())
        logger.info(
            "[intent] tracked masks %d/%d valid  stride=%d  assoc=%d reid=%d hold=%d hand=%d",
            n_valid, t_all, stride,
            counts["assoc"], counts["reid"], counts["hold"], counts["hand"],
        )
        return masks

    def _propagate_mask(
        self, paths: Paths, seed_idx: int, seed_bbox: np.ndarray, n_frames: int
    ) -> np.ndarray:
        """Deprecated alias: long-range SAM2 video propagate. Unused by intent."""
        logger.warning("[intent] _propagate_mask is deprecated; use _track_masks")
        frames = self._load_frames(paths)
        prompt = self._get_object_prompt(paths)
        return self._track_masks(frames[:n_frames], prompt, seed_idx, seed_bbox)

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
            pts_rf = transform_pts(pts_cam, self._T_cam2robot_at(i)).astype(np.float32)

            points_cam.append(pts_cam)
            points_robot.append(pts_rf)
            colors.append(cols)
            centroids_robot[i] = pts_rf.mean(axis=0)
            valid[i] = True

        centroids_robot = self._smooth_centroids(centroids_robot, valid)
        logger.info(
            "[intent] built object point clouds: %d/%d frames valid (centroid smooth win=%d)",
            int(valid.sum()), n, self.centroid_smooth_win,
        )
        return {
            "points_cam": points_cam,
            "points_robot": points_robot,
            "colors": colors,
            "centroids_robot": centroids_robot,
            "valid": valid,
        }

    def _smooth_centroids(self, centroids: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Temporally denoise the object-centroid target track (edge-preserving).

        Small masks jitter in camera-Z; the raw per-frame depth-mean spikes the
        Stage B grasp/release ``key_pos``. We apply a **median** filter (not a
        moving average) of width ``intent_centroid_smooth_win`` so isolated
        depth spikes are removed while the grasp/release *turning points* — the
        very frames the gate scores — are preserved (a mean lags/overshoots
        there). Filtering is done **per contiguous valid run** so occlusion gaps
        are never blended across. Invalid frames are left NaN.
        """
        win = int(self.centroid_smooth_win)
        idx = np.where(valid)[0]
        if len(idx) < 3 or win < 2:
            return centroids
        half = max(win // 2, 1)
        out = centroids.copy()
        n_adj = 0
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for run in runs:
            if len(run) < 3:
                continue
            c = centroids[run].astype(np.float64)
            k = len(c)
            sm = np.empty_like(c)
            for j in range(k):
                lo, hi = max(0, j - half), min(k, j + half + 1)
                sm[j] = np.median(c[lo:hi], axis=0)
            n_adj += int((np.linalg.norm(sm - c, axis=1) > float(self.centroid_reject_m)).sum())
            out[run] = sm.astype(centroids.dtype)
        if n_adj:
            logger.info("[intent] centroid median-smooth: corrected %d frames (>%.0fcm shift)",
                        n_adj, float(self.centroid_reject_m) * 100)
        return out

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
            T_cam2robot_seq=self._T_c2r if self._T_c2r is not None else self.T_cam2robot[None],
            T_place=self._T_place if self._T_place is not None else np.eye(4),
        )

        if contact_result is not None:
            np.savez(
                paths.contact_events,
                phase=contact_result["phase"],
                phase_names=contact_result["phase_names"],
                g_closed=contact_result["g_closed"],
                contact=contact_result["contact"],
                d_finger_obj=contact_result["d_finger_obj"],
                d_eucl=contact_result.get("d_eucl", contact_result["d_finger_obj"]),
                d_xy=contact_result.get("d_xy", np.full_like(contact_result["d_finger_obj"], np.nan)),
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
            d_eu = contact_result.get("d_eucl")
            if np.isfinite(d).any():
                ax0.plot(x, d, color="k", lw=1.2, label="contact score (m)")
                ax0.axhline(self.contact_dist_in, color="g", ls="--", lw=0.8, label="dist_in")
                ax0.axhline(self.contact_dist_out, color="r", ls="--", lw=0.8, label="dist_out")
            if d_eu is not None and np.isfinite(d_eu).any():
                ax0.plot(x, d_eu, color="0.45", lw=0.8, ls=":", label="euclidean (m)")
            dxy = contact_result.get("d_xy")
            if dxy is not None and np.isfinite(dxy).any():
                ax0.plot(x, dxy, color="C1", lw=0.8, alpha=0.7, label="d_xy (m)")
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
    def _hands_for_contact(self, hands: Dict[str, dict]) -> Dict[str, dict]:
        """Prefer ``target_hand`` so an idle table-hand cannot steal contact."""
        th = str(getattr(self, "target_hand", "") or "").lower()
        if th in ("left", "right") and th in hands:
            return {th: hands[th]}
        return hands

    @staticmethod
    def _wrap_aware_score(
        fp: np.ndarray, obj_pts: np.ndarray, xy_inflate: float,
    ) -> Tuple[float, float, float]:
        """Contact score for one fingertip vs a cloud in the *same* frame.

        Inflates the visible cloud by ``xy_inflate`` in XY (cm-scale wrap
        around an occluded contact face) then takes the remaining
        ``hypot(dxy_extra, Δz)``. ``xy_inflate <= 0`` is pure Euclidean.
        A hard |Δz|-if-inside-radius switch is avoided: it cliffs back to
        Euclidean as soon as dxy exceeds the radius and chops the transport
        segment of a wrap grasp.
        """
        delta = obj_pts - fp.reshape(1, 3)
        dist = np.linalg.norm(delta, axis=1)
        j = int(np.argmin(dist))
        d_eucl = float(dist[j])
        dxy = float(np.hypot(delta[j, 0], delta[j, 1]))
        dz = float(abs(delta[j, 2]))
        extra = max(dxy - max(xy_inflate, 0.0), 0.0)
        score = float(np.hypot(extra, dz))
        return score, d_eucl, dxy

    def _detect_contacts(
        self, pcd_result: Dict[str, list], hands: Dict[str, dict]
    ) -> Dict[str, np.ndarray]:
        """Detect grasp/release and segment free/grasp/transport/release phases.

        Primary cue: wrap-aware fingertip-object distance. Visible top-face
        clouds miss the contact patch of a wrap grasp (EgoDex stapler: 4.7 cm
        in camera X, 0.3 cm in Z). When the nearest point is within
        ``intent_contact_xy_inflate`` in XY, the score is |Δz| (camera frame
        if clouds/fingertips are available, else robot frame). Pure Euclidean
        is kept for diagnostics and used when wrap is off.

        No hand keypoints → object-motion fallback.
        """
        n = len(pcd_result["valid"])
        points_robot = pcd_result["points_robot"]
        points_cam = pcd_result.get("points_cam")
        centroids = pcd_result["centroids_robot"]
        obj_valid = np.asarray(pcd_result["valid"], dtype=bool)
        xy_inf = float(getattr(self, "contact_xy_inflate", 0.0) or 0.0)
        use_hands = self._hands_for_contact(hands)

        d_finger_obj = np.full(n, np.nan, dtype=np.float32)
        d_eucl = np.full(n, np.nan, dtype=np.float32)
        d_xy = np.full(n, np.nan, dtype=np.float32)
        aperture = np.full(n, np.nan, dtype=np.float32)
        for t in range(n):
            if not obj_valid[t] or len(points_robot[t]) == 0:
                continue
            obj_rf = points_robot[t]
            obj_cam = None
            if points_cam is not None and t < len(points_cam) and len(points_cam[t]) > 0:
                obj_cam = points_cam[t]
            best = (np.inf, np.inf, np.inf, np.nan)  # score, eucl, dxy, aperture
            for h in use_hands.values():
                if not h["detected"][t]:
                    continue
                fps_rf = h["fingertips"][t]
                fps_cam = h.get("fingertips_cam")
                use_cam = obj_cam is not None and fps_cam is not None
                fps = fps_cam[t] if use_cam else fps_rf
                obj = obj_cam if use_cam else obj_rf
                scores = [
                    self._wrap_aware_score(fp, obj, xy_inf) for fp in fps
                ]
                si = int(np.argmin([s[0] for s in scores]))
                score, eucl, lat = scores[si]
                if score < best[0]:
                    best = (score, eucl, lat, float(h["aperture"][t]))
            if np.isfinite(best[0]):
                d_finger_obj[t] = best[0]
                d_eucl[t] = best[1]
                d_xy[t] = best[2]
                aperture[t] = best[3]

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
            if xy_inf > 0.0:
                source = "fingertip_wrap"
        else:
            contact = self._motion_contact(obj_speed, self.contact_motion_thresh)
            source = "object_motion"
            logger.warning(
                "[intent] no hand keypoints available (%s missing); using "
                "object-motion contact cue", "hand_data_*",
            )
        contact = self._filter_min_run(contact, self.contact_min_run)

        phase, g_state, grasp_kf, release_kf = self._segment_phases(contact)

        n_wrap = int(np.nansum(
            np.isfinite(d_finger_obj) & np.isfinite(d_eucl) & (d_finger_obj < d_eucl - 1e-4)
        ))
        logger.info(
            "[intent] contact source=%s grasp_kf=%s release_kf=%s contact_frames=%d/%d"
            "  d_wrap min=%.3f d_eucl min=%.3f xy_inflate=%.3f wrap_frames=%d",
            source, grasp_kf, release_kf, int(contact.sum()), n,
            float(np.nanmin(d_finger_obj)) if np.isfinite(d_finger_obj).any() else np.nan,
            float(np.nanmin(d_eucl)) if np.isfinite(d_eucl).any() else np.nan,
            xy_inf, n_wrap,
        )
        return {
            "phase": phase,
            "phase_names": np.array([PHASE_NAMES[int(p)] for p in phase], dtype=object),
            "g_closed": g_state,
            "contact": contact,
            "d_finger_obj": d_finger_obj,
            "d_eucl": d_eucl,
            "d_xy": d_xy,
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
            fps_cam = np.zeros((n, len(FINGERTIP_IDXS), 3), dtype=np.float32)
            fps_cam[:m] = kpts_cam[:m][:, FINGERTIP_IDXS, :].astype(np.float32)
            hands[side] = {
                "fingertips": fingertips,
                "fingertips_cam": fps_cam,
                "detected": det,
                "aperture": aperture,
                "kpts_rf": kpts_rf[:m].astype(np.float32),
            }
            logger.info("[intent] loaded %s hand: %d/%d detected", side, int(det.sum()), n)
        return hands

    def _to_robot_frame(self, kpts_cam: np.ndarray) -> np.ndarray:
        """Transform (N,21,3) camera-frame keypoints to robot frame.

        Uses per-frame ``T_cam2robot(t)`` when EgoDex ``T_camera`` is loaded,
        otherwise the constant calibration matrix.
        """
        n = len(kpts_cam)
        pts_h = np.concatenate(
            [kpts_cam, np.ones((*kpts_cam.shape[:2], 1), dtype=kpts_cam.dtype)], axis=-1
        )
        if self._T_c2r is None:
            return np.einsum("ij,bpj->bpi", self.T_cam2robot, pts_h)[..., :3]
        T = np.stack([self._T_cam2robot_at(i) for i in range(n)], axis=0)
        return np.einsum("nij,npj->npi", T, pts_h)[..., :3]

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

    @staticmethod
    def _hand_rot_from_kpts(k3_rf_t: np.ndarray) -> Optional[np.ndarray]:
        """Approximate a hand-aligned gripper frame from 21 robot-frame keypoints."""
        k = np.asarray(k3_rf_t, dtype=np.float32).reshape(-1, 3)
        need = [0, 5, 17, THUMB_TIP_IDX, INDEX_TIP_IDX]
        if any(i >= len(k) for i in need):
            return None
        if not np.isfinite(k[need]).all():
            return None
        wrist = k[0]
        index_mcp = k[5]
        pinky_mcp = k[17]
        thumb_tip = k[THUMB_TIP_IDX]
        index_tip = k[INDEX_TIP_IDX]

        closing = thumb_tip - index_tip
        if float(np.linalg.norm(closing)) < 1e-6:
            return None
        approach = np.cross(index_mcp - wrist, pinky_mcp - wrist)
        if float(np.linalg.norm(approach)) < 1e-6:
            return None
        return IntentProcessor._build_grasp_rotation(approach, closing).astype(np.float32)

    @staticmethod
    def _hold_rotations(R_seq: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Nearest-valid hold for a rotation sequence."""
        R_seq = np.asarray(R_seq, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        if len(R_seq) == 0 or not valid.any():
            return R_seq, valid
        out = R_seq.copy()
        idx = np.flatnonzero(valid)
        first = int(idx[0])
        out[:first] = R_seq[first]
        last_seen = R_seq[first]
        for t in range(first, len(out)):
            if valid[t]:
                last_seen = R_seq[t]
            else:
                out[t] = last_seen
        out[int(idx[-1]) + 1:] = R_seq[int(idx[-1])]
        return out, np.ones(len(out), dtype=bool)

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
            ``[grasp_kf, release_kf]`` it is expressed **object-relative**:
            ``offset`` is locked at grasp onset (human thumb-index midpoint
            minus the object centroid when ``intent_grasp_offset=hand``, else
            ``G_center - centroid``) and re-applied to each frame's centroid.
            When hand pose is available, that offset rotates with the hand's
            relative rotation from the grasp frame.
          * ``R_target`` — EE orientation target. Anchored at the object grasp
            rotation ``G_rot`` and, when hand pose is available, rotated by the
            same hand-relative rotation so release targets reflect the real pose.
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
        R_rel = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        R_rel_valid = np.zeros(n, dtype=bool)
        use_hands = self._hands_for_contact(hands)

        hand_pose = next(iter(use_hands.values()), None) if use_hands else None
        if hand_pose is not None and "kpts_rf" in hand_pose:
            hand_rots = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
            hand_rot_valid = np.zeros(n, dtype=bool)
            kpts_rf = np.asarray(hand_pose["kpts_rf"], dtype=np.float32)
            det = np.asarray(hand_pose.get("detected", np.zeros(n, dtype=bool)), dtype=bool)
            for t in range(min(n, len(kpts_rf))):
                if t < len(det) and not det[t]:
                    continue
                Rt = self._hand_rot_from_kpts(kpts_rf[t])
                if Rt is None:
                    continue
                hand_rots[t] = Rt
                hand_rot_valid[t] = True
            if 0 <= grasp_kf < n and hand_rot_valid[grasp_kf]:
                hand_rots, hand_rot_valid = self._hold_rotations(hand_rots, hand_rot_valid)
                R_ref_inv = hand_rots[grasp_kf].T
                for t in range(n):
                    if hand_rot_valid[t]:
                        R_rel[t] = hand_rots[t] @ R_ref_inv
                R_rel_valid[:] = hand_rot_valid

        # Object-relative offset locked at grasp onset. Prefer the human hand
        # so p_t* is continuous with the free-phase hand-follow; fall back to
        # the antipodal G* when no hand is available.
        offset = None
        offset_mode = str(self.grasp_offset_from).strip().lower()
        if grasp_valid and 0 <= grasp_kf < n and obj_valid[grasp_kf] and offset_mode in (
            "hand", "ee", "fingertip", "auto",
        ):
            hand_pt = self._ee_target_from_hand(use_hands, grasp_kf)
            if hand_pt is not None:
                offset = np.asarray(hand_pt, dtype=np.float32) - centroids[grasp_kf]
        if offset is None and grasp_valid and 0 <= grasp_frame < n and obj_valid[grasp_frame]:
            offset = G_center - centroids[grasp_frame]

        ep_lo = grasp_kf if grasp_kf >= 0 else n
        ep_hi = release_kf if release_kf >= 0 else -1
        for t in range(n):
            in_episode = grasp_valid and offset is not None and ep_lo <= t <= ep_hi
            if in_episode and obj_valid[t]:
                offset_t = R_rel[t] @ offset if R_rel_valid[t] else offset
                p_target[t] = centroids[t] + offset_t
                R_target[t] = (R_rel[t] @ G_rot) if R_rel_valid[t] else G_rot
                p_valid[t] = True
                p_source[t] = "object_relative"
                continue
            # Outside the grasp episode (or missing object): follow the task hand.
            hand_pt = self._ee_target_from_hand(use_hands, t)
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

        Picks the detected hand with the smaller aperture (the one actively
        pinching). Callers should pass only ``target_hand`` via
        ``_hands_for_contact`` so an idle other-hand pinch cannot yank ``p_t*``.
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
        qa = getattr(self, "_last_track_qa", None)
        ax.set_title(
            f"intent EE target — grasp_valid={intent_result['grasp_valid']} "
            f"kf={int(intent_result['grasp_keyframe'])}/{int(intent_result['release_keyframe'])}"
            + (f"  trackQA={'PASS' if qa['pass'] else 'FAIL'}" if qa is not None else "")
        )
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.tight_layout()
        try:
            fig.savefig(str(paths.intent_preview), dpi=120, bbox_inches="tight")
        except Exception as e:  # pragma: no cover
            logger.warning("[intent] failed to write intent preview: %s", e)
        finally:
            plt.close(fig)
