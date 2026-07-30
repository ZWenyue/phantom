"""
Segmentation Processor Module

This module uses SAM2 to create masks of hands and arms in video sequences.

Processing Pipeline:
1. Load video frames and detection/pose data from previous stages
2. Initialize segmentation with highest-quality detection frame
3. Propagate segmentation bidirectionally (forward and reverse)
4. Combine temporal results for complete sequence coverage
5. Generate visualization videos and save segmentation masks

The module supports different segmentation modes:
- HandSegmentationProcessor: Precise hand-only segmentation
- ArmSegmentationProcessor: Combined hand + arm segmentation
"""

import os
import logging
import shutil
from tqdm import tqdm
import numpy as np
import mediapy as media
import argparse
from typing import Dict, Tuple, Optional, List

from phantom.processors.paths import Paths
from phantom.processors.base_processor import BaseProcessor
from phantom.detectors.detector_sam2 import DetectorSam2
from phantom.detectors.detector_detectron2 import DetectorDetectron2
from phantom.processors.phantom_data import HandSequence

logger = logging.getLogger(__name__)

# Configuration constants for segmentation processing
DEFAULT_FPS = 10
DEFAULT_CODEC = "ffv1"
ANNOTATION_CODEC = "h264"

class BaseSegmentationProcessor(BaseProcessor): 
    """
    Base class for video segmentation processing using SAM2.
    
    The base processor establishes the framework for temporal segmentation processing,
    where segmentation masks are propagated both forward and backward through time
    to ensure temporal consistency and complete coverage of the video sequence.
    
    Attributes:
        detector_sam (DetectorSam2): SAM2 segmentation model instance
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the base segmentation processor.
        
        Args:
            args: Command line arguments containing segmentation configuration
        """
        super().__init__(args)
        self.detector_sam = DetectorSam2()

    def process_one_demo(self, data_sub_folder: str) -> None:
        """
        Process a single demonstration - to be implemented by subclasses.
        
        Args:
            data_sub_folder: Path to demonstration data folder
            
        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _load_hamer_data(self, paths: Paths) -> Dict[str, HandSequence]:
        """
        Load hand pose estimation data from previous processing stage.
        
        Args:
            paths: Paths object containing file locations
            
        Returns:
            Dictionary containing left and right hand sequences
        """
        if self.bimanual_setup == "single_arm":
            if self.target_hand == "left":
                return {"left": HandSequence.load(paths.hand_data_left)}
            elif self.target_hand == "right":
                return {"right": HandSequence.load(paths.hand_data_right)}
            else:
                raise ValueError(f"Invalid target hand: {self.target_hand}")
        elif self.bimanual_setup == "shoulders":    
            return {
                "left": HandSequence.load(paths.hand_data_left),
                "right": HandSequence.load(paths.hand_data_right)
            }
        else:
            raise ValueError(f"Invalid bimanual setup: {self.bimanual_setup}")
    
    @staticmethod
    def _load_video(video_path: str) -> np.ndarray:
        """
        Load and validate video frames from disk.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Array of RGB video frames
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video file is empty or corrupted
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        imgs_rgb = media.read_video(video_path)
        if len(imgs_rgb) == 0:
            raise ValueError("Empty video file")
        
        return imgs_rgb
    
    @staticmethod
    def _load_bbox_data(bbox_path: str) -> Dict[str, np.ndarray]:
        """
        Load and validate bounding box detection data.
        
        Args:
            bbox_path: Path to bounding box data file
            
        Returns:
            Dictionary containing detection results from bounding box processor
            
        Raises:
            FileNotFoundError: If bounding box data file doesn't exist
        """
        if not os.path.exists(bbox_path):
            raise FileNotFoundError(f"Bbox data not found: {bbox_path}")
        
        return np.load(bbox_path)
    
    @staticmethod
    def _combine_sam_images(
        imgs_rgb: np.ndarray,
        imgs_forward: Dict[int, np.ndarray],
        imgs_reverse: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Combine forward and reverse SAM visualization images.
        
        This method merges the visualization results from bidirectional
        processing to create a complete visualization sequence.
        
        Args:
            imgs_rgb: Original RGB frames for shape reference
            imgs_forward: Forward propagation visualization results
            imgs_reverse: Reverse propagation visualization results
            
        Returns:
            Combined visualization array
        """
        result = np.zeros_like(imgs_rgb)
        # Fill in forward propagation results
        for idx in imgs_forward:
            result[idx] = imgs_forward[idx]
        # Fill in reverse propagation results (may overwrite forward results)
        for idx in imgs_reverse:
            result[idx] = imgs_reverse[idx]
        return result

    @staticmethod
    def _combine_masks(
        imgs_rgb: np.ndarray,
        masks_forward: Dict[int, np.ndarray],
        masks_reverse: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Combine forward and reverse segmentation masks.
        
        This method merges segmentation masks from bidirectional processing
        to ensure complete temporal coverage of the video sequence.
        
        Args:
            imgs_rgb: Original RGB frames for shape reference
            masks_forward: Forward propagation mask results
            masks_reverse: Reverse propagation mask results
            
        Returns:
            Combined mask array with shape (num_frames, height, width)
        """
        result = np.zeros((len(imgs_rgb), imgs_rgb[0].shape[0], imgs_rgb[0].shape[1]))
        for idx in masks_forward:
            result[idx] = masks_forward[idx][0]
        for idx in masks_reverse:
            result[idx] = masks_reverse[idx][0]
        return result

class ArmSegmentationProcessor(BaseSegmentationProcessor): 
    """
    Processor for segmenting combined hand and arm regions in video sequences.
    
    Attributes:
        detectron_detector (DetectorDetectron2): Detectron2 model for initial detection
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the arm segmentation processor with detection models.
        
        Args:
            args: Command line arguments containing model configuration
        """
        super().__init__(args)

        # Initialize Detectron2 for initial hand/arm detection
        root_dir = "../submodules/phantom-hamer/"
        self.detectron_detector = DetectorDetectron2(root_dir)


    def process_one_demo(self, data_sub_folder: str, hamer_data: Optional[Dict[str, HandSequence]] = None) -> None:
        """
        Process a single video demonstration to generate combined hand + arm segmentation masks.

        Uses Detectron2 instance segmentation (pred_masks) to detect person regions
        per frame, then filters by hand keypoint containment to keep only arm masks.

        Args:
            data_sub_folder: Path to the subfolder containing the demo data
            hamer_data: Optional pre-loaded hand pose data for segmentation guidance
        """
        save_folder = self.get_save_folder(data_sub_folder)
        paths = self.get_paths(save_folder)

        imgs_rgb = self._load_video(paths.video_left)
        bbox_data = self._load_bbox_data(paths.bbox_data)
        if hamer_data is None:
            hamer_data = self._load_hamer_data(paths)

        masks = self._get_detectron_arm_masks(imgs_rgb, bbox_data, hamer_data)

        sam_imgs = self._create_visualization(imgs_rgb, masks)
        self._validate_output_consistency(imgs_rgb, masks, sam_imgs)
        self._save_results(paths, masks, sam_imgs)

    def _get_detectron_arm_masks(
        self,
        imgs_rgb: np.ndarray,
        bbox_data: Dict[str, np.ndarray],
        hamer_data: Dict[str, HandSequence],
    ) -> np.ndarray:
        """
        Generate arm masks using Detectron2 instance segmentation.

        For each frame, runs Detectron2 to get person masks, then keeps only
        masks that contain hand keypoints from HaMeR. When no keypoints are
        available (hand not detected), all person masks are included as fallback.

        Args:
            imgs_rgb: RGB video frames, shape (N, H, W, 3)
            bbox_data: Bounding box data with hand detection flags
            hamer_data: Hand pose data with 2D keypoints

        Returns:
            Boolean mask array, shape (N, H, W)
        """
        n_frames = len(imgs_rgb)
        h, w = imgs_rgb[0].shape[:2]
        masks = np.zeros((n_frames, h, w), dtype=np.bool_)

        kpts_per_frame = self._collect_hand_keypoints(bbox_data, hamer_data)

        for idx in tqdm(range(n_frames), desc="Detectron2 arm segmentation"):
            pred_masks, _, _ = self.detectron_detector.get_person_masks(imgs_rgb[idx])
            if len(pred_masks) == 0:
                continue

            frame_kpts = kpts_per_frame[idx]
            if len(frame_kpts) == 0:
                for m in pred_masks:
                    masks[idx] |= m
            else:
                for m in pred_masks:
                    if self._mask_contains_any_keypoint(m, frame_kpts):
                        masks[idx] |= m

        return masks

    def _collect_hand_keypoints(
        self,
        bbox_data: Dict[str, np.ndarray],
        hamer_data: Dict[str, HandSequence],
    ) -> List[np.ndarray]:
        """
        Collect valid hand keypoints per frame from all available hands.

        Returns:
            List of arrays, one per frame. Each array has shape (K, 2) where K
            is the number of valid keypoints (0 if none).
        """
        n_frames = len(bbox_data["left_hand_detected"]) if "left_hand_detected" in bbox_data else len(bbox_data["right_hand_detected"])
        result = []

        for idx in range(n_frames):
            pts = []
            for side in hamer_data:
                detected_key = f"{side}_hand_detected"
                if detected_key in bbox_data and bbox_data[detected_key][idx]:
                    kpts = hamer_data[side].kpts_2d[idx]
                    if not np.allclose(kpts, 0):
                        pts.append(kpts)
            result.append(np.concatenate(pts, axis=0) if pts else np.empty((0, 2)))

        return result

    @staticmethod
    def _mask_contains_any_keypoint(mask: np.ndarray, keypoints: np.ndarray) -> bool:
        """Check if any keypoint falls inside the mask."""
        h, w = mask.shape[-2:]
        for kpt in keypoints:
            x, y = int(round(kpt[0])), int(round(kpt[1]))
            if 0 <= y < h and 0 <= x < w and mask[..., y, x]:
                return True
        return False

    def _create_visualization(self, imgs_rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        Create visualization by masking out segmented regions.
        
        Args:
            imgs_rgb: Original RGB video frames
            masks: Boolean segmentation masks
            
        Returns:
            Visualization images with masked regions set to black
        """
        sam_imgs = []
        for idx in range(len(imgs_rgb)):
            img = imgs_rgb[idx].copy()  # Create copy to avoid modifying original
            mask = masks[idx]
            img[mask] = 0  # Set masked regions to black
            sam_imgs.append(img)
        return np.array(sam_imgs)

    def _validate_output_consistency(
        self, 
        imgs_rgb: np.ndarray, 
        masks: np.ndarray, 
        sam_imgs: np.ndarray
    ) -> None:
        """
        Validate that output arrays have consistent dimensions.
        
        Args:
            imgs_rgb: Original RGB video frames
            masks: Segmentation masks
            sam_imgs: Visualization images
            
        Raises:
            AssertionError: If dimensions don't match
        """
        assert len(sam_imgs) == len(imgs_rgb), "Visualization length doesn't match input"
        assert len(masks) == len(imgs_rgb), "Masks length doesn't match input"


    @staticmethod
    def _save_results(
        paths: Paths,
        masks: np.ndarray,
        sam_imgs: np.ndarray,
        fps: int = DEFAULT_FPS
    ) -> None:
        """
        Save arm segmentation results to disk.

        Args:
            paths: Paths object containing output file locations
            masks: Combined arm segmentation masks
            sam_imgs: SAM visualization images
            fps: Frames per second for output videos (default: 10)
        """
        ArmSegmentationProcessor._create_output_directory(paths)
        
        try:
            ArmSegmentationProcessor._save_mask_data(paths, masks)
            ArmSegmentationProcessor._create_videos(paths, masks, sam_imgs, fps)
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise

        ArmSegmentationProcessor._cleanup_temp_files(paths)
        ArmSegmentationProcessor._update_annotation_video(paths, masks, sam_imgs, fps)

    @staticmethod
    def _create_output_directory(paths: Paths) -> None:
        """
        Create output directory for segmentation results.
        
        Args:
            paths: Paths object containing output directory location
        """
        if not os.path.exists(paths.segmentation_processor):
            os.makedirs(paths.segmentation_processor)

    @staticmethod
    def _save_mask_data(paths: Paths, masks: np.ndarray) -> None:
        """
        Save mask data to disk.
        
        Args:
            paths: Paths object containing output file locations
            masks: Segmentation masks to save
        """
        np.save(paths.masks_arm, masks)

    @staticmethod
    def _create_videos(paths: Paths, masks: np.ndarray, sam_imgs: np.ndarray, fps: int) -> None:
        """
        Create visualization videos from masks and SAM images.
        
        Args:
            paths: Paths object containing output file locations
            masks: Segmentation masks
            sam_imgs: SAM visualization images
            fps: Frames per second for output videos
        """
        for name, data in [
            ("video_masks_arm", masks),
            ("video_sam_arm", sam_imgs),
        ]:
            output_path = getattr(paths, name)
            media.write_video(output_path, data, fps=fps, codec=DEFAULT_CODEC)

    @staticmethod
    def _cleanup_temp_files(paths: Paths) -> None:
        """
        Clean up temporary directories created during processing.
        
        Args:
            paths: Paths object containing temporary directory locations
        """
        if os.path.exists(paths.original_images_folder):
            shutil.rmtree(paths.original_images_folder)
        if os.path.exists(paths.original_images_folder_reverse):
            shutil.rmtree(paths.original_images_folder_reverse)

    @staticmethod
    def _update_annotation_video(paths: Paths, masks: np.ndarray, sam_imgs: np.ndarray, fps: int) -> None:
        """
        Update existing annotation video with segmentation results.
        
        Args:
            paths: Paths object containing annotation video location
            masks: Segmentation masks
            sam_imgs: SAM visualization images
            fps: Frames per second for output video
        """
        if os.path.exists(paths.video_annot):
            annot_imgs = media.read_video(paths.video_annot)
            for idx in range(len(annot_imgs)):
                annot_img = annot_imgs[idx]
                h = masks[idx].shape[0]
                w = masks[idx].shape[1]
                # Insert segmentation visualization in the top-right quadrant
                annot_img[:h, w:, :] = sam_imgs[idx]
            media.write_video(paths.video_annot, annot_imgs, fps=fps, codec=ANNOTATION_CODEC)



class HandSegmentationProcessor(BaseSegmentationProcessor): 
    """
    Processor for precise hand-only segmentation in video sequences.
    
    Attributes:
        Inherits detector_sam from BaseSegmentationProcessor
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the hand segmentation processor.
        
        Args:
            args: Command line arguments containing segmentation configuration
        """
        super().__init__(args)

    def process_one_demo(self, data_sub_folder: str, hamer_data: Optional[Dict[str, HandSequence]] = None) -> None:
        """
        Process a single video demonstration to generate precise hand segmentation masks.

        Args:
            data_sub_folder: Path to the subfolder containing the demo data
            hamer_data: Optional pre-loaded hand pose data for segmentation guidance

        Raises:
            FileNotFoundError: If required input files are not found
            ValueError: If video frames or bounding boxes are invalid
        """
        save_folder = self.get_save_folder(data_sub_folder)

        paths = self.get_paths(save_folder)
        paths._setup_original_images()
        paths._setup_original_images_reverse()

        # Load and validate input data
        imgs_rgb = self._load_video(paths.video_left)
        bbox_data = self._load_bbox_data(paths.bbox_data)
        if hamer_data is None:
            hamer_data = self._load_hamer_data(paths)

        # Process left and right hands separately for precise segmentation
        left_data = self._process_hand_data(
            imgs_rgb,
            bbox_data["left_bboxes"],
            bbox_data["left_bbox_min_dist_to_edge"],
            bbox_data["left_hand_detected"],
            hamer_data["left"],
            paths,
            "left"
        )

        right_data = self._process_hand_data(
            imgs_rgb,
            bbox_data["right_bboxes"],
            bbox_data["right_bbox_min_dist_to_edge"],
            bbox_data["right_hand_detected"],
            hamer_data["right"],
            paths,
            "right"
        )

        # Convert to boolean masks
        left_masks = left_data["left_masks"].astype(np.bool_)
        left_sam_imgs = left_data["left_sam_imgs"]
        right_masks = right_data["right_masks"].astype(np.bool_)
        right_sam_imgs = right_data["right_sam_imgs"]

        # Save results with separate left/right hand data
        self._save_results(paths, left_masks, left_sam_imgs, right_masks, right_sam_imgs)


    def _process_hand_data(
        self,
        imgs_rgb: np.ndarray,
        bboxes: np.ndarray,
        bbox_min_dist: np.ndarray,
        hand_detected: np.ndarray,
        hamer_data: HandSequence,
        paths: Paths,
        hand_side: str
    ) -> Dict[str, np.ndarray]:
        """
        Process hand segmentation data for a single hand (left or right).

        Args:
            imgs_rgb: RGB video frames
            bboxes: Hand bounding boxes from detection stage
            bbox_min_dist: Minimum distances to image edges (quality metric)
            hand_detected: Boolean flags indicating valid hand detections
            hamer_data: Hand pose data for segmentation guidance
            paths: Paths object for file management
            hand_side: "left" or "right" specifying which hand to process

        Returns:
            Dictionary containing segmentation masks and visualization images
        """
        # Handle cases with no valid detections
        if not hand_detected.any() or max(bbox_min_dist) == 0:
            return {
                f"{hand_side}_masks": np.zeros((len(imgs_rgb), imgs_rgb[0].shape[0], imgs_rgb[0].shape[1])),
                f"{hand_side}_sam_imgs": np.zeros((len(imgs_rgb), imgs_rgb[0].shape[0], imgs_rgb[0].shape[1], 3))
            }
        
        # Extract hand pose keypoints for segmentation guidance
        kpts_2d = hamer_data.kpts_2d
                
        # Find the frame with highest quality (furthest from edges)
        max_dist_idx = np.argmax(bbox_min_dist)
        bbox = bboxes[max_dist_idx]
        points = np.expand_dims(kpts_2d[max_dist_idx], axis=1)

        # Process segmentation in both temporal directions
        masks_forward, sam_imgs_forward = self._run_sam_segmentation(
            paths, bbox, points, max_dist_idx, reverse=False, output_bboxes=bboxes
        )
        masks_reverse, sam_imgs_reverse = self._run_sam_segmentation(
            paths, bbox, points, max_dist_idx, reverse=True, output_bboxes=bboxes
        )

        # Combine bidirectional results
        sam_imgs = self._combine_sam_images(imgs_rgb, sam_imgs_forward, sam_imgs_reverse)
        masks = self._combine_masks(imgs_rgb, masks_forward, masks_reverse)

        return {
            f"{hand_side}_masks": masks,
            f"{hand_side}_sam_imgs": sam_imgs
        }
    

    def _run_sam_segmentation(
        self,
        paths: Paths,
        bbox: np.ndarray,
        points: np.ndarray,
        max_dist_idx: int,
        reverse: bool,
        output_bboxes: np.ndarray
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Process video segmentation in either forward or reverse temporal direction.
        
        Args:
            paths: Paths object for file management
            bbox: Initial bounding box for segmentation
            points: Hand keypoints for segmentation guidance
            max_dist_idx: Index of highest-quality frame for initialization
            reverse: Whether to process in reverse temporal order
            output_bboxes: All bounding boxes for the sequence
            
        Returns:
            Tuple of (segmentation_masks, visualization_images)
        """
        return self.detector_sam.segment_video(
            paths.original_images_folder,
            bbox,
            points,
            [max_dist_idx],
            reverse=reverse,
            output_bboxes=output_bboxes
        )

    @staticmethod
    def _save_results(
        paths: Paths,
        left_masks: np.ndarray,
        left_sam_imgs: np.ndarray,
        right_masks: np.ndarray,
        right_sam_imgs: np.ndarray,
        fps: int = DEFAULT_FPS
    ) -> None:
        """
        Save hand segmentation results to disk.

        Args:
            paths: Paths object containing output file locations
            left_masks: Left hand segmentation masks
            left_sam_imgs: Left hand SAM visualization images
            right_masks: Right hand segmentation masks
            right_sam_imgs: Right hand SAM visualization images
            fps: Frames per second for output videos (default: 10)
        """
        HandSegmentationProcessor._create_output_directory(paths)
        
        try:
            HandSegmentationProcessor._save_hand_mask_data(paths, left_masks, right_masks)
            HandSegmentationProcessor._create_hand_videos(paths, left_masks, left_sam_imgs, right_masks, right_sam_imgs, fps)
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise
        
        HandSegmentationProcessor._cleanup_temp_files(paths)

    @staticmethod
    def _create_output_directory(paths: Paths) -> None:
        """
        Create output directory for segmentation results.
        
        Args:
            paths: Paths object containing output directory location
        """
        if not os.path.exists(paths.segmentation_processor):
            os.makedirs(paths.segmentation_processor)

    @staticmethod
    def _save_hand_mask_data(paths: Paths, left_masks: np.ndarray, right_masks: np.ndarray) -> None:
        """
        Save hand mask data to disk.
        
        Args:
            paths: Paths object containing output file locations
            left_masks: Left hand segmentation masks
            right_masks: Right hand segmentation masks
        """
        np.save(paths.masks_hand_left, left_masks)
        np.save(paths.masks_hand_right, right_masks)

    @staticmethod
    def _create_hand_videos(
        paths: Paths, 
        left_masks: np.ndarray, 
        left_sam_imgs: np.ndarray,
        right_masks: np.ndarray, 
        right_sam_imgs: np.ndarray, 
        fps: int
    ) -> None:
        """
        Create visualization videos for hand segmentation.
        
        Args:
            paths: Paths object containing output file locations
            left_masks: Left hand segmentation masks
            left_sam_imgs: Left hand SAM visualization images
            right_masks: Right hand segmentation masks
            right_sam_imgs: Right hand SAM visualization images
            fps: Frames per second for output videos
        """
        for name, data in [
            ("video_masks_hand_left", left_masks),
            ("video_masks_hand_right", right_masks),
            ("video_sam_hand_left", left_sam_imgs),
            ("video_sam_hand_right", right_sam_imgs),
        ]:
            output_path = getattr(paths, name)
            media.write_video(output_path, data, fps=fps, codec=DEFAULT_CODEC)

    @staticmethod
    def _cleanup_temp_files(paths: Paths) -> None:
        """
        Clean up temporary directories created during processing.
        
        Args:
            paths: Paths object containing temporary directory locations
        """
        if os.path.exists(paths.original_images_folder):
            shutil.rmtree(paths.original_images_folder)
        if os.path.exists(paths.original_images_folder_reverse):
            shutil.rmtree(paths.original_images_folder_reverse)

