"""
Path management for Phantom.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import yaml

from phantom.utils.image_utils import convert_video_to_images

@dataclass
class Paths:
    """Data class containing all file paths used by processors."""
    data_path: Path
    robot_name: str = "franka"

    def __post_init__(self):
        """Compute derived paths based on base paths."""
        # Convert string paths to Path objects if needed
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        
        # Validate data path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
            
        # Videos
        self.video_left = self.data_path / "video_L.mp4"
        self.video_right = self.data_path / "video_R.mp4"
        self.video_rgb_imgs = self.data_path / "video_rgb_imgs.mkv"

        # Image folders
        self.original_images_folder = self.data_path / "original_images"
        # self._setup_original_images()
        self.original_images_folder_reverse = self.data_path / "original_images_reverse"
        # self._setup_original_images_reverse()

        # Epic annotations
        self.hand_detection_data = self.data_path / "hand_det.pkl"
        self.cam_extrinsics_data = self.data_path / "extrinsics.npy"

        # Depth
        self.depth = self.data_path / "depth.npy"

        # Bbox processor
        self.bbox_processor = self.data_path / "bbox_processor"
        self.bbox_data = self.bbox_processor / "bbox_data.npz"
        self.video_bboxes = self.bbox_processor / "video_bboxes.mkv"

        # Segmentation processor
        self.segmentation_processor = self.data_path / "segmentation_processor"
        self.masks_arm = self.segmentation_processor / "masks_arm.npy"
        self.video_masks_arm = self.segmentation_processor / "video_masks_arm.mkv"
        self.video_sam_arm = self.segmentation_processor / "video_sam_arm.mkv"
        for side in ["left", "right"]:
            setattr(self, f"masks_hand_{side}", self.segmentation_processor / f"masks_hand_{side}.npy")
            setattr(self, f"video_masks_hand_{side}", self.segmentation_processor / f"video_masks_hand_{side}.mkv")
            setattr(self, f"video_sam_hand_{side}", self.segmentation_processor / f"video_sam_hand_{side}.mkv")

        # Hand Processor
        self.hand_processor = self.data_path / f"hand_processor"
        for side in ["left", "right"]:
            setattr(self, f"hand_data_{side}", self.hand_processor / f"hand_data_{side}.npz")
            setattr(self, f"hand_data_3d_{side}", self.hand_processor / f"hand_data_3d_{side}.npz")
        self.video_annot = self.data_path / "video_annot.mp4"

        # Intent processor (contact-grounded retargeting, Stage A)
        self.intent_processor = self.data_path / "intent_processor"
        self.T_camera_c2w = self.data_path / "T_camera_c2w.npy"
        self.object_masks = self.intent_processor / "object_masks.npy"
        self.object_pcd = self.intent_processor / "object_pcd.npz"
        self.video_object_mask = self.intent_processor / "video_object_mask.mp4"
        self.object_pcd_preview = self.intent_processor / "object_pcd_preview.png"
        self.contact_events = self.intent_processor / "contact_events.npz"
        self.contact_diagnostic = self.intent_processor / "contact_diagnostic.png"
        self.grasp = self.intent_processor / "grasp.npz"
        self.grasp_preview = self.intent_processor / "grasp_preview.png"
        self.intent = self.intent_processor / "intent.npz"
        self.intent_preview = self.intent_processor / "intent_preview.png"

        # Stage B (whole-trajectory optimization) output.
        self.stageb_processor = self.data_path / "stageb_processor"
        self.joint_trajectory = self.stageb_processor / "q_trajectory.npz"
        self.stageb_diagnostic = self.stageb_processor / "stageb_diagnostic.png"

        # Stage C (retarget render + label + trajectory-level pruning) output.
        self.retarget_processor = self.data_path / "retarget_processor"
        self.retarget_video_overlay = self.retarget_processor / "video_overlay.mkv"
        self.retarget_training_data = self.retarget_processor / "training_data.npz"
        self.retarget_quality = self.retarget_processor / "quality_report.npz"
        self.retarget_diagnostic = self.retarget_processor / "retarget_diagnostic.png"

        # Action processor
        self.action_processor = self.data_path / "action_processor"
        for side in ["left", "right"]:
            setattr(self, f"actions_{side}", self.action_processor / f"actions_{side}.npz")
        
        # Smoothing processor
        self.smoothing_processor = self.data_path / f"smoothing_processor"
        for side in ["left", "right"]:
            setattr(self, f"smoothed_actions_{side}", self.smoothing_processor / f"smoothed_actions_{side}.npz")

        # Inpaint processor
        self.inpaint_processor = self.data_path / "inpaint_processor"
        self.video_overlay = self.data_path / "video_overlay.mkv"
        self.video_human_inpaint = self.inpaint_processor / "video_human_inpaint.mkv"
        self.video_inpaint_overlay = self.inpaint_processor / "video_inpaint_overlay.mkv"
        self.video_birdview = self.inpaint_processor / "video_birdview.mkv"
        self.training_data = self.inpaint_processor / "training_data.npz"

    def _setup_original_images(self):
        """Set up original images paths."""
        convert_video_to_images(self.video_left, self.original_images_folder, square=False)
        image_paths = sorted(
            list(self.original_images_folder.glob("*.jpg")), 
            key=lambda x: int(x.stem)
        )
        self.original_images = image_paths

    def _setup_original_images_reverse(self):
        """Set up original images paths."""
        convert_video_to_images(self.video_left, self.original_images_folder_reverse, square=False, reverse=True)
        image_paths = sorted(
            list(self.original_images_folder_reverse.glob("*.jpg")), 
            key=lambda x: int(x.stem)
        )
        self.original_images_reverse = image_paths
    
    def ensure_directories_exist(self):
        """
        Create necessary directories if they don't exist.
        """
        # Create all necessary directories
        directories = [
            self.data_path,
        ]
        
        for directory in directories:
            if isinstance(directory, Path) and not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)



class PathsConfig:
    """
    Configuration for paths used in the project.
    
    This class handles loading and saving path configurations from files,
    and provides methods for creating Paths objects.
    """
    
    def __init__(self, config_file: Optional[str] = None) -> None:
        """
        Initialize paths configuration.
        
        Args:
            config_file: Path to configuration file. If None, use default config.
        """
        self.config: dict[str, str] = {}
        if config_file:
            self.load_config(config_file)
        else:
            self.set_default_config()
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_file: Path to configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in configuration file {config_file}: {e}")
    
    def save_config(self, config_file: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config_file: Path to save configuration file
            
        Raises:
            OSError: If unable to write to the file
        """
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def set_default_config(self) -> None:
        """Set default configuration values."""
        self.config = {
            'data_root': './data',
            'processed_root': './processed_data',
            'project_name': 'phantom',
        }
    
    def get_paths(self, demo_name: str, robot_name: str = "franka") -> Paths:
        """
        Get Paths object for a specific demo.
        
        Args:
            demo_name: Name of the demo
            robot_name: Name of the robot
            
        Returns:
            Paths object for the demo
        """
        data_path = os.path.join(self.config['data_root'], demo_name)
        
        return Paths(
            data_path=Path(data_path),
            robot_name=robot_name
        )
    
    def get_all_demo_paths(self) -> List[str]:
        """
        Get list of all demo paths in data root.
        
        Returns:
            List of demo paths
        """
        data_root = self.config['data_root']
        all_data_collection_folders = [
            f for f in os.listdir(data_root) 
            if os.path.isdir(os.path.join(data_root, f))
        ]
        
        all_data_folders = [
            os.path.join(d1, d2) 
            for d1 in os.listdir(data_root) 
            if os.path.isdir(os.path.join(data_root, d1)) 
            for d2 in os.listdir(os.path.join(data_root, d1)) 
            if os.path.isdir(os.path.join(data_root, d1, d2))
        ]
        
        return sorted(all_data_folders, key=lambda x: tuple(map(int, x.rsplit('/', 2)[-2:])))