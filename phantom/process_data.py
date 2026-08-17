import logging
from enum import Enum
from tqdm import tqdm
from joblib import Parallel, delayed  # type: ignore
import hydra
from omegaconf import DictConfig

from phantom.processors.base_processor import BaseProcessor

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")

class ProcessingMode(Enum):
    """Enumeration of valid processing modes."""
    BBOX = "bbox"
    HAND2D = "hand2d"
    HAND3D = "hand3d"
    HAND_SEGMENTATION = "hand_segmentation"
    ARM_SEGMENTATION = "arm_segmentation"
    INTENT = "intent"
    STAGEB = "stageb"
    RETARGET_INPAINT = "retarget_inpaint"
    ACTION = "action"
    SMOOTHING = "smoothing"
    HAND_INPAINT = "hand_inpaint"
    ROBOT_INPAINT = "robot_inpaint"
    ALL = "all"

PROCESSING_ORDER = [
    "bbox",
    "hand2d",
    "arm_segmentation",
    "hand_segmentation",
    "hand3d",
    "intent",
    "stageb",
    "action",
    "smoothing",
    "hand_inpaint",
    "retarget_inpaint",
    "robot_inpaint",
]

PROCESSING_ORDER_EPIC = [
    "bbox",
    "hand2d",
    "arm_segmentation",
    "intent",
    "stageb",
    "action",
    "smoothing",
    "hand_inpaint",
    "retarget_inpaint",
    "robot_inpaint",
]

def process_one_demo(data_sub_folder: str, cfg: DictConfig, processor_classes: dict) -> None:
    # Choose processing order based on epic flag
    processing_order = PROCESSING_ORDER_EPIC if cfg.epic else PROCESSING_ORDER
    
    # Handle both string and list modes
    if isinstance(cfg.mode, str):
        # Handle comma-separated string format
        if ',' in cfg.mode:
            selected_modes = []
            for mode in cfg.mode.split(','):
                mode = mode.strip()
                if mode == "all":
                    selected_modes.extend(processing_order)
                elif mode in processing_order:
                    selected_modes.append(mode)
        else:
            selected_modes = [m for m in processing_order if m in cfg.mode or "all" in cfg.mode]
    else:
        # For list of modes, use the order provided by user
        selected_modes = []
        for mode in cfg.mode:
            if mode == "all":
                selected_modes.extend(processing_order)
            elif mode in processing_order:
                selected_modes.append(mode)
    
    for mode in selected_modes:
        print(f"----------------- {mode.upper()} PROCESSOR -----------------")
        processor_cls = processor_classes[mode]
        processor = processor_cls(cfg)
        try:
            processor.process_one_demo(data_sub_folder)
        except Exception as e:
            print(f"Error in {mode} processing: {e}")
            if cfg.debug:
                raise

def process_all_demos(cfg: DictConfig, processor_classes: dict) -> None:
    # Choose processing order based on epic flag
    processing_order = PROCESSING_ORDER_EPIC if cfg.epic else PROCESSING_ORDER
    
    # Handle both string and list modes
    if isinstance(cfg.mode, str):
        # Handle comma-separated string format
        if ',' in cfg.mode:
            selected_modes = []
            for mode in cfg.mode.split(','):
                mode = mode.strip()
                if mode == "all":
                    selected_modes.extend(processing_order)
                elif mode in processing_order:
                    selected_modes.append(mode)
        else:
            selected_modes = [m for m in processing_order if m in cfg.mode or "all" in cfg.mode]
    else:
        # For list of modes, use the order provided by user
        selected_modes = []
        for mode in cfg.mode:
            if mode == "all":
                selected_modes.extend(processing_order)
            elif mode in processing_order:
                selected_modes.append(mode)
    
    base_processor = BaseProcessor(cfg)
    all_data_folders = base_processor.all_data_folders.copy()
    for mode in selected_modes:
        print(f"----------------- {mode.upper()} PROCESSOR -----------------")
        processor_cls = processor_classes[mode]
        processor = processor_cls(cfg)
        for data_sub_folder in tqdm(all_data_folders):
            try:
                processor.process_one_demo(data_sub_folder)
            except Exception as e:
                print(f"Error in {mode} processing: {e}")
                if cfg.debug:
                    raise

def process_all_demos_parallel(cfg: DictConfig, processor_classes: dict) -> None:
    # Choose processing order based on epic flag
    processing_order = PROCESSING_ORDER_EPIC if cfg.epic else PROCESSING_ORDER
    
    # Handle both string and list modes
    if isinstance(cfg.mode, str):
        # Handle comma-separated string format
        if ',' in cfg.mode:
            selected_modes = []
            for mode in cfg.mode.split(','):
                mode = mode.strip()
                if mode == "all":
                    selected_modes.extend(processing_order)
                elif mode in processing_order:
                    selected_modes.append(mode)
        else:
            selected_modes = [m for m in processing_order if m in cfg.mode or "all" in cfg.mode]
    else:
        # For list of modes, use the order provided by user
        selected_modes = []
        for mode in cfg.mode:
            if mode == "all":
                selected_modes.extend(processing_order)
            elif mode in processing_order:
                selected_modes.append(mode)
    
    base_processor = BaseProcessor(cfg)
    all_data_folders = base_processor.all_data_folders.copy()
    for mode in selected_modes:
        print(f"----------------- {mode.upper()} PROCESSOR -----------------")
        processor_cls = processor_classes[mode]
        processor = processor_cls(cfg) 
        Parallel(n_jobs=cfg.n_processes)(
            delayed(processor.process_one_demo)(data_sub_folder) for data_sub_folder in all_data_folders
        )

# mode -> (module path, class name). Imported lazily so that a broken/heavy
# dependency for one processor (e.g. detectron2/mmpose for hand stages) does not
# block running an unrelated mode (e.g. `intent`).
PROCESSOR_REGISTRY = {
    "bbox": ("phantom.processors.bbox_processor", "BBoxProcessor"),
    "hand2d": ("phantom.processors.hand_processor", "Hand2DProcessor"),
    "hand3d": ("phantom.processors.hand_processor", "Hand3DProcessor"),
    "hand_segmentation": ("phantom.processors.segmentation_processor", "HandSegmentationProcessor"),
    "arm_segmentation": ("phantom.processors.segmentation_processor", "ArmSegmentationProcessor"),
    "action": ("phantom.processors.action_processor", "ActionProcessor"),
    "intent": ("phantom.processors.intent_processor", "IntentProcessor"),
    "stageb": ("phantom.processors.stageb_processor", "StageBProcessor"),
    "retarget_inpaint": ("phantom.processors.retarget_inpaint_processor", "RetargetInpaintProcessor"),
    "smoothing": ("phantom.processors.smoothing_processor", "SmoothingProcessor"),
    "robot_inpaint": ("phantom.processors.robotinpaint_processor", "RobotInpaintProcessor"),
    "hand_inpaint": ("phantom.processors.handinpaint_processor", "HandInpaintProcessor"),
}


class _UnavailableProcessor:
    """Placeholder for a processor whose module failed to import.

    Only raises when the mode is actually instantiated, so unrelated modes keep
    working even if this processor's dependencies are missing/broken.
    """

    def __init__(self, mode: str, error: Exception):
        self.mode = mode
        self.error = error

    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"Processor '{self.mode}' is unavailable due to an import error: {self.error!r}"
        )


def _requested_modes(cfg: DictConfig) -> list:
    """Modes this Hydra run will actually instantiate (not the full registry)."""
    if isinstance(cfg.mode, str):
        modes = [m.strip() for m in cfg.mode.split(",")] if "," in cfg.mode else [cfg.mode]
    else:
        modes = list(cfg.mode)
    if "all" in modes:
        return list(PROCESSOR_REGISTRY.keys())
    return modes


def get_processor_classes(cfg: DictConfig) -> dict:
    """Lazily import processor classes; tolerate import failures per mode.

    Only the requested modes are imported. Importing the full registry pulls in
    mmcv/mmpose CUDA ops that collide with Grounding-DINO on some GPUs.
    """
    import importlib

    requested = set(_requested_modes(cfg))
    classes = {}
    for mode, (module_path, class_name) in PROCESSOR_REGISTRY.items():
        if mode not in requested:
            continue
        try:
            module = importlib.import_module(module_path)
            classes[mode] = getattr(module, class_name)
        except Exception as e:  # noqa: BLE001
            classes[mode] = _UnavailableProcessor(mode, e)
    return classes

def validate_mode(cfg: DictConfig) -> None:
    """
    Validate that the mode parameter contains only valid processing modes.
    
    Args:
        cfg: Configuration object containing mode parameter
        
    Raises:
        ValueError: If mode contains invalid options
    """
    if isinstance(cfg.mode, str):
        # Handle comma-separated string format
        if ',' in cfg.mode:
            modes = [mode.strip() for mode in cfg.mode.split(',')]
        else:
            modes = [cfg.mode]
    else:
        modes = cfg.mode
    
    # Get valid modes from enum
    valid_modes = {mode.value for mode in ProcessingMode}
    invalid_modes = [mode for mode in modes if mode not in valid_modes]
    
    if invalid_modes:
        valid_mode_list = [mode.value for mode in ProcessingMode]
        raise ValueError(
            f"Invalid mode(s): {invalid_modes}. "
            f"Valid modes are: {valid_mode_list}"
        )

def main(cfg: DictConfig):
    # Validate mode parameter
    validate_mode(cfg)
    
    # Get processor classes
    processor_classes = get_processor_classes(cfg)
    
    if cfg.n_processes > 1:
        process_all_demos_parallel(cfg, processor_classes)
    elif cfg.demo_num is not None:
        process_one_demo(cfg.demo_num, cfg, processor_classes)
    else:
        process_all_demos(cfg, processor_classes)

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def hydra_main(cfg: DictConfig):
    """
    Main entry point using Hydra configuration.
    
    Example usage:
    - Process all demos with bbox: python process_data.py mode=bbox
    - Process single demo: python process_data.py mode=bbox demo_num=0
    - Use EPIC dataset: python process_data.py dataset=epic mode=bbox
    - Parallel processing: python process_data.py mode=bbox n_processes=4
    - Process multiple modes sequentially: python process_data.py mode=bbox,hand3d
    - Process with custom order: python process_data.py mode=hand3d,bbox,action
    - Process with bracket notation (use quotes): python process_data.py "mode=[bbox,hand3d]"
    """
    main(cfg)

if __name__ == "__main__":
    hydra_main()
