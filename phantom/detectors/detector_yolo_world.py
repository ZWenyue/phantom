"""
Wrapper around YOLO-World for open-vocabulary object detection.

Drop-in replacement for DetectorDino: same get_bboxes / get_best_bbox API,
text prompt in, xyxy boxes + scores out. Used as the SAM2 seed detector.
"""
from typing import Optional, Tuple

import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolov8x-worldv2.pt"


class DetectorYoloWorld:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        from ultralytics import YOLOWorld
        import torch

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLOWorld(model_id)
        self.model.to(self.device)
        self._classes: Optional[Tuple[str, ...]] = None

    def _set_prompt(self, object_name: str) -> None:
        prompt = object_name.strip().rstrip(".")
        classes = (prompt,)
        if classes == self._classes:
            return
        self.model.set_classes(list(classes))
        self._classes = classes

    def _predict(self, frame: np.ndarray, threshold: float):
        # Ultralytics treats numpy arrays as OpenCV BGR.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
        try:
            return self.model.predict(
                source=bgr,
                conf=threshold,
                verbose=False,
                save=False,
                device=self.device,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if self.device != "cpu" and ("cuda" in msg or "kernel image is invalid" in msg):
                logger.warning("YOLO-World CUDA failed (%s); falling back to CPU", exc)
                self.device = "cpu"
                self.model.to("cpu")
                return self.model.predict(
                    source=bgr,
                    conf=threshold,
                    verbose=False,
                    save=False,
                    device="cpu",
                )
            raise

    def get_bboxes(
        self,
        frame: np.ndarray,
        object_name: str,
        threshold: float = 0.01,
        visualize: bool = False,
        pause_visualization: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect objects in a frame and return their bounding boxes and confidence scores.

        Args:
            frame: Input image as numpy array in RGB format
            object_name: Target object category / text prompt
            threshold: Confidence threshold for detection (0.0-1.0)
            visualize: If True, displays detection results visually
            pause_visualization: If True, waits for key press when visualizing

        Returns:
            Tuple of (bounding_boxes, confidence_scores) as numpy arrays.
            Empty arrays if no objects detected.
        """
        self._set_prompt(object_name)
        results = self._predict(frame, threshold)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.array([]), np.array([])

        boxes = results[0].boxes
        bboxes = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()

        if visualize:
            img_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
            for bbox, score in zip(bboxes, scores):
                cv2.rectangle(
                    img_bgr,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img_bgr,
                    f"{score:.4f}",
                    (int(bbox[0]), int(bbox[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("Detection", img_bgr)
            cv2.waitKey(0 if pause_visualization else 1)
        return bboxes, scores

    def get_best_bbox(
        self,
        frame: np.ndarray,
        object_name: str,
        threshold: float = 0.01,
        visualize: bool = False,
        pause_visualization: bool = True,
    ) -> Optional[np.ndarray]:
        bboxes, scores = self.get_bboxes(frame, object_name, threshold)
        if len(bboxes) == 0:
            return None
        best_idx = np.array(scores).argmax()
        best_bbox, best_score = bboxes[best_idx], scores[best_idx]

        if visualize:
            img_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(
                img_bgr,
                (int(best_bbox[0]), int(best_bbox[1])),
                (int(best_bbox[2]), int(best_bbox[3])),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img_bgr,
                f"{best_score:.4f}",
                (int(best_bbox[0]), int(best_bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Detection", img_bgr)
            cv2.waitKey(0 if pause_visualization else 1)
        return best_bbox
