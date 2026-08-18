"""
Wrapper around HuggingFace OWLv2 for open-vocabulary object detection.

Drop-in replacement for DetectorYoloWorld: same get_bboxes / get_best_bbox API,
text prompt in, xyxy boxes + scores out. No custom CUDA ops (unlike
Grounding-DINO). Used as an optional IntentProcessor seed detector
(``intent_seed_detector=owlv2``).
"""
from typing import Optional, Tuple

import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"


class DetectorOwlv2:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        import torch
        from transformers import AutoProcessor, Owlv2ForObjectDetection

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        kwargs = {"attn_implementation": "eager"}
        try:
            self.model = Owlv2ForObjectDetection.from_pretrained(model_id, **kwargs)
        except TypeError:
            self.model = Owlv2ForObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def _to_device(self, inputs, device: str):
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

    def _post_process(self, outputs, threshold: float, target_sizes):
        post = getattr(self.processor, "post_process_object_detection", None)
        if post is None:
            post = self.processor.image_processor.post_process_object_detection
        return post(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

    def _predict(self, frame: np.ndarray, object_name: str, threshold: float):
        import torch
        from PIL import Image

        prompt = object_name.strip().rstrip(".")
        img = Image.fromarray(frame) if frame.ndim == 3 else Image.fromarray(frame)
        inputs = self.processor(text=[[prompt]], images=img, return_tensors="pt")
        target_sizes = torch.tensor([(img.height, img.width)])

        def _run(device: str):
            batched = self._to_device(inputs, device)
            sizes = target_sizes.to(device)
            with torch.inference_mode():
                outputs = self.model(**batched)
            return self._post_process(outputs, threshold, sizes)

        try:
            return _run(self.device)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if self.device != "cpu" and ("cuda" in msg or "kernel image is invalid" in msg):
                logger.warning("OWLv2 CUDA failed (%s); falling back to CPU", exc)
                self.device = "cpu"
                self.model.to("cpu")
                return _run("cpu")
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
        if not str(object_name).strip():
            return np.array([]), np.array([])

        results = self._predict(frame, object_name, threshold)
        if not results:
            return np.array([]), np.array([])
        result = results[0]
        boxes, scores = result["boxes"], result["scores"]
        if hasattr(boxes, "detach"):
            bboxes = boxes.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
        else:
            bboxes = np.asarray(boxes)
            scores = np.asarray(scores)
        if bboxes.size == 0:
            return np.array([]), np.array([])
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

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
