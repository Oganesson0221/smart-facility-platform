from abc import ABC, abstractmethod
from threading import Lock

import cv2
import numpy as np

from app.config import settings
from app.services.cv.types import Detection


class Detector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class DemoDetector(Detector):
    """Deterministic no-model detector for exercising the complete application."""

    name = "demo"

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 50, 130)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        minimum_area = width * height * 0.018
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < minimum_area or w > width * 0.96 or h > height * 0.96:
                continue
            aspect = w / max(h, 1)
            label = "vehicle" if aspect > 1.15 else "large object"
            confidence = min(0.95, 0.55 + area / (width * height))
            detections.append(Detection(label, confidence, (x, y, x + w, y + h)))
        return detections


class GroundingDinoDetector(Detector):
    name = "grounding_dino"

    def __init__(self):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Grounding DINO dependencies are missing. "
                "Run: pip install -r requirements-nvidia.txt"
            ) from exc

        self.torch = torch
        if settings.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = settings.device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda but PyTorch cannot access a CUDA GPU")
        self.processor = AutoProcessor.from_pretrained(settings.grounding_dino_model)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            settings.grounding_dino_model, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray) -> list[Detection]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(
            images=rgb, text=settings.detection_prompt, return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=settings.detection_threshold,
            text_threshold=settings.detection_threshold,
            target_sizes=[image.shape[:2]],
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
        return [
            Detection(
                str(label).lower(),
                float(score.item()),
                tuple(int(v) for v in box.tolist()),
            )
            for box, score, label in zip(result["boxes"], result["scores"], labels)
        ]


_detector: Detector | None = None
_lock = Lock()


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        with _lock:
            if _detector is None:
                provider = settings.detector_provider.lower()
                if provider == "grounding_dino":
                    _detector = GroundingDinoDetector()
                elif provider == "demo":
                    _detector = DemoDetector()
                else:
                    raise RuntimeError(f"Unsupported detector provider: {provider}")
    return _detector
