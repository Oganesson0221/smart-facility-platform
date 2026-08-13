from abc import ABC, abstractmethod
import logging
from threading import Lock

import cv2
import numpy as np

from app.config import settings
from app.services.cv.types import Detection


LOGGER = logging.getLogger(__name__)


def _shared_gpu_safe_mode_active() -> bool:
    """Keep CV off a GPU already reserved by both local vLLM servers."""
    return bool(
        settings.cv_shared_gpu_safe_mode
        and settings.llm_enabled
        and settings.vision_enabled
    )


def _is_cuda_out_of_memory(error: BaseException) -> bool:
    message = str(error).lower()
    return "cuda" in message and ("out of memory" in message or "memoryallocation" in message)


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
            self.device = (
                "cpu"
                if _shared_gpu_safe_mode_active()
                else "cuda" if torch.cuda.is_available() else "cpu"
            )
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


class YoloDetector(Detector):
    name = "yolo"

    def __init__(self):
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO support requires Ultralytics. Run: pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        self.device = self._resolve_device()
        try:
            self.model = YOLO(settings.yolo_model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load YOLO model '{settings.yolo_model_path}'. "
                "If this is the first run, either allow Ultralytics to download the "
                "weights or point YOLO_MODEL_PATH to a local .pt file."
            ) from exc
        self.class_names = self.model.names

    def _resolve_device(self) -> str:
        requested = settings.yolo_device if settings.yolo_device != "auto" else settings.device
        if requested == "auto":
            if _shared_gpu_safe_mode_active():
                LOGGER.info(
                    "YOLO auto device resolved to CPU because both local vLLM servers are enabled"
                )
                return "cpu"
            return "cuda:0" if self.torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not self.torch.cuda.is_available():
            return "cpu"
        return requested

    def detect(self, image: np.ndarray) -> list[Detection]:
        try:
            results = self._predict(image)
        except RuntimeError as exc:
            if not self.device.startswith("cuda") or not _is_cuda_out_of_memory(exc):
                raise
            LOGGER.warning(
                "YOLO exhausted CUDA memory; moving the detector to CPU and retrying once"
            )
            self.device = "cpu"
            try:
                self.model.to("cpu")
            finally:
                self.torch.cuda.empty_cache()
            results = self._predict(image)
        if not results:
            return []
        return suppress_duplicate_detections(
            convert_yolo_results(results[0].boxes, self.class_names)
        )

    def _predict(self, image: np.ndarray):
        return self.model.predict(
            source=image,
            conf=settings.yolo_confidence_threshold,
            imgsz=settings.yolo_image_size,
            device=self.device,
            verbose=False,
            agnostic_nms=True,
        )


def convert_yolo_results(boxes, class_names) -> list[Detection]:
    detections: list[Detection] = []
    for box in boxes:
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
        detections.append(
            Detection(
                str(class_names.get(cls_id, cls_id)).lower(),
                float(box.conf.item()),
                (x1, y1, x2, y2),
            )
        )
    return detections


def _box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - inter_area
    return inter_area / union if union > 0 else 0.0


def suppress_duplicate_detections(
    detections: list[Detection], overlap_threshold: float = 0.9
) -> list[Detection]:
    filtered: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(_box_iou(candidate.box, kept.box) >= overlap_threshold for kept in filtered):
            continue
        filtered.append(candidate)
    return filtered


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
                elif provider == "yolo":
                    _detector = YoloDetector()
                elif provider == "demo":
                    _detector = DemoDetector()
                else:
                    raise RuntimeError(f"Unsupported detector provider: {provider}")
    return _detector
