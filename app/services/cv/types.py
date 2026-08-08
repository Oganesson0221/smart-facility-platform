from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    track_id: int | None = None


@dataclass
class SegmentationResult:
    mask: np.ndarray
    polygon: list[tuple[int, int]]
    area_pixels: int
    prompt_box: tuple[int, int, int, int]
    prompt_point: tuple[int, int] | None = None
    score: float | None = None
    model_name: str = ""
    inference_ms: float = 0.0


@dataclass
class Obstruction:
    detection: Detection
    overlap: float
    object_intrusion_ratio: float
    exit_blockage_ratio: float
    is_blocking: bool
    blocked_duration_seconds: float = 0.0
    segmentation: SegmentationResult | None = None
    spatial_method: str = "yolo_box_fallback"
    mask_zone_iou: float | None = None
    fallback_reason: str | None = None
