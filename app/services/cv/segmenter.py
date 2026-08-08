from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
import logging
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter

import cv2
import numpy as np

from app.config import ROOT, settings
from app.services.cv.types import SegmentationResult


LOGGER = logging.getLogger(__name__)
_VENDORED_SAM2_ROOT = ROOT / "third_party" / "sam2"


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


def clamp_box_to_image(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int] | tuple[int, int, int],
    expand_ratio: float = 0.0,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    expand_x = int(round(max(0.0, expand_ratio) * max(0, x2 - x1)))
    expand_y = int(round(max(0.0, expand_ratio) * max(0, y2 - y1)))
    x1 = max(0, min(width - 1, x1 - expand_x))
    y1 = max(0, min(height - 1, y1 - expand_y))
    x2 = max(x1 + 1, min(width, x2 + expand_x))
    y2 = max(y1 + 1, min(height, y2 + expand_y))
    return int(x1), int(y1), int(x2), int(y2)


def box_centroid_point(
    box: tuple[int, int, int, int],
) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))


def _contour_box(contour: np.ndarray) -> tuple[int, int, int, int]:
    x, y, width, height = cv2.boundingRect(contour)
    return int(x), int(y), int(x + width), int(y + height)


def _model_key() -> str:
    return settings.sam_model_size.strip().lower()


def _default_checkpoint_filename() -> str | None:
    mapping = {
        "tiny": "sam2.1_hiera_tiny.pt",
        "small": "sam2.1_hiera_small.pt",
        "base": "sam2.1_hiera_base_plus.pt",
        "base_plus": "sam2.1_hiera_base_plus.pt",
        "base-plus": "sam2.1_hiera_base_plus.pt",
        "large": "sam2.1_hiera_large.pt",
    }
    return mapping.get(_model_key())


def _resolve_checkpoint_path() -> Path | None:
    if settings.sam_checkpoint_path:
        return Path(settings.sam_checkpoint_path).expanduser()
    filename = _default_checkpoint_filename()
    if not filename:
        return None
    vendored = _VENDORED_SAM2_ROOT / "checkpoints" / filename
    return vendored if vendored.exists() else vendored


def _ensure_vendored_sam2_path() -> None:
    if not _VENDORED_SAM2_ROOT.exists():
        return
    root = str(_VENDORED_SAM2_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class Segmenter(ABC):
    name = "base"

    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> SegmentationResult:
        raise NotImplementedError


class Sam2Segmenter(Segmenter):
    name = "sam2"
    MODEL_CONFIGS = {
        "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny"),
        "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small"),
        "base": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus"),
        "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus"),
        "base-plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus"),
        "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large"),
    }

    def __init__(self):
        _ensure_vendored_sam2_path()
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM support requires the official SAM 2 package and compatible "
                "PyTorch dependencies. See README.md and requirements-sam.txt."
            ) from exc

        model_key = settings.sam_model_size.strip().lower()
        if model_key not in self.MODEL_CONFIGS:
            raise RuntimeError(
                f"Unsupported SAM_MODEL_SIZE '{settings.sam_model_size}'. "
                "Use tiny, small, base_plus, or large."
            )
        checkpoint = _resolve_checkpoint_path()
        if checkpoint is None:
            raise RuntimeError(
                "SAM_ENABLED=true but no SAM checkpoint path could be resolved."
            )
        if not checkpoint.exists():
            raise RuntimeError(
                f"SAM checkpoint not found at '{checkpoint}'. "
                "Download an official SAM 2.1 checkpoint and set SAM_CHECKPOINT_PATH."
            )

        self.torch = torch
        self.device = self._resolve_device()
        self.model_cfg, model_name = self.MODEL_CONFIGS[model_key]
        self.model_name = f"{model_name} ({self.device})"
        try:
            model = build_sam2(
                self.model_cfg,
                str(checkpoint),
                device=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not initialise the SAM 2 image predictor. "
                "Ensure the official SAM 2 package is installed with its configs."
            ) from exc
        self.predictor = SAM2ImagePredictor(model)
        LOGGER.info(
            "Initialised SAM segmenter provider=%s model=%s device=%s checkpoint=%s",
            self.name,
            model_name,
            self.device,
            checkpoint,
        )

    def _resolve_device(self) -> str:
        requested = settings.sam_device.strip().lower()
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            LOGGER.warning("SAM requested CUDA but no CUDA device is available; falling back to CPU")
            return "cpu"
        return requested

    def _autocast_context(self):
        if not self.device.startswith("cuda") or not settings.sam_use_fp16:
            return nullcontext()
        dtype = (
            self.torch.bfloat16
            if hasattr(self.torch.cuda, "is_bf16_supported") and self.torch.cuda.is_bf16_supported()
            else self.torch.float16
        )
        return self.torch.autocast("cuda", dtype=dtype)

    def _cleanup_mask(
        self,
        mask: np.ndarray,
        prompt_box: tuple[int, int, int, int],
        prompt_point: tuple[int, int] | None = None,
    ) -> np.ndarray:
        mask_uint8 = (mask.astype(np.uint8) * 255)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, 8)
        if count <= 1:
            return mask.astype(bool)

        x1, y1, x2, y2 = prompt_box
        prompt_area = max(1, (x2 - x1) * (y2 - y1))
        min_component_area = max(24, int(prompt_area * 0.01))
        best_label = 0
        best_score = -1.0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_component_area:
                continue
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            candidate_box = (left, top, left + width, top + height)
            score = (_box_iou(candidate_box, prompt_box) * 2.0) + min(area / prompt_area, 1.5)
            if prompt_point is not None:
                point_x, point_y = prompt_point
                contains_point = (
                    0 <= point_y < labels.shape[0]
                    and 0 <= point_x < labels.shape[1]
                    and labels[point_y, point_x] == label
                )
                if contains_point:
                    score += 3.0
            if score > best_score:
                best_score = score
                best_label = label
        if best_label <= 0:
            return mask.astype(bool)
        return labels == best_label

    def _mask_to_polygon(
        self,
        mask: np.ndarray,
        prompt_box: tuple[int, int, int, int],
        image_shape: tuple[int, int],
    ) -> list[tuple[int, int]]:
        contours, _ = cv2.findContours(
            (mask.astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            raise RuntimeError("SAM did not return a usable contour")
        prompt = prompt_box
        contour = max(
            contours,
            key=lambda item: (_box_iou(_contour_box(item), prompt), cv2.contourArea(item)),
        )
        epsilon = max(1.0, float(settings.sam_mask_simplification_epsilon))
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        contour_points = simplified if len(simplified) >= 3 else contour
        height, width = image_shape
        polygon = [
            (
                int(max(0, min(width - 1, point[0][0]))),
                int(max(0, min(height - 1, point[0][1]))),
            )
            for point in contour_points
        ]
        if len(polygon) < 3:
            raise RuntimeError("SAM contour simplification produced fewer than three points")
        return polygon

    def segment(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> SegmentationResult:
        prompt_box = clamp_box_to_image(
            box,
            image.shape,
            expand_ratio=max(0.0, float(settings.sam_prompt_box_expand_ratio)),
        )
        prompt_point = box_centroid_point(prompt_box)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        started = perf_counter()
        with self.torch.inference_mode(), self._autocast_context():
            self.predictor.set_image(rgb)
            predict_kwargs = {
                "box": np.asarray(prompt_box, dtype=np.float32),
                "multimask_output": False,
            }
            point_coords = np.asarray([prompt_point], dtype=np.float32)
            point_labels = np.asarray([1], dtype=np.int32)
            try:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    **predict_kwargs,
                )
            except TypeError:
                LOGGER.warning(
                    "SAM predictor rejected point prompts; retrying with the YOLO box only"
                )
                masks, scores, _ = self.predictor.predict(**predict_kwargs)
        inference_ms = (perf_counter() - started) * 1000
        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM returned no masks")
        mask = np.asarray(masks[0] if np.ndim(masks) == 3 else masks).astype(bool)
        if mask.size == 0 or not mask.any():
            raise RuntimeError("SAM returned an empty mask")

        cleaned = self._cleanup_mask(mask, prompt_box, prompt_point=prompt_point)
        if not cleaned.any():
            raise RuntimeError("SAM mask cleanup removed all regions")

        area_pixels = int(cleaned.sum())
        box_area = max(1, (prompt_box[2] - prompt_box[0]) * (prompt_box[3] - prompt_box[1]))
        area_ratio = area_pixels / box_area
        if area_ratio < 0.01:
            raise RuntimeError("SAM mask area is implausibly small for the prompt box")
        if area_ratio > 4.0:
            raise RuntimeError("SAM mask area is implausibly large for the prompt box")

        polygon = self._mask_to_polygon(cleaned, prompt_box, image.shape[:2])
        score = None
        if scores is not None and len(scores):
            try:
                score = float(np.asarray(scores).reshape(-1)[0])
            except (TypeError, ValueError):
                score = None
        return SegmentationResult(
            mask=cleaned,
            polygon=polygon,
            area_pixels=area_pixels,
            prompt_box=prompt_box,
            prompt_point=prompt_point,
            score=score,
            model_name=self.model_name,
            inference_ms=inference_ms,
        )


_segmenter: Segmenter | None = None
_lock = Lock()


def reset_segmenter_cache() -> None:
    global _segmenter
    _segmenter = None


def get_segmenter() -> Segmenter | None:
    global _segmenter
    if not settings.sam_enabled:
        return None
    if _segmenter is None:
        with _lock:
            if _segmenter is None:
                provider = settings.sam_provider.strip().lower()
                if provider == "sam2":
                    _segmenter = Sam2Segmenter()
                else:
                    raise RuntimeError(f"Unsupported SAM provider: {provider}")
    return _segmenter


def segmenter_runtime_status() -> dict[str, object]:
    _ensure_vendored_sam2_path()
    checkpoint = _resolve_checkpoint_path()
    try:
        import sam2  # noqa: F401
        importable = True
        detail = "ready" if checkpoint and checkpoint.exists() else "checkpoint missing"
    except ImportError as exc:
        importable = False
        detail = f"dependency missing: {exc}"
    return {
        "enabled": settings.sam_enabled,
        "provider": settings.sam_provider,
        "model_size": settings.sam_model_size,
        "checkpoint_configured": bool(settings.sam_checkpoint_path or _default_checkpoint_filename()),
        "checkpoint_exists": bool(checkpoint and checkpoint.exists()),
        "checkpoint_path": str(checkpoint) if checkpoint is not None else "",
        "device": settings.sam_device,
        "fail_open": settings.sam_fail_open,
        "importable": importable,
        "ready": bool(settings.sam_enabled and importable and checkpoint and checkpoint.exists()),
        "detail": detail,
    }
