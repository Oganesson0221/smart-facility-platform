import cv2
import numpy as np
from shapely.geometry import Polygon, box

from app.services.cv.types import Detection, Obstruction, SegmentationResult


def validate_polygon(points: list[list[float]]) -> Polygon:
    if len(points) < 3:
        raise ValueError("The fire-exit clearance zone needs at least three points")
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("The fire-exit clearance zone polygon is invalid")
    return polygon


def calculate_overlap(detection_box: tuple[int, int, int, int], polygon: Polygon) -> float:
    object_shape = box(*detection_box)
    if object_shape.area <= 0:
        return 0.0
    return float(object_shape.intersection(polygon).area / object_shape.area)


def calculate_object_intrusion_ratio(
    detection_box: tuple[int, int, int, int], polygon: Polygon
) -> float:
    object_shape = box(*detection_box)
    if object_shape.area <= 0:
        return 0.0
    return float(object_shape.intersection(polygon).area / object_shape.area)


def calculate_exit_blockage_ratio(
    detection_box: tuple[int, int, int, int], polygon: Polygon
) -> float:
    object_shape = box(*detection_box)
    if polygon.area <= 0:
        return 0.0
    return float(object_shape.intersection(polygon).area / polygon.area)


def calculate_box_zone_iou(
    detection_box: tuple[int, int, int, int], polygon: Polygon
) -> float:
    object_shape = box(*detection_box)
    if object_shape.area <= 0 or polygon.area <= 0:
        return 0.0
    intersection_area = object_shape.intersection(polygon).area
    union_area = object_shape.union(polygon).area
    return float(intersection_area / union_area) if union_area > 0 else 0.0


def box_intersects_or_near_polygon(
    detection_box: tuple[int, int, int, int],
    polygon: Polygon,
    margin_pixels: float = 0.0,
) -> bool:
    object_shape = box(*detection_box)
    if object_shape.intersects(polygon):
        return True
    return object_shape.distance(polygon) <= max(0.0, float(margin_pixels))


def calculate_mask_spatial_metrics(
    mask: np.ndarray,
    polygon_points: list[list[float]],
) -> dict[str, float]:
    if mask.ndim != 2:
        raise ValueError("SAM masks must be 2D")
    object_mask = mask.astype(bool)
    object_area = int(object_mask.sum())
    if object_area <= 0:
        return {
            "object_intrusion_ratio": 0.0,
            "exit_blockage_ratio": 0.0,
            "mask_zone_iou": 0.0,
        }

    polygon_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    polygon = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(polygon_mask, [polygon], 1)
    polygon_bool = polygon_mask.astype(bool)
    polygon_area = int(polygon_bool.sum())
    if polygon_area <= 0:
        return {
            "object_intrusion_ratio": 0.0,
            "exit_blockage_ratio": 0.0,
            "mask_zone_iou": 0.0,
        }

    intersection_area = int(np.logical_and(object_mask, polygon_bool).sum())
    union_area = int(np.logical_or(object_mask, polygon_bool).sum())
    return {
        "object_intrusion_ratio": float(intersection_area / object_area),
        "exit_blockage_ratio": float(intersection_area / polygon_area),
        "mask_zone_iou": float(intersection_area / union_area) if union_area > 0 else 0.0,
    }


def evaluate_detection(
    detection: Detection,
    polygon: Polygon,
    blocked_classes: set[str],
    minimum_overlap: float,
    minimum_exit_blockage_ratio: float = 0.0,
    segmentation: SegmentationResult | None = None,
    polygon_points: list[list[float]] | None = None,
    reject_candidate: bool = False,
    fallback_reason: str | None = None,
    allow_all_classes: bool = False,
) -> Obstruction:
    if segmentation is not None and polygon_points:
        metrics = calculate_mask_spatial_metrics(segmentation.mask, polygon_points)
        overlap = metrics["object_intrusion_ratio"]
        exit_blockage_ratio = metrics["exit_blockage_ratio"]
        mask_zone_iou = metrics["mask_zone_iou"]
        spatial_method = "sam_mask"
    else:
        overlap = calculate_overlap(detection.box, polygon)
        exit_blockage_ratio = calculate_exit_blockage_ratio(detection.box, polygon)
        mask_zone_iou = None
        spatial_method = "yolo_box_fallback"
    normalized = detection.label.strip().lower()
    is_blocked_class = allow_all_classes or normalized in blocked_classes or any(
        item in normalized for item in blocked_classes
    )
    return Obstruction(
        detection=detection,
        overlap=overlap,
        object_intrusion_ratio=overlap,
        exit_blockage_ratio=exit_blockage_ratio,
        is_blocking=(
            not reject_candidate
            and
            is_blocked_class
            and overlap >= minimum_overlap
            and exit_blockage_ratio >= minimum_exit_blockage_ratio
        ),
        segmentation=segmentation,
        spatial_method=spatial_method,
        mask_zone_iou=mask_zone_iou,
        fallback_reason=fallback_reason,
    )
