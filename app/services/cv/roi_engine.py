from shapely.geometry import Polygon, box

from app.services.cv.types import Detection, Obstruction


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


def evaluate_detection(
    detection: Detection,
    polygon: Polygon,
    blocked_classes: set[str],
    minimum_overlap: float,
    minimum_exit_blockage_ratio: float = 0.0,
) -> Obstruction:
    overlap = calculate_overlap(detection.box, polygon)
    exit_blockage_ratio = calculate_exit_blockage_ratio(detection.box, polygon)
    normalized = detection.label.strip().lower()
    is_blocked_class = normalized in blocked_classes or any(
        item in normalized for item in blocked_classes
    )
    return Obstruction(
        detection=detection,
        overlap=overlap,
        object_intrusion_ratio=overlap,
        exit_blockage_ratio=exit_blockage_ratio,
        is_blocking=(
            is_blocked_class
            and overlap >= minimum_overlap
            and exit_blockage_ratio >= minimum_exit_blockage_ratio
        ),
    )
