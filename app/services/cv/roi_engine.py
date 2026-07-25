from shapely.geometry import Polygon, box

from app.services.cv.types import Detection, Obstruction


def validate_polygon(points: list[list[float]]) -> Polygon:
    if len(points) < 3:
        raise ValueError("The monitored zone needs at least three points")
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("The monitored zone polygon is invalid")
    return polygon


def calculate_overlap(detection_box: tuple[int, int, int, int], polygon: Polygon) -> float:
    object_shape = box(*detection_box)
    if object_shape.area <= 0:
        return 0.0
    return float(object_shape.intersection(polygon).area / object_shape.area)


def evaluate_detection(
    detection: Detection,
    polygon: Polygon,
    blocked_classes: set[str],
    minimum_overlap: float,
) -> Obstruction:
    overlap = calculate_overlap(detection.box, polygon)
    normalized = detection.label.strip().lower()
    is_blocked_class = normalized in blocked_classes or any(
        item in normalized for item in blocked_classes
    )
    return Obstruction(
        detection=detection,
        overlap=overlap,
        is_blocking=is_blocked_class and overlap >= minimum_overlap,
    )
