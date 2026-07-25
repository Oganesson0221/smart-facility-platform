from pathlib import Path

import cv2
import numpy as np

from app.services.cv.types import Obstruction


def annotate(
    image: np.ndarray,
    polygon_points: list[list[float]],
    obstructions: list[Obstruction],
    destination: Path,
) -> None:
    output = image.copy()
    polygon = np.array(polygon_points, dtype=np.int32)
    overlay = output.copy()
    cv2.fillPoly(overlay, [polygon], (30, 190, 100))
    output = cv2.addWeighted(overlay, 0.18, output, 0.82, 0)
    cv2.polylines(output, [polygon], True, (57, 230, 150), 3)
    for item in obstructions:
        x1, y1, x2, y2 = item.detection.box
        color = (40, 60, 240) if item.is_blocking else (30, 200, 230)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        label = (
            f"{item.detection.label} {item.detection.confidence:.0%} "
            f"overlap {item.overlap:.0%}"
        )
        cv2.rectangle(output, (x1, max(0, y1 - 27)), (x1 + len(label) * 9, y1), color, -1)
        cv2.putText(
            output, label, (x1 + 4, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), output)


def annotate_scene(
    image: np.ndarray,
    assessment: dict,
    destination: Path,
) -> None:
    output = image.copy()
    height, width = output.shape[:2]
    violation = bool(assessment.get("violation"))
    color = (45, 55, 235) if violation else (45, 185, 95)
    title = (
        f"VIOLATION: {assessment.get('category', 'Safety issue')}"
        if violation
        else "NO VISIBLE VIOLATION"
    )
    summary = str(assessment.get("summary", ""))

    overlay = output.copy()
    banner_height = max(92, int(height * 0.16))
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (12, 16, 22), -1)
    output = cv2.addWeighted(overlay, 0.86, output, 0.14, 0)
    cv2.rectangle(output, (0, 0), (max(10, int(width * 0.014)), banner_height), color, -1)
    scale = max(0.55, min(1.1, width / 1100))
    cv2.putText(
        output, title[:70], (30, int(banner_height * 0.43)),
        cv2.FONT_HERSHEY_DUPLEX, scale, color, 2, cv2.LINE_AA
    )
    max_chars = max(35, int(width / (11 * scale)))
    cv2.putText(
        output, summary[:max_chars], (30, int(banner_height * 0.78)),
        cv2.FONT_HERSHEY_SIMPLEX, scale * 0.72, (245, 245, 245), 1, cv2.LINE_AA
    )
    for item in assessment.get("annotations", []):
        box = item.get("box", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (
            int(box[0] * width / 1000),
            int(box[1] * height / 1000),
            int(box[2] * width / 1000),
            int(box[3] * height / 1000),
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, max(3, width // 450))
        label = str(item.get("label") or "violation evidence")[:50]
        label_y = max(banner_height + 24, y1)
        cv2.rectangle(
            output,
            (x1, max(banner_height, label_y - 27)),
            (min(width, x1 + max(130, len(label) * 10)), label_y),
            color,
            -1,
        )
        cv2.putText(
            output, label, (x1 + 6, label_y - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), output):
        raise OSError(f"Could not write annotated image to {destination}")
