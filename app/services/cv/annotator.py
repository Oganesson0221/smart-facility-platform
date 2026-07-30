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
    height, width = output.shape[:2]
    polygon = np.array(polygon_points, dtype=np.int32)
    overlay = output.copy()
    cv2.fillPoly(overlay, [polygon], (30, 190, 100))
    output = cv2.addWeighted(overlay, 0.18, output, 0.82, 0)
    cv2.polylines(output, [polygon], True, (57, 230, 150), 3)
    banner_color = (40, 60, 240) if any(item.is_blocking for item in obstructions) else (35, 150, 110)
    cv2.rectangle(output, (0, 0), (width, max(52, height // 12)), (12, 16, 22), -1)
    cv2.rectangle(output, (0, 0), (max(12, width // 80), max(52, height // 12)), banner_color, -1)
    cv2.putText(
        output,
        "FIRE EXIT OBSTRUCTION",
        (24, max(34, height // 18)),
        cv2.FONT_HERSHEY_DUPLEX,
        max(0.6, min(1.0, width / 1200)),
        banner_color,
        2,
        cv2.LINE_AA,
    )
    for item in obstructions:
        x1, y1, x2, y2 = item.detection.box
        color = (40, 60, 240) if item.is_blocking else (30, 200, 230)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        label = (
            f"{item.detection.label} {item.detection.confidence:.0%} "
            f"track {item.detection.track_id or '-'} "
            f"in {item.object_intrusion_ratio:.0%} "
            f"zone {item.exit_blockage_ratio:.0%} "
            f"t {item.blocked_duration_seconds:.1f}s"
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
    annotations = assessment.get("grounded_annotations") or assessment.get("annotations", [])
    stroke = max(4, width // 260)
    outer_stroke = stroke + 2
    font_scale = max(0.72, min(1.02, width / 1180))
    label_height = max(34, int(26 * font_scale))
    for item in annotations:
        box = item.get("box", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (
            int(box[0] * width / 1000),
            int(box[1] * height / 1000),
            int(box[2] * width / 1000),
            int(box[3] * height / 1000),
        )
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width - 1, x2))
        y2 = max(y1 + 1, min(height - 1, y2))

        # Add a subtle tint inside the box so the annotation survives UI scaling.
        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        output = cv2.addWeighted(overlay, 0.1, output, 0.9, 0)

        # White under-stroke plus colored stroke improves visibility on mixed backgrounds.
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), outer_stroke)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, stroke)
        label = str(item.get("label") or "violation evidence")[:50]
        label_y = max(label_height, y1)
        text_width = max(148, int(len(label) * 12 * font_scale))
        label_x2 = min(width, x1 + text_width)
        label_y1 = max(0, label_y - label_height)

        cv2.rectangle(
            output,
            (x1, label_y1),
            (label_x2, label_y),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            output,
            (x1 + 2, min(height - 1, label_y1 + 2)),
            (max(x1 + 2, label_x2 - 2), max(label_y1 + 2, label_y - 2)),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 8, max(22, label_y - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (18, 20, 24),
            2,
            cv2.LINE_AA,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), output):
        raise OSError(f"Could not write annotated image to {destination}")
