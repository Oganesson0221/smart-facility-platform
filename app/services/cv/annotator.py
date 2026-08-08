from pathlib import Path

import cv2
import numpy as np

from app.services.cv.types import Obstruction


def _primary_obstruction(obstructions: list[Obstruction]) -> Obstruction | None:
    if not obstructions:
        return None
    blocking = [item for item in obstructions if item.is_blocking]
    pool = blocking or obstructions
    return max(
        pool,
        key=lambda item: (
            item.object_intrusion_ratio,
            item.exit_blockage_ratio,
            item.detection.confidence,
        ),
    )


def _line_width(width: int) -> int:
    return max(2, width // 420)


def _is_full_frame_zone(
    polygon_points: list[list[float]],
    width: int,
    height: int,
) -> bool:
    if len(polygon_points) < 4:
        return False
    xs = [int(point[0]) for point in polygon_points]
    ys = [int(point[1]) for point in polygon_points]
    return (
        min(xs) <= 1
        and min(ys) <= 1
        and max(xs) >= width - 2
        and max(ys) >= height - 2
    )


def annotate(
    image: np.ndarray,
    polygon_points: list[list[float]],
    obstructions: list[Obstruction],
    destination: Path,
) -> None:
    output = image.copy()
    height, width = output.shape[:2]
    polygon = np.array(polygon_points, dtype=np.int32)
    full_frame_zone = _is_full_frame_zone(polygon_points, width, height)
    if full_frame_zone:
        cv2.rectangle(output, (0, 0), (width - 1, height - 1), (57, 230, 150), _line_width(width) + 1)
    else:
        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon], (30, 190, 100))
        output = cv2.addWeighted(overlay, 0.18, output, 0.82, 0)
        cv2.polylines(output, [polygon], True, (57, 230, 150), _line_width(width) + 1)
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
    yolo_color = (52, 104, 240)
    sam_fill = np.array((245, 168, 55), dtype=np.uint8)
    sam_outline = (255, 214, 130)
    stroke = _line_width(width)

    for item in obstructions:
        if item.segmentation is None:
            continue
        mask = item.segmentation.mask.astype(bool)
        if mask.shape[:2] != output.shape[:2] or not mask.any():
            continue
        mask_overlay = output.copy()
        mask_overlay[mask] = (
            mask_overlay[mask] * 0.45 + sam_fill * 0.55
        ).astype(np.uint8)
        output = cv2.addWeighted(mask_overlay, 0.38, output, 0.62, 0)
        polygon_points = np.array(item.segmentation.polygon, dtype=np.int32)
        if len(polygon_points) >= 3:
            contour_overlay = output.copy()
            cv2.fillPoly(contour_overlay, [polygon_points], (245, 168, 55))
            output = cv2.addWeighted(contour_overlay, 0.10, output, 0.90, 0)
            cv2.polylines(output, [polygon_points], True, sam_outline, stroke + 1)
            label_anchor = tuple(int(value) for value in polygon_points[0])
            cv2.putText(
                output,
                f"SAM {str(item.detection.label).upper()[:18]}",
                (label_anchor[0], max(20, label_anchor[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.55, width / 1800),
                sam_outline,
                2,
                cv2.LINE_AA,
            )

    for item in obstructions:
        x1, y1, x2, y2 = item.detection.box
        color = banner_color if item.is_blocking else yolo_color
        cv2.rectangle(output, (x1, y1), (x2, y2), color, stroke + 1)
        label = (
            f"{item.detection.label} {item.detection.confidence:.0%} "
            f"track {item.detection.track_id or '-'}"
        )
        metrics = (
            f"in {item.object_intrusion_ratio:.0%} "
            f"zone {item.exit_blockage_ratio:.0%} "
            f"t {item.blocked_duration_seconds:.1f}s"
        )
        label_width = max(
            cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0],
            cv2.getTextSize(metrics, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0],
        ) + 10
        label_top = max(0, y1 - 42)
        cv2.rectangle(output, (x1, label_top), (x1 + label_width, y1), color, -1)
        cv2.putText(
            output, label, (x1 + 4, max(16, y1 - 24)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            output, metrics, (x1 + 4, max(32, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )

    primary = _primary_obstruction(obstructions)
    if primary is not None:
        panel_width = min(340, max(220, width // 3))
        panel_height = 116
        panel_left = max(10, width - panel_width - 16)
        panel_top = max(62, height // 18)
        panel = output.copy()
        cv2.rectangle(
            panel,
            (panel_left, panel_top),
            (panel_left + panel_width, panel_top + panel_height),
            (10, 16, 22),
            -1,
        )
        output = cv2.addWeighted(panel, 0.74, output, 0.26, 0)
        cv2.rectangle(
            output,
            (panel_left, panel_top),
            (panel_left + panel_width, panel_top + panel_height),
            (120, 138, 130),
            1,
        )
        details = [
            f"Object inside zone: {primary.object_intrusion_ratio:.0%}",
            f"Exit area blocked: {primary.exit_blockage_ratio:.0%}",
            f"Duration: {primary.blocked_duration_seconds:.1f} s",
            f"Method: {'SAM mask' if primary.spatial_method == 'sam_mask' else 'YOLO bounding box'}",
            f"Status: {'Confirmed obstruction' if primary.is_blocking else 'Observed'}",
        ]
        cv2.putText(
            output,
            "SPATIAL METRICS",
            (panel_left + 12, panel_top + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 232, 226),
            1,
            cv2.LINE_AA,
        )
        for index, line in enumerate(details, start=1):
            cv2.putText(
                output,
                line,
                (panel_left + 12, panel_top + 20 + index * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 245, 242),
                1,
                cv2.LINE_AA,
            )

    if obstructions:
        legend_width = min(320, max(220, width // 3))
        legend_height = 74
        legend_left = 14
        legend_top = max(62, height - legend_height - 18)
        legend = output.copy()
        cv2.rectangle(
            legend,
            (legend_left, legend_top),
            (legend_left + legend_width, legend_top + legend_height),
            (8, 12, 16),
            -1,
        )
        output = cv2.addWeighted(legend, 0.72, output, 0.28, 0)
        cv2.rectangle(
            output,
            (legend_left, legend_top),
            (legend_left + legend_width, legend_top + legend_height),
            (120, 138, 130),
            1,
        )
        cv2.putText(
            output,
            "LEGEND",
            (legend_left + 12, legend_top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 232, 226),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(output, (legend_left + 12, legend_top + 28), (legend_left + 54, legend_top + 46), yolo_color, 2)
        cv2.putText(
            output,
            "Rectangle: YOLO detection",
            (legend_left + 64, legend_top + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 245, 242),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(output, (legend_left + 12, legend_top + 52), (legend_left + 54, legend_top + 68), (245, 168, 55), -1)
        cv2.rectangle(output, (legend_left + 12, legend_top + 52), (legend_left + 54, legend_top + 68), sam_outline, 2)
        cv2.putText(
            output,
            "Filled contour: SAM object mask",
            (legend_left + 64, legend_top + 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 245, 242),
            1,
            cv2.LINE_AA,
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
