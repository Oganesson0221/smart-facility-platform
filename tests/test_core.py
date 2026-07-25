import asyncio
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.services.cv.annotator import annotate
from app.services.cv.roi_engine import calculate_overlap, evaluate_detection, validate_polygon
from app.services.cv.tracker import IoUTracker
from app.services.cv.types import Detection, Obstruction
from app.services.processing import classify_scene_assessment, synthetic_demo_frame
from app.services.scene_reasoning import _parse_assessment
from app.services.cv.annotator import annotate_scene
from app.services.llm import _parse_result
from app.services.sop import search_sops


class RoiTests(unittest.TestCase):
    def test_overlap_and_blocking_rule(self):
        polygon = validate_polygon([[0, 0], [100, 0], [100, 100], [0, 100]])
        detection = Detection("vehicle", 0.9, (50, 50, 150, 150))
        self.assertAlmostEqual(calculate_overlap(detection.box, polygon), 0.25)
        result = evaluate_detection(detection, polygon, {"vehicle"}, 0.25)
        self.assertTrue(result.is_blocking)

    def test_non_blocked_class_does_not_trigger(self):
        polygon = validate_polygon([[0, 0], [100, 0], [100, 100], [0, 100]])
        result = evaluate_detection(
            Detection("person", 0.9, (10, 10, 90, 90)), polygon, {"vehicle"}, 0.1
        )
        self.assertFalse(result.is_blocking)

    def test_invalid_polygon_rejected(self):
        with self.assertRaises(ValueError):
            validate_polygon([[0, 0], [1, 1]])


class TrackerTests(unittest.TestCase):
    def test_track_id_persists_for_overlapping_detection(self):
        tracker = IoUTracker()
        first = tracker.update([Detection("vehicle", 0.9, (10, 10, 100, 100))])[0]
        second = tracker.update([Detection("vehicle", 0.9, (14, 12, 104, 102))])[0]
        self.assertEqual(first.track_id, second.track_id)


class SupportServiceTests(unittest.TestCase):
    def test_scene_assessment_parser_normalizes_model_json(self):
        result = _parse_assessment(
            '{"violation":true,"category":"Parking","summary":"Car in a no-parking '
            'zone.","evidence":["No Parking sign"],"confidence":1.4,'
            '"visible_objects":["car","sign"],'
            '"annotations":[{"label":"car","box":[100,200,800,900]}]}'
        )
        self.assertTrue(result["violation"])
        self.assertEqual(result["category"], "Parking")
        self.assertEqual(result["confidence"], 1.0)
        self.assertEqual(result["annotations"][0]["label"], "car")

    def test_sop_retrieval_prefers_relevant_procedure(self):
        results = search_sops("exit_blocked", "Building A", "vehicle")
        self.assertTrue(results)
        self.assertIn("Exit", results[0].title)

    def test_demo_frame_and_annotation_are_valid(self):
        image = synthetic_demo_frame()
        self.assertEqual(image.shape, (720, 1100, 3))
        obstruction = Obstruction(
            Detection("vehicle", 0.9, (350, 350, 780, 625)), 0.9, True
        )
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "annotated.jpg"
            annotate(
                image,
                [[250, 100], [850, 100], [850, 680], [250, 680]],
                [obstruction],
                destination,
            )
            result = cv2.imread(str(destination))
            self.assertIsNotNone(result)

    def test_scene_violation_annotation_is_valid(self):
        image = np.full((480, 640, 3), 40, dtype=np.uint8)
        assessment = {
            "violation": True,
            "category": "No parking",
            "summary": "A vehicle is parked beside a visible no-parking sign.",
        }
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "scene-annotated.jpg"
            annotate_scene(image, assessment, destination)
            result = cv2.imread(str(destination))
            self.assertIsNotNone(result)
            self.assertFalse(np.array_equal(image, result))

    def test_agent_parser_accepts_local_model_thinking_prefix(self):
        result = _parse_result(
            '<think>Used the local SOP tool.</think>\\n'
            '{"summary":"Vehicle blocks the exit.",'
            '"recommended_action":"Notify Facilities Security."}',
            ("fallback summary", "fallback action"),
        )
        self.assertEqual(result[0], "Vehicle blocks the exit.")
        self.assertEqual(result[1], "Notify Facilities Security.")

    def test_scene_classification_maps_visible_exit_obstruction_to_sop_event(self):
        event_type, object_type = classify_scene_assessment(
            {
                "category": "Safety Violation",
                "summary": "A vehicle obstructs the fire exit.",
                "evidence": ["FIRE EXIT sign is visible"],
                "visible_objects": ["FIRE EXIT sign", "vehicle"],
            }
        )
        self.assertEqual(event_type, "exit_blocked")
        self.assertEqual(object_type, "vehicle")


if __name__ == "__main__":
    unittest.main()
