import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import AnalysisJob, Camera, Incident
from app.services.cv.annotator import annotate, annotate_scene
from app.services.cv.detector import convert_yolo_results, suppress_duplicate_detections
from app.services.cv.roi_engine import (
    calculate_exit_blockage_ratio,
    calculate_object_intrusion_ratio,
    evaluate_detection,
    validate_polygon,
)
from app.services.cv.tracker import IoUTracker
from app.services.cv.types import Detection, Obstruction
from app.services.llm import _parse_result
from app.services.processing import (
    classify_scene_assessment,
    ground_scene_assessment,
    parse_scene_detections_payload,
    process_video_job,
    serialize_scene_detections,
    synthetic_demo_frame,
)
from app.services.scene_reasoning import (
    _extract_chat_content,
    _parse_assessment,
    _parse_fire_exit_validation,
)
from app.services.sop import search_sops


class FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class FakeYoloBox:
    def __init__(self, cls_id, confidence, xyxy):
        self.cls = FakeScalar(cls_id)
        self.conf = FakeScalar(confidence)
        self.xyxy = [np.array(xyxy, dtype=float)]


class FakeVideoCapture:
    def __init__(self, frames, fps=1.0):
        self.frames = list(frames)
        self.fps = fps
        self.index = 0

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0

    def release(self):
        return None


class FakeDetector:
    name = "yolo"

    def __init__(self, detections_per_frame):
        self.detections_per_frame = list(detections_per_frame)
        self.index = 0

    def detect(self, _image):
        idx = min(self.index, len(self.detections_per_frame) - 1)
        self.index += 1
        return [Detection(item.label, item.confidence, item.box) for item in self.detections_per_frame[idx]]


class RoiTests(unittest.TestCase):
    def test_yolo_result_conversion_to_detection_type(self):
        detections = convert_yolo_results(
            [FakeYoloBox(2, 0.81, [11, 22, 101, 202])],
            {2: "car"},
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "car")
        self.assertEqual(detections[0].box, (11, 22, 101, 202))
        self.assertAlmostEqual(detections[0].confidence, 0.81)

    def test_duplicate_overlapping_yolo_detections_are_suppressed(self):
        detections = suppress_duplicate_detections(
            [
                Detection("car", 0.53, (1, 171, 845, 1276)),
                Detection("truck", 0.44, (1, 173, 854, 1260)),
                Detection("person", 0.91, (880, 120, 950, 380)),
            ]
        )
        self.assertEqual(
            [(item.label, round(item.confidence, 2)) for item in detections],
            [("person", 0.91), ("car", 0.53)],
        )

    def test_object_intrusion_and_exit_blockage_ratios(self):
        polygon = validate_polygon([[0, 0], [100, 0], [100, 100], [0, 100]])
        outside = Detection("car", 0.9, (120, 120, 180, 180))
        partial = Detection("car", 0.9, (50, 50, 150, 150))
        inside = Detection("car", 0.9, (10, 10, 90, 90))

        self.assertEqual(calculate_object_intrusion_ratio(outside.box, polygon), 0.0)
        self.assertEqual(calculate_exit_blockage_ratio(outside.box, polygon), 0.0)
        self.assertAlmostEqual(calculate_object_intrusion_ratio(partial.box, polygon), 0.25)
        self.assertAlmostEqual(calculate_exit_blockage_ratio(partial.box, polygon), 0.25)
        self.assertAlmostEqual(calculate_object_intrusion_ratio(inside.box, polygon), 1.0)
        self.assertAlmostEqual(calculate_exit_blockage_ratio(inside.box, polygon), 0.64)

    def test_class_filtering_and_thresholds(self):
        polygon = validate_polygon([[0, 0], [100, 0], [100, 100], [0, 100]])
        result = evaluate_detection(
            Detection("backpack", 0.9, (20, 20, 80, 80)),
            polygon,
            {"car", "truck"},
            0.25,
            0.05,
        )
        self.assertFalse(result.is_blocking)
        result = evaluate_detection(
            Detection("car", 0.9, (80, 80, 150, 150)),
            polygon,
            {"car", "truck"},
            0.25,
            0.25,
        )
        self.assertFalse(result.is_blocking)

    def test_invalid_polygon_rejected(self):
        with self.assertRaises(ValueError):
            validate_polygon([[0, 0], [1, 1]])


class TrackerTests(unittest.TestCase):
    def test_track_id_persists_for_overlapping_detection(self):
        tracker = IoUTracker()
        first = tracker.update([Detection("car", 0.9, (10, 10, 100, 100))])[0]
        second = tracker.update([Detection("car", 0.9, (14, 12, 104, 102))])[0]
        self.assertEqual(first.track_id, second.track_id)


class ProcessingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        self.camera = Camera(
            id="cam-1",
            name="South Exit Camera",
            facility="Building A",
            zone="Fire Exit South",
            exit_zone=[[0, 0], [100, 0], [100, 100], [0, 100]],
            blocked_classes=["car", "truck", "bus", "motorcycle", "bicycle", "person"],
            confidence_threshold=0.35,
            minimum_overlap=0.25,
            persistence_seconds=1.0,
            alert_cooldown_seconds=300,
            enabled=True,
        )
        with self.SessionLocal() as db:
            db.add(self.camera)
            db.add(AnalysisJob(id="job-1", media_type="video", filename="test.mp4", camera_id=self.camera.id))
            db.commit()
        self.frames = [np.full((120, 120, 3), 40, dtype=np.uint8) for _ in range(4)]
        self.noop_notify = AsyncMock(side_effect=lambda db, incident, camera: incident)
        self.noop_broadcast = AsyncMock()

    async def _run_video(self, detections_per_frame, *, validate_result=None):
        fake_detector = FakeDetector(detections_per_frame)
        with patch("app.services.processing.SessionLocal", self.SessionLocal), patch(
            "app.services.processing.get_detector", return_value=fake_detector
        ), patch(
            "app.services.processing.cv2.VideoCapture",
            return_value=FakeVideoCapture(self.frames[: len(detections_per_frame)], fps=1.0),
        ), patch(
            "app.services.processing.enrich_and_notify", self.noop_notify
        ), patch(
            "app.services.processing.event_hub.broadcast", self.noop_broadcast
        ), patch(
            "app.services.processing.event_hub.broadcast_threadsafe"
        ), patch(
            "app.services.processing.settings.validate_fire_exit_incidents_with_vision",
            validate_result is not None,
        ), patch(
            "app.services.processing.settings.vision_validation_fail_closed", False
        ):
            if validate_result is None:
                await process_video_job("job-1", Path("video.mp4"), self.camera.exit_zone)
            else:
                with patch(
                    "app.services.processing.validate_fire_exit_obstruction",
                    AsyncMock(return_value=validate_result),
                ):
                    await process_video_job("job-1", Path("video.mp4"), self.camera.exit_zone)

    def _incident_count(self):
        return len(self._incidents())

    def _incidents(self):
        with self.SessionLocal() as db:
            return db.scalars(select(Incident).order_by(Incident.id)).all()

    async def test_persistence_threshold_creates_incident(self):
        inside = Detection("car", 0.9, (10, 10, 90, 90))
        await self._run_video([[inside], [inside]])
        self.assertEqual(len(self._incidents()), 1)

    async def test_timer_resets_when_object_leaves_zone(self):
        inside = Detection("car", 0.9, (10, 10, 90, 90))
        outside = Detection("car", 0.9, (110, 110, 150, 150))
        await self._run_video([[inside], [outside], [inside]])
        self.assertEqual(len(self._incidents()), 0)

    async def test_no_duplicate_incident_for_same_track(self):
        inside = Detection("car", 0.9, (10, 10, 90, 90))
        await self._run_video([[inside], [inside], [inside], [inside]])
        self.assertEqual(len(self._incidents()), 1)

    async def test_person_ignored_by_default(self):
        inside = Detection("person", 0.9, (10, 10, 90, 90))
        with patch("app.services.processing.settings.include_person_as_obstruction", False):
            await self._run_video([[inside], [inside], [inside]])
        self.assertEqual(len(self._incidents()), 0)

    async def test_person_uses_optional_longer_persistence(self):
        inside = Detection("person", 0.9, (10, 10, 90, 90))
        with patch("app.services.processing.settings.include_person_as_obstruction", True), patch(
            "app.services.processing.settings.person_minimum_duration_seconds", 2.5
        ):
            await self._run_video([[inside], [inside], [inside], [inside]])
        self.assertEqual(len(self._incidents()), 1)
        self.assertGreaterEqual(self._incidents()[0].duration_seconds, 2.5)

    async def test_gemma_validation_accepted(self):
        inside = Detection("car", 0.9, (10, 10, 90, 90))
        await self._run_video(
            [[inside], [inside]],
            validate_result={
                "confirmed": True,
                "category": "fire_exit_obstruction",
                "summary": "Vehicle blocks the zone.",
                "visible_evidence": ["Vehicle overlaps the marked area"],
                "confidence": 0.92,
            },
        )
        incidents = self._incidents()
        self.assertEqual(len(incidents), 1)
        self.assertTrue(incidents[0].incident_metadata["vision_validation"]["confirmed"])

    async def test_gemma_validation_rejected(self):
        inside = Detection("car", 0.9, (10, 10, 90, 90))
        await self._run_video(
            [[inside], [inside]],
            validate_result={
                "confirmed": False,
                "category": "fire_exit_obstruction",
                "summary": "No obstruction confirmed.",
                "visible_evidence": ["Object does not block the route"],
                "confidence": 0.28,
            },
        )
        self.assertEqual(len(self._incidents()), 0)

    async def test_gemma_unavailable_fail_open(self):
        inside = Detection("car", 0.9, (10, 10, 90, 90))
        fake_detector = FakeDetector([[inside], [inside]])
        with patch("app.services.processing.SessionLocal", self.SessionLocal), patch(
            "app.services.processing.get_detector", return_value=fake_detector
        ), patch(
            "app.services.processing.cv2.VideoCapture",
            return_value=FakeVideoCapture(self.frames[:2], fps=1.0),
        ), patch(
            "app.services.processing.enrich_and_notify", self.noop_notify
        ), patch(
            "app.services.processing.event_hub.broadcast", self.noop_broadcast
        ), patch(
            "app.services.processing.event_hub.broadcast_threadsafe"
        ), patch(
            "app.services.processing.settings.validate_fire_exit_incidents_with_vision", True
        ), patch(
            "app.services.processing.settings.vision_validation_fail_closed", False
        ), patch(
            "app.services.processing.validate_fire_exit_obstruction",
            AsyncMock(side_effect=RuntimeError("vision unavailable")),
        ):
            await process_video_job("job-1", Path("video.mp4"), self.camera.exit_zone)
        incidents = self._incidents()
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0].incident_metadata["vision_validation"]["mode"], "unavailable")

    async def test_gemma_unavailable_fail_closed(self):
        inside = Detection("car", 0.9, (10, 10, 90, 90))
        fake_detector = FakeDetector([[inside], [inside]])
        with patch("app.services.processing.SessionLocal", self.SessionLocal), patch(
            "app.services.processing.get_detector", return_value=fake_detector
        ), patch(
            "app.services.processing.cv2.VideoCapture",
            return_value=FakeVideoCapture(self.frames[:2], fps=1.0),
        ), patch(
            "app.services.processing.enrich_and_notify", self.noop_notify
        ), patch(
            "app.services.processing.event_hub.broadcast", self.noop_broadcast
        ), patch(
            "app.services.processing.event_hub.broadcast_threadsafe"
        ), patch(
            "app.services.processing.settings.validate_fire_exit_incidents_with_vision", True
        ), patch(
            "app.services.processing.settings.vision_validation_fail_closed", True
        ), patch(
            "app.services.processing.validate_fire_exit_obstruction",
            AsyncMock(side_effect=RuntimeError("vision unavailable")),
        ):
            await process_video_job("job-1", Path("video.mp4"), self.camera.exit_zone)
        self.assertEqual(len(self._incidents()), 0)


class ApiFallbackTests(unittest.TestCase):
    def test_image_analysis_without_polygon_helper(self):
        from app.api import analyse_uploaded_image

        class DummyUpload:
            filename = "frame.jpg"

        camera = Camera(
            id="cam-api",
            name="Cam",
            facility="Building A",
            zone="Zone A",
            exit_zone=[],
            blocked_classes=[],
            confidence_threshold=0.35,
            minimum_overlap=0.25,
            persistence_seconds=5,
            alert_cooldown_seconds=300,
            enabled=True,
        )
        image = np.full((20, 20, 3), 60, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        db.add(camera)
        db.commit()
        try:
            with patch("app.api._read_upload", AsyncMock(return_value=encoded.tobytes())), patch(
                "app.api.detect_scene_objects",
                return_value=("yolo", []),
            ), patch(
                "app.api.assess_scene",
                AsyncMock(
                    return_value={
                        "violation": False,
                        "category": "General",
                        "summary": "No visible issue.",
                        "evidence": [],
                        "confidence": 0.2,
                        "visible_objects": [],
                        "supporting_objects": [],
                        "annotations": [],
                        "model": "gemma3:27b",
                        "local": True,
                    }
                ),
            ), patch("app.api.create_scene_incident", AsyncMock(return_value=None)):
                result = asyncio.run(analyse_uploaded_image(DummyUpload(), "cam-api", "", db))
            self.assertEqual(result["summary"], "No visible issue.")
            self.assertEqual(result["incidents"], [])
        finally:
            db.close()

    def test_scene_detect_helper_returns_grounded_detections(self):
        from app.api import detect_scene

        class DummyUpload:
            filename = "frame.jpg"

        camera = Camera(
            id="cam-detect",
            name="Cam",
            facility="Building A",
            zone="Zone A",
            exit_zone=[],
            blocked_classes=[],
            confidence_threshold=0.35,
            minimum_overlap=0.25,
            persistence_seconds=5,
            alert_cooldown_seconds=300,
            enabled=True,
        )
        image = np.full((20, 20, 3), 60, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        db.add(camera)
        db.commit()
        try:
            detections = [Detection("car", 0.91, (2, 3, 10, 18))]
            with patch("app.api._read_upload", AsyncMock(return_value=encoded.tobytes())), patch(
                "app.api.detect_scene_objects",
                return_value=("yolo", detections),
            ):
                result = asyncio.run(detect_scene(DummyUpload(), "cam-detect", db))
            self.assertEqual(result["provider"], "yolo")
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["scene_detections"][0]["label"], "car")
        finally:
            db.close()


class SupportServiceTests(unittest.TestCase):
    def test_scene_assessment_parser_normalizes_model_json(self):
        result = _parse_assessment(
            '{"violation":true,"category":"Parking","summary":"Car in a no-parking '
            'zone.","evidence":["No Parking sign"],"confidence":1.4,'
            '"visible_objects":["car","sign"],"supporting_objects":["car"],'
            '"annotations":[{"label":"car","box":[100,200,800,900]}]}'
        )
        self.assertTrue(result["violation"])
        self.assertEqual(result["category"], "Parking")
        self.assertEqual(result["confidence"], 1.0)
        self.assertEqual(result["annotations"][0]["label"], "car")
        self.assertEqual(result["supporting_objects"], ["car"])

    def test_scene_assessment_parser_accepts_thinking_prefix_and_scalar_lists(self):
        result = _parse_assessment(
            '<think>Inspecting the image.</think>\n```json\n'
            '{"violation":"true","category":"Exit blocked","summary":"A car blocks the exit.",'
            '"evidence":"FIRE EXIT sign visible; vehicle overlaps the marked zone",'
            '"confidence":"0.91","visible_objects":"car, fire exit sign",'
            '"supporting_objects":"car",'
            '"annotations":{"label":"car","box":[120,220,820,940]}}\n```'
        )
        self.assertTrue(result["violation"])
        self.assertEqual(
            result["evidence"],
            ["FIRE EXIT sign visible", "vehicle overlaps the marked zone"],
        )
        self.assertEqual(result["visible_objects"], ["car", "fire exit sign"])
        self.assertEqual(result["supporting_objects"], ["car"])
        self.assertEqual(result["annotations"][0]["box"], [120.0, 220.0, 820.0, 940.0])

    def test_fire_exit_validation_parser(self):
        result = _parse_fire_exit_validation(
            '{"confirmed":true,"category":"fire_exit_obstruction","summary":"Vehicle blocks the zone.",'
            '"visible_evidence":["Vehicle overlaps the marked restricted area"],"confidence":1.2}'
        )
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["category"], "fire_exit_obstruction")
        self.assertEqual(result["confidence"], 1.0)

    def test_fire_exit_validation_parser_accepts_chunked_chat_content(self):
        content = _extract_chat_content(
            {
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '<think>Checking overlap.</think>\n'
                                '{"confirmed":"yes","category":"fire_exit_obstruction",'
                                '"summary":"Vehicle blocks the route.",'
                                '"visible_evidence":"Vehicle overlaps the doorway",'
                                '"confidence":"0.84"}'
                            ),
                        }
                    ]
                }
            }
        )
        result = _parse_fire_exit_validation(content)
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["visible_evidence"], ["Vehicle overlaps the doorway"])
        self.assertAlmostEqual(result["confidence"], 0.84)

    def test_sop_retrieval_prefers_relevant_procedure(self):
        results = search_sops("fire_exit_obstruction", "Building A", "vehicle")
        self.assertTrue(results)
        self.assertIn("Fire Exit", results[0].title)

    def test_demo_frame_and_annotation_are_valid(self):
        image = synthetic_demo_frame()
        self.assertEqual(image.shape, (720, 1100, 3))
        obstruction = Obstruction(
            Detection("vehicle", 0.9, (350, 350, 780, 625)),
            0.9,
            0.9,
            0.35,
            True,
            5.0,
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
            "annotations": [{"label": "car", "box": [200, 250, 700, 850]}],
        }
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "scene-annotated.jpg"
            annotate_scene(image, assessment, destination)
            result = cv2.imread(str(destination))
            self.assertIsNotNone(result)
            self.assertFalse(np.array_equal(image, result))
            self.assertTrue(np.array_equal(result[0, 0], image[0, 0]))
            self.assertFalse(np.array_equal(result[120, 128], image[120, 128]))

    def test_agent_parser_accepts_local_model_thinking_prefix(self):
        result = _parse_result(
            '<think>Used the local SOP tool.</think>\n'
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
                "supporting_objects": ["car"],
                "scene_detections": [
                    {"label": "car", "confidence": 0.94, "box": [10, 20, 90, 100]}
                ],
            }
        )
        self.assertEqual(event_type, "exit_blocked")
        self.assertEqual(object_type, "car")

    def test_scene_classification_prefers_yolo_label_over_vision_wording(self):
        assessment = {
            "category": "Fire Safety",
            "summary": "A vehicle obstructs the fire exit while a hand truck is nearby.",
            "evidence": [
                "A vehicle is parked directly in front of the fire exit.",
                "A hand truck is visible beside the doorway.",
            ],
            "visible_objects": ["Fire Exit", "vehicle", "Hand Truck"],
            "supporting_objects": ["car"],
            "scene_detections": [
                {"label": "car", "confidence": 0.92, "box": [10, 10, 90, 90]}
            ],
        }
        event_type, object_type = classify_scene_assessment(assessment)
        self.assertEqual(event_type, "exit_blocked")
        self.assertEqual(object_type, "car")

    def test_scene_grounding_prefers_detector_boxes_over_model_boxes(self):
        image = np.full((200, 300, 3), 50, dtype=np.uint8)
        assessment = {
            "violation": True,
            "category": "Restricted parking",
            "summary": "A car is parked in a restricted access area.",
            "evidence": ["A car is visible in the restricted lane."],
            "visible_objects": ["car", "sign"],
            "supporting_objects": ["car"],
            "annotations": [{"label": "car", "box": [700, 700, 950, 950]}],
        }
        grounded = ground_scene_assessment(
            image,
            assessment,
            [Detection("car", 0.93, (30, 40, 180, 170))],
            "yolo",
        )

        self.assertEqual(grounded["grounded_annotation_source"], "yolo")
        self.assertEqual(grounded["annotations"][0]["label"], "car")
        self.assertEqual(grounded["annotations"][0]["box"], [100.0, 200.0, 600.0, 850.0])
        self.assertEqual(grounded["model_annotations"][0]["box"], [700, 700, 950, 950])

    def test_scene_grounding_suppresses_unrelated_detector_boxes(self):
        image = np.full((200, 300, 3), 50, dtype=np.uint8)
        assessment = {
            "violation": True,
            "category": "Fire Safety",
            "summary": "The fire exit is obstructed by boxes and pallets.",
            "evidence": ["Boxes and pallets are stacked in front of the fire exit."],
            "visible_objects": ["Boxes", "Pallets", "Hand Truck"],
            "supporting_objects": ["box"],
            "annotations": [{"label": "Boxes", "box": [600, 200, 800, 400]}],
        }
        grounded = ground_scene_assessment(
            image,
            assessment,
            [Detection("car", 0.91, (40, 30, 220, 180))],
            "yolo",
        )

        self.assertEqual(grounded["annotations"], [])
        self.assertEqual(grounded["grounded_annotation_source"], "yolo")

    def test_scene_detection_payload_round_trip(self):
        detections = [
            Detection("car", 0.914, (10, 20, 90, 100)),
            Detection("person", 0.621, (2, 4, 30, 70)),
        ]
        payload = serialize_scene_detections(detections)
        restored = parse_scene_detections_payload(json.dumps(payload))
        self.assertEqual([item.label for item in restored], ["car", "person"])
        self.assertEqual(restored[0].box, (10, 20, 90, 100))
        self.assertAlmostEqual(restored[1].confidence, 0.621)

    def test_sop_retrieval_prefers_fire_exit_playbook_for_box_obstruction(self):
        results = search_sops("fire_exit_obstruction", "Building A", "box")
        self.assertTrue(results)
        self.assertEqual(results[0].title, "Fire Exit Obstruction")


if __name__ == "__main__":
    unittest.main()
