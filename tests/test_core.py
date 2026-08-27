import asyncio
from contextlib import nullcontext
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import httpx
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import Base
from app.models import AnalysisJob, Camera, Incident
from app.services.cv.annotator import annotate, annotate_scene
from app.services.cv.detector import (
    YoloDetector,
    convert_yolo_results,
    suppress_duplicate_detections,
)
from app.services.cv.segmenter import (
    Sam2Segmenter,
    box_centroid_point,
    clamp_box_to_image,
    reset_segmenter_cache,
)
from app.services.cv.roi_engine import (
    box_intersects_or_near_polygon,
    calculate_box_zone_iou,
    calculate_exit_blockage_ratio,
    calculate_mask_spatial_metrics,
    calculate_object_intrusion_ratio,
    evaluate_detection,
    validate_polygon,
)
from app.services.cv.tracker import IoUTracker
from app.services.cv.types import Detection, Obstruction, SegmentationResult
from app.services.agent import enrich_and_notify
from app.services.llm import llm_runtime_status
from app.services.llm import _parse_result
from app.services.llm import answer_sop_question_direct
from app.services.llm import create_grounded_summary
from app.services.llm import SYSTEM_PROMPT
from app.services.llm import TELEGRAM_ASSISTANT_PROMPT
from app.services.nemo_agent_client import _parse_json_content
from app.services.processing import (
    _confirm_fire_exit_candidate,
    _crop_validation_region,
    _evaluate_obstructions,
    _prepare_validation_image,
    analyse_image,
    classify_scene_assessment,
    ground_scene_assessment,
    parse_scene_detections_payload,
    preview_image_analysis,
    process_video_job,
    serialize_scene_detections,
    synthetic_demo_frame,
)
from app.services.scene_reasoning import (
    _build_fire_exit_validation_prompt,
    _call_vision_model,
    _extract_chat_content,
    _parse_assessment,
    _parse_fire_exit_validation,
    _prepare_vision_image_bytes,
    validate_fire_exit_obstruction,
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


class FakeSegmenter:
    name = "sam2"

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def segment(self, image, box):
        self.calls.append((image.shape, box))
        if self.error is not None:
            raise self.error
        return self.result


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
        self.assertAlmostEqual(calculate_box_zone_iou(inside.box, polygon), 0.64)

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

    def test_full_frame_mode_can_treat_any_detected_class_as_obstruction_candidate(self):
        polygon = validate_polygon([[0, 0], [100, 0], [100, 100], [0, 100]])
        result = evaluate_detection(
            Detection("chair", 0.9, (20, 20, 80, 80)),
            polygon,
            {"car", "truck"},
            0.25,
            0.05,
            allow_all_classes=True,
        )
        self.assertTrue(result.is_blocking)

    def test_invalid_polygon_rejected(self):
        with self.assertRaises(ValueError):
            validate_polygon([[0, 0], [1, 1]])

    def test_mask_spatial_metrics(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True
        metrics = calculate_mask_spatial_metrics(
            mask,
            [[0, 0], [49, 0], [49, 99], [0, 99]],
        )
        self.assertAlmostEqual(metrics["object_intrusion_ratio"], 0.5, places=2)
        self.assertAlmostEqual(metrics["exit_blockage_ratio"], 0.25, places=2)
        self.assertAlmostEqual(metrics["mask_zone_iou"], 0.2, places=2)

    def test_box_precheck_detects_near_zone_candidates(self):
        polygon = validate_polygon([[0, 0], [100, 0], [100, 100], [0, 100]])
        self.assertFalse(box_intersects_or_near_polygon((150, 150, 200, 200), polygon, 20))
        self.assertTrue(box_intersects_or_near_polygon((102, 20, 140, 70), polygon, 5))

    def test_full_frame_validation_preserves_complete_exit_context(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        polygon = [[0, 0], [299, 0], [299, 199], [0, 199]]
        crop = _crop_validation_region(image, polygon, (120, 70, 180, 140))
        self.assertEqual(crop.shape, image.shape)

    def test_drawn_zone_validation_stays_cropped_to_zone_and_detection(self):
        image = np.zeros((400, 500, 3), dtype=np.uint8)
        polygon = [[100, 80], [300, 80], [300, 280], [100, 280]]
        crop = _crop_validation_region(image, polygon, (250, 220, 340, 330))
        self.assertLess(crop.shape[0], image.shape[0])
        self.assertLess(crop.shape[1], image.shape[1])
        self.assertGreaterEqual(crop.shape[0], 250)
        self.assertGreaterEqual(crop.shape[1], 240)

    def test_validation_image_is_downscaled_before_nemotron(self):
        image = np.zeros((1800, 1200, 3), dtype=np.uint8)
        with patch("app.services.processing.settings.vision_validation_image_max_dim", 512):
            resized = _prepare_validation_image(image)
        self.assertLessEqual(max(resized.shape[:2]), 512)


class SegmenterTests(unittest.TestCase):
    def tearDown(self):
        reset_segmenter_cache()

    def test_sam_box_prompt_is_clamped_and_expanded(self):
        box = clamp_box_to_image((-10, 5, 120, 90), (100, 100, 3), expand_ratio=0.1)
        self.assertEqual(box, (0, 0, 100, 98))

    def test_box_centroid_prompt_uses_yolo_box_center(self):
        self.assertEqual(box_centroid_point((10, 20, 50, 80)), (30, 50))

    def test_sam_mask_cleanup_prefers_region_matching_prompt_box(self):
        segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
        mask = np.zeros((80, 80), dtype=bool)
        mask[5:18, 5:18] = True
        mask[20:55, 22:58] = True
        cleaned = segmenter._cleanup_mask(mask, (18, 18, 60, 60))
        self.assertTrue(cleaned[30, 30])
        self.assertFalse(cleaned[10, 10])

    def test_sam_mask_cleanup_can_prefer_region_containing_prompt_point(self):
        segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
        mask = np.zeros((80, 80), dtype=bool)
        mask[5:18, 5:18] = True
        mask[20:55, 22:58] = True
        cleaned = segmenter._cleanup_mask(
            mask,
            (5, 5, 58, 58),
            prompt_point=(10, 10),
        )
        self.assertTrue(cleaned[10, 10])
        self.assertFalse(cleaned[30, 30])

    def test_sam_mask_converts_to_polygon(self):
        segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
        mask = np.zeros((60, 60), dtype=bool)
        mask[10:40, 12:38] = True
        polygon = segmenter._mask_to_polygon(mask, (10, 10, 40, 40), (60, 60))
        self.assertGreaterEqual(len(polygon), 4)

    def test_empty_mask_polygon_conversion_fails(self):
        segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
        with self.assertRaises(RuntimeError):
            segmenter._mask_to_polygon(np.zeros((20, 20), dtype=bool), (1, 1, 10, 10), (20, 20))

    def test_segmenter_is_loaded_lazily(self):
        from app.services.cv import segmenter as segmenter_module

        with patch.object(segmenter_module.settings, "sam_enabled", True), patch.object(
            segmenter_module.settings, "sam_provider", "sam2"
        ), patch.object(segmenter_module, "Sam2Segmenter") as mock_segmenter:
            reset_segmenter_cache()
            instance = object()
            mock_segmenter.return_value = instance
            self.assertIs(segmenter_module.get_segmenter(), instance)
            self.assertIs(segmenter_module.get_segmenter(), instance)
            mock_segmenter.assert_called_once()

    def test_segmenter_passes_positive_centroid_prompt_to_predictor(self):
        class FakePredictor:
            def __init__(self):
                self.image_shape = None
                self.kwargs = None

            def set_image(self, image):
                self.image_shape = image.shape

            def predict(self, **kwargs):
                self.kwargs = kwargs
                mask = np.zeros((1, 60, 60), dtype=bool)
                mask[0, 10:40, 12:38] = True
                return mask, np.array([0.92], dtype=float), None

        segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
        segmenter.torch = SimpleNamespace(inference_mode=lambda: nullcontext())
        segmenter._autocast_context = lambda: nullcontext()
        segmenter.predictor = FakePredictor()
        segmenter.model_name = "sam2.1_hiera_tiny (cpu)"
        image = np.zeros((60, 60, 3), dtype=np.uint8)

        with patch("app.services.cv.segmenter.settings.sam_prompt_box_expand_ratio", 0.0):
            result = segmenter.segment(image, (10, 10, 40, 40))

        point_coords = np.asarray(segmenter.predictor.kwargs["point_coords"]).reshape(-1, 2)
        point_labels = np.asarray(segmenter.predictor.kwargs["point_labels"]).reshape(-1)
        self.assertEqual(tuple(int(value) for value in point_coords[0]), (25, 25))
        self.assertEqual(point_labels.tolist(), [1])
        self.assertEqual(result.prompt_point, (25, 25))


class RuntimeSafetyTests(unittest.TestCase):
    @staticmethod
    def _fake_torch(*, cuda_available=True, on_empty_cache=None):
        return SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: cuda_available,
                empty_cache=on_empty_cache or (lambda: None),
            )
        )

    def test_yolo_auto_device_uses_cpu_when_both_vllm_servers_are_enabled(self):
        detector = YoloDetector.__new__(YoloDetector)
        detector.torch = self._fake_torch()
        with patch("app.services.cv.detector.settings.yolo_device", "auto"), patch(
            "app.services.cv.detector.settings.device", "auto"
        ), patch(
            "app.services.cv.detector.settings.cv_shared_gpu_safe_mode", True
        ), patch(
            "app.services.cv.detector.settings.llm_enabled", True
        ), patch(
            "app.services.cv.detector.settings.vision_enabled", True
        ):
            self.assertEqual(detector._resolve_device(), "cpu")

    def test_explicit_yolo_cuda_device_overrides_shared_gpu_safe_mode(self):
        detector = YoloDetector.__new__(YoloDetector)
        detector.torch = self._fake_torch()
        with patch("app.services.cv.detector.settings.yolo_device", "cuda:0"), patch(
            "app.services.cv.detector.settings.cv_shared_gpu_safe_mode", True
        ):
            self.assertEqual(detector._resolve_device(), "cuda:0")

    def test_sam_auto_device_uses_cpu_when_both_vllm_servers_are_enabled(self):
        segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
        segmenter.torch = self._fake_torch()
        with patch("app.services.cv.segmenter.settings.sam_device", "auto"), patch(
            "app.services.cv.segmenter.settings.cv_shared_gpu_safe_mode", True
        ), patch(
            "app.services.cv.segmenter.settings.llm_enabled", True
        ), patch(
            "app.services.cv.segmenter.settings.vision_enabled", True
        ):
            self.assertEqual(segmenter._resolve_device(), "cpu")

    def test_yolo_cuda_oom_retries_once_on_cpu(self):
        events = []

        class FakeModel:
            def predict(self, **kwargs):
                events.append(f"predict:{kwargs['device']}")
                if kwargs["device"].startswith("cuda"):
                    raise RuntimeError("CUDA error: out of memory")
                return []

            def to(self, device):
                events.append(f"move:{device}")

        detector = YoloDetector.__new__(YoloDetector)
        detector.device = "cuda:0"
        detector.model = FakeModel()
        detector.class_names = {}
        detector.torch = self._fake_torch(
            on_empty_cache=lambda: events.append("empty_cache")
        )

        self.assertEqual(detector.detect(np.zeros((20, 20, 3), dtype=np.uint8)), [])
        self.assertEqual(
            events,
            ["predict:cuda:0", "move:cpu", "empty_cache", "predict:cpu"],
        )
        self.assertEqual(detector.device, "cpu")

    def test_nemo_non_json_tool_failure_has_actionable_error(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "NeMo workflow returned non-JSON content: <empty response>",
        ):
            _parse_json_content("")

    def test_text_only_nemo_prompts_forbid_unrelated_tool_calls(self):
        self.assertIn("Do not call any tool", SYSTEM_PROMPT)
        self.assertIn("Do not call any tool", TELEGRAM_ASSISTANT_PROMPT)


class TrackerTests(unittest.TestCase):
    def test_track_id_persists_for_overlapping_detection(self):
        tracker = IoUTracker()
        first = tracker.update([Detection("car", 0.9, (10, 10, 100, 100))])[0]
        second = tracker.update([Detection("car", 0.9, (14, 12, 104, 102))])[0]
        self.assertEqual(first.track_id, second.track_id)


class ProcessingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Unit tests use their local detector/segmenter fakes by default. Keep
        # an enabled deployment .env from sending accidental live NeMo calls;
        # orchestration-specific tests opt back in with their own patch.
        nemo_agent_patch = patch(
            "app.services.processing.settings.nemo_agent_enabled", False
        )
        nemo_agent_patch.start()
        self.addCleanup(nemo_agent_patch.stop)
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
            "app.services.processing.settings.sam_enabled", False,
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

    async def test_nemotron_validation_accepted(self):
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

    async def test_high_sam_iou_skips_nemotron_validation(self):
        obstruction = Obstruction(
            detection=Detection("car", 0.9, (10, 10, 90, 90)),
            overlap=0.9,
            object_intrusion_ratio=0.9,
            exit_blockage_ratio=0.8,
            is_blocking=True,
            mask_zone_iou=0.70,
        )
        validator = AsyncMock()
        with patch(
            "app.services.processing.settings.validate_fire_exit_incidents_with_vision",
            True,
        ), patch(
            "app.services.processing.settings.vision_validation_iou_threshold", 0.70
        ), patch(
            "app.services.processing.validate_fire_exit_obstruction", validator
        ):
            confirmed, validation, crop = await _confirm_fire_exit_candidate(
                None, self.frames[0], obstruction, self.camera.exit_zone
            )
        self.assertTrue(confirmed)
        self.assertEqual(validation["mode"], "deterministic_iou")
        self.assertIsNone(crop)
        validator.assert_not_awaited()

    async def test_low_sam_iou_escalates_to_nemotron(self):
        obstruction = Obstruction(
            detection=Detection("car", 0.9, (10, 10, 90, 90)),
            overlap=0.9,
            object_intrusion_ratio=0.9,
            exit_blockage_ratio=0.3,
            is_blocking=True,
            mask_zone_iou=0.69,
        )
        validator = AsyncMock(return_value={"confirmed": True, "confidence": 0.91})
        with patch(
            "app.services.processing.settings.validate_fire_exit_incidents_with_vision",
            True,
        ), patch(
            "app.services.processing.settings.vision_validation_iou_threshold", 0.70
        ), patch(
            "app.services.processing.validate_fire_exit_obstruction", validator
        ):
            confirmed, validation, crop = await _confirm_fire_exit_candidate(
                None, self.frames[0], obstruction, self.camera.exit_zone
            )
        self.assertTrue(confirmed)
        self.assertTrue(validation["confirmed"])
        self.assertIsNotNone(crop)
        validator.assert_awaited_once()

    async def test_nemotron_validation_rejected(self):
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

    async def test_nemotron_unavailable_fail_open(self):
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
            "app.services.processing.settings.sam_enabled", False
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

    async def test_nemotron_unavailable_fail_closed(self):
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
            "app.services.processing.settings.sam_enabled", False
        ), patch(
            "app.services.processing.settings.vision_validation_fail_closed", True
        ), patch(
            "app.services.processing.validate_fire_exit_obstruction",
            AsyncMock(side_effect=RuntimeError("vision unavailable")),
        ):
            await process_video_job("job-1", Path("video.mp4"), self.camera.exit_zone)
        self.assertEqual(len(self._incidents()), 0)

    def test_sam_disabled_keeps_yolo_box_path(self):
        image = np.full((120, 120, 3), 40, dtype=np.uint8)
        detection = Detection("car", 0.9, (10, 10, 90, 90))
        with patch("app.services.processing.settings.sam_enabled", False):
            _, obstructions = _evaluate_obstructions(
                self.camera,
                [detection],
                self.camera.exit_zone,
                image,
            )
        self.assertEqual(obstructions[0].spatial_method, "yolo_box_fallback")
        self.assertIsNone(obstructions[0].segmentation)

    def test_no_sam_call_for_detection_outside_zone(self):
        image = np.full((120, 120, 3), 40, dtype=np.uint8)
        detection = Detection("car", 0.9, (145, 145, 170, 170))
        fake_segmenter = FakeSegmenter()
        with patch("app.services.processing.settings.sam_enabled", True), patch(
            "app.services.processing.settings.sam_only_for_zone_candidates", True
        ), patch(
            "app.services.processing.get_segmenter", return_value=fake_segmenter
        ):
            _, obstructions = _evaluate_obstructions(
                self.camera,
                [detection],
                self.camera.exit_zone,
                image,
            )
        self.assertEqual(fake_segmenter.calls, [])
        self.assertEqual(obstructions[0].spatial_method, "yolo_box_fallback")

    def test_no_sam_call_for_irrelevant_class(self):
        image = np.full((120, 120, 3), 40, dtype=np.uint8)
        detection = Detection("backpack", 0.9, (10, 10, 90, 90))
        fake_segmenter = FakeSegmenter()
        with patch("app.services.processing.settings.sam_enabled", True), patch(
            "app.services.processing.get_segmenter", return_value=fake_segmenter
        ):
            _, obstructions = _evaluate_obstructions(
                self.camera,
                [detection],
                self.camera.exit_zone,
                image,
            )
        self.assertEqual(fake_segmenter.calls, [])
        self.assertFalse(obstructions[0].is_blocking)

    def test_sam_runs_for_zone_candidate_and_uses_mask_metrics(self):
        image = np.full((120, 120, 3), 40, dtype=np.uint8)
        detection = Detection("car", 0.9, (45, 10, 95, 90))
        mask = np.zeros((120, 120), dtype=bool)
        mask[10:90, 45:95] = True
        segmentation = SegmentationResult(
            mask=mask,
            polygon=[(45, 10), (95, 10), (95, 90), (45, 90)],
            area_pixels=int(mask.sum()),
            prompt_box=detection.box,
            score=0.88,
            model_name="sam2.1_hiera_tiny",
            inference_ms=14.2,
        )
        fake_segmenter = FakeSegmenter(result=segmentation)
        with patch("app.services.processing.settings.sam_enabled", True), patch(
            "app.services.processing.get_segmenter", return_value=fake_segmenter
        ):
            _, obstructions = _evaluate_obstructions(
                self.camera,
                [detection],
                self.camera.exit_zone,
                image,
            )
        self.assertEqual(len(fake_segmenter.calls), 1)
        self.assertEqual(obstructions[0].spatial_method, "sam_mask")
        self.assertIsNotNone(obstructions[0].segmentation)
        self.assertGreater(obstructions[0].mask_zone_iou, 0)

    def test_segmentation_failure_fail_open_falls_back_to_yolo(self):
        image = np.full((120, 120, 3), 40, dtype=np.uint8)
        detection = Detection("car", 0.9, (10, 10, 90, 90))
        fake_segmenter = FakeSegmenter(error=RuntimeError("checkpoint missing"))
        with patch("app.services.processing.settings.sam_enabled", True), patch(
            "app.services.processing.settings.sam_fail_open", True
        ), patch(
            "app.services.processing.get_segmenter", return_value=fake_segmenter
        ):
            _, obstructions = _evaluate_obstructions(
                self.camera,
                [detection],
                self.camera.exit_zone,
                image,
            )
        self.assertEqual(obstructions[0].spatial_method, "yolo_box_fallback")
        self.assertTrue(obstructions[0].is_blocking)
        self.assertIn("checkpoint missing", obstructions[0].fallback_reason)

    def test_segmentation_failure_fail_closed_rejects_candidate(self):
        image = np.full((120, 120, 3), 40, dtype=np.uint8)
        detection = Detection("car", 0.9, (10, 10, 90, 90))
        fake_segmenter = FakeSegmenter(error=RuntimeError("checkpoint missing"))
        with patch("app.services.processing.settings.sam_enabled", True), patch(
            "app.services.processing.settings.sam_fail_open", False
        ), patch(
            "app.services.processing.get_segmenter", return_value=fake_segmenter
        ):
            _, obstructions = _evaluate_obstructions(
                self.camera,
                [detection],
                self.camera.exit_zone,
                image,
            )
        self.assertEqual(obstructions[0].spatial_method, "sam_rejected")
        self.assertFalse(obstructions[0].is_blocking)

    async def test_image_analysis_without_polygon_uses_full_frame_sam_workflow(self):
        image = np.full((120, 120, 3), 40, dtype=np.uint8)
        detection = Detection("car", 0.9, (20, 20, 90, 90))
        mask = np.zeros((120, 120), dtype=bool)
        mask[20:90, 20:90] = True
        segmentation = SegmentationResult(
            mask=mask,
            polygon=[(20, 20), (90, 20), (90, 90), (20, 90)],
            area_pixels=int(mask.sum()),
            prompt_box=detection.box,
            score=0.86,
            model_name="sam2.1_hiera_tiny",
            inference_ms=11.5,
        )
        fake_detector = FakeDetector([[detection]])
        fake_segmenter = FakeSegmenter(result=segmentation)
        with self.SessionLocal() as db, patch(
            "app.services.processing.get_detector", return_value=fake_detector
        ), patch(
            "app.services.processing.get_segmenter", return_value=fake_segmenter
        ), patch(
            "app.services.processing.settings.sam_enabled", True
        ), patch(
            "app.services.processing.settings.validate_fire_exit_incidents_with_vision", False
        ), patch(
            "app.services.processing.enrich_and_notify", self.noop_notify
        ), patch(
            "app.services.processing.event_hub.broadcast", self.noop_broadcast
        ):
            result = await analyse_image(db, self.camera, image, [])
        self.assertEqual(result["zone_mode"], "full_frame")
        self.assertEqual(len(fake_segmenter.calls), 1)
        self.assertEqual(result["detections"][0]["spatial_method"], "sam_mask")
        self.assertEqual(len(result["incidents"]), 1)
        incidents = self._incidents()
        self.assertEqual(incidents[0].incident_metadata["zone_mode"], "full_frame")

    async def test_full_frame_image_analysis_uses_detected_class_even_without_class_match(self):
        image = np.full((120, 120, 3), 40, dtype=np.uint8)
        detection = Detection("chair", 0.81, (20, 20, 90, 90))
        mask = np.zeros((120, 120), dtype=bool)
        mask[20:90, 20:90] = True
        segmentation = SegmentationResult(
            mask=mask,
            polygon=[(20, 20), (90, 20), (90, 90), (20, 90)],
            area_pixels=int(mask.sum()),
            prompt_box=detection.box,
            score=0.77,
            model_name="sam2.1_hiera_tiny",
            inference_ms=9.8,
        )
        fake_detector = FakeDetector([[detection]])
        fake_segmenter = FakeSegmenter(result=segmentation)
        with self.SessionLocal() as db:
            camera = db.get(Camera, self.camera.id)
            camera.blocked_classes = ["car", "truck"]
            db.commit()
            with patch(
                "app.services.processing.get_detector", return_value=fake_detector
            ), patch(
                "app.services.processing.get_segmenter", return_value=fake_segmenter
            ), patch(
                "app.services.processing.settings.sam_enabled", True
            ), patch(
                "app.services.processing.settings.validate_fire_exit_incidents_with_vision", False
            ), patch(
                "app.services.processing.enrich_and_notify", self.noop_notify
            ), patch(
                "app.services.processing.event_hub.broadcast", self.noop_broadcast
            ):
                result = await analyse_image(db, camera, image, [])
        self.assertEqual(result["zone_mode"], "full_frame")
        self.assertEqual(result["detections"][0]["label"], "chair")
        self.assertTrue(result["detections"][0]["is_blocking"])
        self.assertEqual(result["detections"][0]["spatial_method"], "sam_mask")
        self.assertEqual(len(fake_segmenter.calls), 1)
        self.assertEqual(len(result["incidents"]), 1)
        incidents = self._incidents()
        self.assertEqual(incidents[0].object_type, "chair")
        self.assertEqual(incidents[0].incident_metadata["zone_mode"], "full_frame")

    async def test_image_pipeline_orders_nemo_yolo_sam_nemotron_then_notification(self):
        image = np.full((120, 120, 3), 40, dtype=np.uint8)
        events = []

        def detect_with_nemo(_image, confidence_threshold=None):
            events.append("yolo_tool")
            return {
                "provider": "nemo-yolo",
                "detections": [
                    {"label": "car", "confidence": 0.91, "box": [10, 10, 20, 20]}
                ],
            }

        def segment_with_nemo(_image, box):
            events.append("sam_tool")
            self.assertEqual(box, (10, 10, 20, 20))
            return {
                "provider": "nemo-sam",
                "image_shape": [120, 120],
                "segmentation": {
                    "polygon": [[10, 10], [20, 10], [20, 20], [10, 20]],
                    "area_pixels": 121,
                    "prompt_box": [10, 10, 20, 20],
                    "prompt_point": [15, 15],
                    "score": 0.9,
                    "model_name": "sam2.1_hiera_tiny (cpu)",
                    "inference_ms": 12.0,
                },
            }

        async def validate_with_nemotron(*_args, **_kwargs):
            events.append("nemotron")
            return {
                "confirmed": True,
                "category": "fire_exit_obstruction",
                "summary": "A car overlaps the exit.",
                "visible_evidence": ["Car is visible in the exit frame"],
                "confidence": 0.93,
            }

        async def notify(_db, incident, _camera):
            events.append("incident_and_telegram")
            return incident

        with self.SessionLocal() as db, patch(
            "app.services.processing.orchestrate_object_detection_sync",
            side_effect=detect_with_nemo,
        ), patch(
            "app.services.processing.orchestrate_segmentation_sync",
            side_effect=segment_with_nemo,
        ), patch(
            "app.services.processing.validate_fire_exit_obstruction",
            side_effect=validate_with_nemotron,
        ), patch(
            "app.services.processing.enrich_and_notify",
            side_effect=notify,
        ), patch(
            "app.services.processing.event_hub.broadcast", self.noop_broadcast
        ), patch(
            "app.services.processing.settings.nemo_agent_enabled", True
        ), patch(
            "app.services.processing.settings.nemo_agent_orchestrate_cv", True
        ), patch(
            "app.services.processing.settings.sam_enabled", True
        ), patch(
            "app.services.processing.settings.validate_fire_exit_incidents_with_vision", True
        ), patch(
            "app.services.processing.settings.minimum_exit_blockage_ratio", 0.05
        ):
            result = await analyse_image(db, self.camera, image, [])

        self.assertEqual(
            events,
            ["yolo_tool", "sam_tool", "nemotron", "incident_and_telegram"],
        )
        self.assertEqual(result["zone_mode"], "full_frame")
        self.assertTrue(result["detections"][0]["is_blocking"])
        self.assertLess(result["detections"][0]["exit_blockage_ratio"], 0.05)
        self.assertEqual(len(result["incidents"]), 1)
        self.assertTrue(result["vision_validations"][0]["accepted"])
        self.assertTrue(result["vision_validations"][0]["confirmed"])

    async def test_image_result_explains_nemotron_rejection(self):
        image = np.full((120, 120, 3), 40, dtype=np.uint8)
        detection = Detection("chair", 0.82, (20, 20, 90, 90))
        mask = np.zeros((120, 120), dtype=bool)
        mask[20:90, 20:90] = True
        segmentation = SegmentationResult(
            mask=mask,
            polygon=[(20, 20), (90, 20), (90, 90), (20, 90)],
            area_pixels=int(mask.sum()),
            prompt_box=detection.box,
            score=0.88,
            model_name="sam2.1_hiera_tiny",
            inference_ms=8.0,
        )
        with self.SessionLocal() as db, patch(
            "app.services.processing.get_detector",
            return_value=FakeDetector([[detection]]),
        ), patch(
            "app.services.processing.get_segmenter",
            return_value=FakeSegmenter(result=segmentation),
        ), patch(
            "app.services.processing.settings.sam_enabled", True
        ), patch(
            "app.services.processing.settings.validate_fire_exit_incidents_with_vision", True
        ), patch(
            "app.services.processing.validate_fire_exit_obstruction",
            AsyncMock(
                return_value={
                    "confirmed": False,
                    "category": "fire_exit_obstruction",
                    "summary": "The chair is not in an exit route.",
                    "visible_evidence": ["No exit context is visible"],
                    "confidence": 0.74,
                    "model": "nemotron-test",
                }
            ),
        ):
            result = await analyse_image(db, self.camera, image, [])

        self.assertEqual(result["incidents"], [])
        self.assertEqual(result["telegram_status"], "not_sent")
        self.assertEqual(len(result["vision_validations"]), 1)
        self.assertFalse(result["vision_validations"][0]["accepted"])
        self.assertEqual(
            result["vision_validations"][0]["summary"],
            "The chair is not in an exit route.",
        )

    async def test_image_preview_token_reuses_cached_yolo_and_sam_stage(self):
        image = np.full((120, 120, 3), 40, dtype=np.uint8)
        detection = Detection("car", 0.9, (20, 20, 90, 90))
        mask = np.zeros((120, 120), dtype=bool)
        mask[20:90, 20:90] = True
        segmentation = SegmentationResult(
            mask=mask,
            polygon=[(20, 20), (90, 20), (90, 90), (20, 90)],
            area_pixels=int(mask.sum()),
            prompt_box=detection.box,
            score=0.86,
            model_name="sam2.1_hiera_tiny",
            inference_ms=11.5,
        )
        fake_detector = FakeDetector([[detection]])
        fake_segmenter = FakeSegmenter(result=segmentation)
        with self.SessionLocal() as db, patch(
            "app.services.processing.get_detector", return_value=fake_detector
        ), patch(
            "app.services.processing.get_segmenter", return_value=fake_segmenter
        ), patch(
            "app.services.processing.settings.sam_enabled", True
        ), patch(
            "app.services.processing.settings.validate_fire_exit_incidents_with_vision", False
        ), patch(
            "app.services.processing.enrich_and_notify", self.noop_notify
        ), patch(
            "app.services.processing.event_hub.broadcast", self.noop_broadcast
        ):
            preview = preview_image_analysis(self.camera, image, [])
            result = await analyse_image(
                db,
                self.camera,
                image,
                [],
                preview_token=preview["preview_token"],
            )
        self.assertEqual(preview["zone_mode"], "full_frame")
        self.assertEqual(result["detections"][0]["spatial_method"], "sam_mask")
        self.assertEqual(len(fake_segmenter.calls), 1)
        self.assertEqual(result["incidents"], [self._incidents()[0].id])


class AgentRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_deterministic_iou_alert_skips_all_llm_generation(self):
        incident = SimpleNamespace(
            id="INC-DIRECT",
            facility="Building A",
            zone="South Exit",
            event_type="fire_exit_obstruction",
            object_type="car",
            confidence=0.94,
            overlap=0.82,
            duration_seconds=0.0,
            first_seen=datetime.now(timezone.utc),
            summary="",
            recommended_action="",
            incident_metadata={
                "vision_validation": {
                    "mode": "deterministic_iou",
                    "iou": 0.70,
                    "threshold": 0.70,
                }
            },
        )
        camera = SimpleNamespace(id="cam-1")
        db = MagicMock()
        llm = AsyncMock()
        fallback = MagicMock(
            return_value=("Car detected at South Exit.", "Notify Facilities Security.")
        )
        sender = AsyncMock(return_value=("sent", "telegram-message"))
        with patch("app.services.agent.search_sops", return_value=[]), patch(
            "app.services.agent.create_grounded_summary", llm
        ), patch(
            "app.services.agent.create_grounded_summary_fallback", fallback
        ), patch(
            "app.services.agent.subscriber_chat_ids", return_value=["operator"]
        ), patch("app.services.agent.send_incident_alert", sender):
            result = await enrich_and_notify(db, incident, camera)
        llm.assert_not_awaited()
        fallback.assert_called_once()
        sender.assert_awaited_once_with(incident, ["operator"])
        self.assertEqual(result.telegram_status, "sent")


class LlmRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_rag_query_is_pinned_to_qwen(self):
        captured = {}

        async def fake_completion(**kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "Use the local SOP."}}]}

        routed = AsyncMock()
        with patch("app.services.llm.settings.llm_enabled", True), patch(
            "app.services.llm.settings.telegram_query_model",
            "Qwen/Qwen2.5-7B-Instruct",
        ), patch(
            "app.services.llm.chat_completions", side_effect=fake_completion
        ), patch("app.services.llm.routed_text_completion", routed):
            answer = await answer_sop_question_direct(
                "What should I do?",
                None,
                "sops/reference.txt",
                "Notify Facilities Security.",
                [],
            )
        self.assertEqual(answer, "Use the local SOP.")
        self.assertEqual(captured["model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(captured["base_url"], settings.llm_base_url)
        routed.assert_not_awaited()


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
                "app.api.analyse_image",
                AsyncMock(
                    return_value={
                        "provider": "yolo",
                        "detections": [],
                        "incidents": [],
                        "annotated_image": "/evidence/preview.jpg",
                        "zone_mode": "full_frame",
                    }
                ),
            ) as analyse_fire_exit, patch(
                "app.api.create_scene_incident", AsyncMock(return_value=None)
            ) as create_scene:
                result = asyncio.run(analyse_uploaded_image(DummyUpload(), "cam-api", "", db))
            self.assertEqual(result["zone_mode"], "full_frame")
            self.assertEqual(result["incidents"], [])
            analyse_fire_exit.assert_awaited()
            create_scene.assert_not_awaited()
        finally:
            db.close()

    def test_image_analysis_uses_full_frame_path_when_exit_zone_payload_is_empty(self):
        from app.api import analyse_uploaded_image

        class DummyUpload:
            filename = "frame.jpg"

        camera = Camera(
            id="cam-api-empty-zone",
            name="Cam",
            facility="Building A",
            zone="Zone A",
            exit_zone=[[0, 0], [100, 0], [100, 100], [0, 100]],
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
                "app.api.analyse_image",
                AsyncMock(
                    return_value={
                        "provider": "yolo",
                        "detections": [],
                        "incidents": [],
                        "annotated_image": "/evidence/preview.jpg",
                        "zone_mode": "full_frame",
                    }
                )
            ) as analyse_fire_exit:
                result = asyncio.run(analyse_uploaded_image(DummyUpload(), "cam-api-empty-zone", "[]", db))
            self.assertEqual(result["zone_mode"], "full_frame")
            analyse_fire_exit.assert_awaited()
        finally:
            db.close()

    def test_image_preview_route_uses_preview_pipeline(self):
        from app.api import preview_uploaded_image_analysis

        class DummyUpload:
            filename = "frame.jpg"

        camera = Camera(
            id="cam-api-preview",
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
                "app.api.preview_image_analysis",
                return_value={
                    "provider": "yolo",
                    "detections": [],
                    "incidents": [],
                    "annotated_image": "/evidence/preview.jpg",
                    "zone_mode": "full_frame",
                    "preview_token": "preview-token-1",
                    "blocking_candidates": 1,
                    "will_validate_with_vision": True,
                    "next_step": "Nemotron validation",
                    "telegram_status": "not_sent",
                },
            ) as preview_image:
                result = asyncio.run(preview_uploaded_image_analysis(DummyUpload(), "cam-api-preview", "", db))
            self.assertEqual(result["preview_token"], "preview-token-1")
            preview_image.assert_called_once()
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

    def test_fire_exit_validation_parser_preserves_vehicle_identifier(self):
        result = _parse_fire_exit_validation(
            '{"confirmed":true,"category":"fire_exit_obstruction","summary":"Vehicle blocks the zone.",'
            '"visible_evidence":["Vehicle overlaps the marked restricted area"],"confidence":0.91,'
            '"vehicle_identifier":"ABC1234","vehicle_identifier_type":"license_plate",'
            '"vehicle_identifier_confidence":0.88}'
        )
        self.assertEqual(result["vehicle_identifier"], "ABC1234")
        self.assertEqual(result["vehicle_identifier_type"], "license_plate")
        self.assertAlmostEqual(result["vehicle_identifier_confidence"], 0.88)

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

    def test_fire_exit_validation_prompt_includes_primary_yolo_label(self):
        with patch(
            "app.services.scene_reasoning.settings.vision_enabled", True
        ), patch(
            "app.services.scene_reasoning.settings.nemo_agent_orchestrate_vision", False
        ), patch(
            "app.services.scene_reasoning._call_vision_model",
            AsyncMock(
                return_value='{"confirmed":true,"category":"fire_exit_obstruction","summary":"Chair blocks the route.","visible_evidence":["Chair blocks the route"],"confidence":0.84}'
            ),
        ) as call:
            result = asyncio.run(validate_fire_exit_obstruction(b"image-bytes", object_label="chair"))
        self.assertTrue(result["confirmed"])
        self.assertIn("Primary YOLO object: chair.", call.await_args.args[0])

    def test_vision_call_retries_a_length_truncated_json_response(self):
        truncated = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": '{"confirmed":true,"summary":"cut'},
                }
            ]
        }
        complete = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"confirmed":true,"summary":"ok"}'},
                }
            ]
        }
        with patch(
            "app.services.scene_reasoning.chat_completions",
            AsyncMock(side_effect=[truncated, complete]),
        ) as completion:
            content = asyncio.run(
                _call_vision_model("prompt", b"image", max_response_tokens=120)
            )
        self.assertEqual(content, '{"confirmed":true,"summary":"ok"}')
        self.assertEqual(completion.await_count, 2)
        self.assertEqual(completion.await_args_list[0].kwargs["max_tokens"], 120)
        self.assertEqual(completion.await_args_list[1].kwargs["max_tokens"], 512)

    def test_fire_exit_validation_prompt_is_compact(self):
        prompt = _build_fire_exit_validation_prompt("chair")
        self.assertIn("Primary YOLO object: chair.", prompt)
        self.assertIn("whole image is that zone", prompt)
        self.assertLess(len(prompt), 700)

    def test_prepare_vision_image_bytes_downscales_large_crop(self):
        image = np.zeros((1600, 1200, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        with patch("app.services.scene_reasoning.settings.vision_validation_image_max_dim", 640), patch(
            "app.services.scene_reasoning.settings.vision_validation_jpeg_quality", 75
        ):
            optimized = _prepare_vision_image_bytes(encoded.tobytes())
        restored = cv2.imdecode(np.frombuffer(optimized, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(restored)
        self.assertLessEqual(max(restored.shape[:2]), 640)

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

    def test_annotation_with_sam_overlay_is_valid(self):
        image = synthetic_demo_frame()
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[360:620, 360:790] = True
        obstruction = Obstruction(
            Detection("vehicle", 0.9, (350, 350, 780, 625)),
            0.9,
            0.9,
            0.35,
            True,
            5.0,
            segmentation=SegmentationResult(
                mask=mask,
                polygon=[(360, 360), (790, 360), (790, 620), (360, 620)],
                area_pixels=int(mask.sum()),
                prompt_box=(350, 350, 780, 625),
                score=0.91,
                model_name="sam2.1_hiera_tiny",
                inference_ms=12.8,
            ),
            spatial_method="sam_mask",
            mask_zone_iou=0.44,
        )
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "annotated-sam.jpg"
            annotate(
                image,
                [[250, 100], [850, 100], [850, 680], [250, 680]],
                [obstruction],
                destination,
            )
            result = cv2.imread(str(destination))
            self.assertIsNotNone(result)
            self.assertFalse(np.array_equal(image, result))

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

    def test_disabled_text_model_uses_deterministic_summary(self):
        with patch("app.services.llm.settings.llm_enabled", False):
            summary, action = asyncio.run(
                create_grounded_summary(
                    {
                        "object_type": "car",
                        "zone": "South Access Zone",
                        "facility": "Building A",
                        "confidence": 0.91,
                        "overlap": 0.42,
                    },
                    search_sops("fire_exit_obstruction", "Building A", "car"),
                )
            )
        self.assertIn("Car detected", summary)
        self.assertTrue(action)

    def test_llm_runtime_status_reports_model_availability(self):
        with patch("app.services.llm.settings.llm_enabled", True), patch(
            "app.services.llm.settings.llm_model", "Qwen/Qwen2.5-7B-Instruct"
        ), patch(
            "app.services.llm.settings.llm_base_url", "http://127.0.0.1:8001/v1"
        ), patch(
            "app.services.llm.settings.llm_api_key", ""
        ), patch(
            "app.services.llm.settings.llm_timeout_seconds", 30.0
        ), patch(
            "app.services.llm.list_models",
            AsyncMock(return_value=["Qwen/Qwen2.5-7B-Instruct"]),
        ):
            status = asyncio.run(llm_runtime_status())

        self.assertTrue(status["enabled"])
        self.assertTrue(status["reachable"])
        self.assertTrue(status["model_available"])
        self.assertEqual(status["detail"], "ready")

    def test_incident_schema_exposes_segmentation_fields(self):
        incident = Incident(
            id="INC-SAM-001",
            camera_id="cam-1",
            facility="Building A",
            zone="South Access Zone",
            event_type="exit_blocked",
            object_type="car",
            confidence=0.91,
            overlap=0.4,
            duration_seconds=12.0,
            status="open",
            severity="high",
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
            incident_metadata={
                "spatial_method": "sam_mask",
                "object_intrusion_ratio": 0.42,
                "exit_blockage_ratio": 0.18,
                "mask_zone_iou": 0.11,
                "sam_polygon": [[1, 2], [3, 4], [5, 6]],
                "sam_model": "sam2.1_hiera_tiny",
                "sam_score": 0.83,
                "sam_inference_ms": 12.5,
            },
            summary="Car blocks the exit.",
            recommended_action="Notify Facilities Security.",
            sop_title="Fire Exit Obstruction",
            sop_sources=[],
            telegram_status="sent",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        from app.schemas import IncidentOut

        payload = IncidentOut.model_validate(incident)
        self.assertEqual(payload.spatial_method, "sam_mask")
        self.assertEqual(payload.sam_model, "sam2.1_hiera_tiny")
        self.assertEqual(payload.sam_polygon, [[1, 2], [3, 4], [5, 6]])


if __name__ == "__main__":
    unittest.main()
