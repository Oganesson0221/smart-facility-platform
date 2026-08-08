from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[5]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import register_function

from app.services.nemo_tools import detect_image_objects_payload
from app.services.nemo_tools import inspect_scene_image_payload
from app.services.nemo_tools import lookup_safety_sop_payload
from app.services.nemo_tools import segment_image_box_payload
from app.services.nemo_tools import validate_fire_exit_image_payload


class FacilitySOPConfig(FunctionBaseConfig, name="smart_facility_sop"):
    pass


class FacilityObjectDetectorConfig(FunctionBaseConfig, name="smart_facility_object_detector"):
    default_confidence_threshold: float | None = None


class FacilitySegmenterConfig(FunctionBaseConfig, name="smart_facility_segmenter"):
    pass


class FacilitySceneAssessorConfig(FunctionBaseConfig, name="smart_facility_scene_assessor"):
    pass


class FacilityFireExitValidatorConfig(
    FunctionBaseConfig,
    name="smart_facility_fire_exit_validator",
):
    pass


@register_function(config_type=FacilitySOPConfig)
async def build_facility_sop(_config: FacilitySOPConfig, _builder: Builder):
    async def lookup_safety_sop(
        event_type: str,
        object_type: str,
        facility: str = "",
    ) -> str:
        """Retrieve local safety procedures relevant to a facility incident."""
        result = lookup_safety_sop_payload(event_type, object_type, facility)
        return json.dumps(result, ensure_ascii=True)

    yield FunctionInfo.from_fn(
        lookup_safety_sop,
        description=(
            "Retrieve local SOP content and metadata. Call this before recommending "
            "actions for a confirmed facility incident."
        ),
    )


@register_function(config_type=FacilityObjectDetectorConfig)
async def build_facility_object_detector(
    config: FacilityObjectDetectorConfig,
    _builder: Builder,
):
    async def detect_image_objects(
        image_path: str,
        confidence_threshold: float | None = None,
    ) -> str:
        """Run the configured local detector, such as YOLO, on one image."""
        result = detect_image_objects_payload(
            image_path,
            confidence_threshold=(
                confidence_threshold
                if confidence_threshold is not None
                else config.default_confidence_threshold
            ),
        )
        return json.dumps(result, ensure_ascii=True)

    yield FunctionInfo.from_fn(
        detect_image_objects,
        description=(
            "Run local object detection against an image and return grounded boxes "
            "with labels and confidences."
        ),
    )


@register_function(config_type=FacilitySegmenterConfig)
async def build_facility_segmenter(_config: FacilitySegmenterConfig, _builder: Builder):
    async def segment_image_box(image_path: str, box_json: str) -> str:
        """Run local SAM segmentation for one detected object box."""
        result = segment_image_box_payload(image_path, box_json)
        return json.dumps(result, ensure_ascii=True)

    yield FunctionInfo.from_fn(
        segment_image_box,
        description=(
            "Refine one detected bounding box with Segment Anything and return the "
            "polygon and mask metadata."
        ),
    )


@register_function(config_type=FacilitySceneAssessorConfig)
async def build_facility_scene_assessor(
    _config: FacilitySceneAssessorConfig,
    _builder: Builder,
):
    async def inspect_scene_image(
        image_path: str,
        yolo_detections_json: str = "[]",
    ) -> str:
        """Assess a facility scene image with grounded detections and the local vision model."""
        result = await inspect_scene_image_payload(image_path, yolo_detections_json)
        return json.dumps(result, ensure_ascii=True)

    yield FunctionInfo.from_fn(
        inspect_scene_image,
        description=(
            "Perform grounded scene reasoning for a facility image. When detections "
            "are missing, it may run the local detector first."
        ),
    )


@register_function(config_type=FacilityFireExitValidatorConfig)
async def build_fire_exit_validator(
    _config: FacilityFireExitValidatorConfig,
    _builder: Builder,
):
    async def validate_fire_exit_image(
        image_path: str,
        object_label: str = "",
    ) -> str:
        """Validate whether a cropped candidate image shows a true fire-exit obstruction."""
        result = await validate_fire_exit_image_payload(image_path, object_label)
        return json.dumps(result, ensure_ascii=True)

    yield FunctionInfo.from_fn(
        validate_fire_exit_image,
        description=(
            "Use the local multimodal model to confirm whether the visible candidate "
            "is truly blocking a fire exit or protected clearance area."
        ),
    )
