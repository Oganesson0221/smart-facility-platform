from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Smart Facility Platform"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    public_base_url: str = "http://127.0.0.1:8000"
    database_url: str = "sqlite:///./data/exitwatch.db"
    max_upload_mb: int = 250

    detector_provider: str = "demo"
    device: str = "auto"
    cv_shared_gpu_safe_mode: bool = True
    yolo_model_path: str = "yolo11n.pt"
    yolo_confidence_threshold: float = 0.35
    yolo_image_size: int = 640
    yolo_device: str = "auto"
    sam_enabled: bool = False
    sam_provider: str = "sam2"
    sam_model_size: str = "tiny"
    sam_checkpoint_path: str = ""
    sam_device: str = "auto"
    sam_use_fp16: bool = True
    sam_min_yolo_confidence: float = 0.35
    sam_only_for_zone_candidates: bool = True
    sam_boundary_margin_pixels: int = 20
    sam_mask_simplification_epsilon: float = 2.0
    sam_prompt_box_expand_ratio: float = 0.03
    sam_fail_open: bool = True
    grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny"
    detection_prompt: str = (
        "vehicle . car . truck . motorcycle . chair . trolley . pallet . "
        "cardboard box . large object . person ."
    )
    detection_threshold: float = 0.35
    blocked_classes: str = ""
    fire_exit_obstruction_classes: str = "car,truck,bus,motorcycle,bicycle,chair"
    include_person_as_obstruction: bool = False
    person_minimum_duration_seconds: float = 15.0
    minimum_object_intrusion_ratio: float = 0.25
    minimum_overlap: float | None = None
    minimum_exit_blockage_ratio: float = 0.05
    minimum_duration_seconds: float = 5.0
    clear_after_seconds: float = 3.0
    video_sample_fps: float = 3.0
    alert_cooldown_seconds: int = 300

    llm_enabled: bool = False
    llm_base_url: str = "http://127.0.0.1:8001/v1"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_api_key: str = ""
    llm_timeout_seconds: float = 45.0

    nemo_agent_enabled: bool = True
    nemo_agent_base_url: str = "http://127.0.0.1:8010/v1"
    nemo_agent_model: str = "smart-facility-agent"
    nemo_agent_api_key: str = ""
    nemo_agent_timeout_seconds: float = 60.0
    nemo_agent_required: bool = False
    nemo_agent_orchestrate_cv: bool = True
    nemo_agent_orchestrate_vision: bool = True

    vision_enabled: bool = True
    vision_base_url: str = "http://127.0.0.1:8002/v1"
    vision_model: str = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
    vision_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("VISION_API_KEY", "LLM_API_KEY"),
    )
    vision_keep_alive: str = "30m"
    vision_enable_thinking: bool = False
    vision_timeout_seconds: float = 180.0
    vision_validation_timeout_seconds: float = 60.0
    vision_validation_image_max_dim: int = 768
    vision_validation_jpeg_quality: int = 80
    vision_validation_max_response_tokens: int = 300
    validate_fire_exit_incidents_with_vision: bool = True
    vision_validation_fail_closed: bool = False

    telegram_bot_token: str = ""
    telegram_alert_chat_id: str = Field(
        default="",
        validation_alias=AliasChoices("TELEGRAM_ALERT_CHAT_ID"),
    )
    user_id: str = Field(
        default="",
        validation_alias=AliasChoices("USER_ID", "TELEGRAM_USER_ID"),
    )
    telegram_webhook_secret: str = "change-me-before-enabling-webhooks"
    telegram_bot_username: str = "SmartFacilityAssistant_bot"
    telegram_polling_enabled: bool = True
    telegram_poll_timeout_seconds: int = 25

    @property
    def blocked_class_set(self) -> set[str]:
        source = self.blocked_classes or self.fire_exit_obstruction_classes
        return {value.strip().lower() for value in source.split(",") if value.strip()}

    @property
    def default_minimum_overlap(self) -> float:
        return (
            self.minimum_overlap
            if self.minimum_overlap is not None
            else self.minimum_object_intrusion_ratio
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

for folder in ("data", "uploads", "evidence"):
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
