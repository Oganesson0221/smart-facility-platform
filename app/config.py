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
    yolo_model_path: str = "yolo11n.pt"
    yolo_confidence_threshold: float = 0.35
    yolo_image_size: int = 640
    yolo_device: str = "auto"
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
    llm_model: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    llm_api_key: str = ""
    llm_timeout_seconds: float = 45.0

    nemo_agent_enabled: bool = True
    nemo_agent_base_url: str = "http://127.0.0.1:8010/v1"
    nemo_agent_model: str = "smart-facility-agent"
    nemo_agent_api_key: str = ""
    nemo_agent_timeout_seconds: float = 60.0
    nemo_agent_required: bool = False

    vision_enabled: bool = True
    vision_base_url: str = "http://127.0.0.1:11434"
    vision_model: str = "gemma3:27b"
    vision_timeout_seconds: float = 180.0
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
