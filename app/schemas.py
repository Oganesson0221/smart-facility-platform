from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CameraIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    facility: str = Field(min_length=1, max_length=120)
    zone: str = Field(min_length=1, max_length=120)
    rtsp_url: str | None = None
    exit_zone: list[list[float]] = []
    blocked_classes: list[str] = []
    confidence_threshold: float = Field(0.35, ge=0, le=1)
    minimum_overlap: float = Field(0.25, ge=0, le=1)
    persistence_seconds: float = Field(5, ge=0)
    alert_cooldown_seconds: int = Field(300, ge=0)
    enabled: bool = True


class CameraOut(CameraIn):
    model_config = ConfigDict(from_attributes=True)
    id: str
    created_at: datetime


class IncidentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    camera_id: str
    facility: str
    zone: str
    event_type: str
    object_type: str
    confidence: float
    overlap: float
    duration_seconds: float
    status: str
    severity: str
    first_seen: datetime
    last_seen: datetime
    evidence_image: str | None
    evidence_clip: str | None
    summary: str
    recommended_action: str
    sop_title: str
    sop_sources: list[Any]
    telegram_status: str
    acknowledged_by: str | None
    acknowledged_at: datetime | None
    created_at: datetime
    updated_at: datetime


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    media_type: str
    filename: str
    camera_id: str
    status: str
    progress: float
    message: str
    incidents: list[str]
    created_at: datetime
    updated_at: datetime
