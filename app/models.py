from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(120))
    facility: Mapped[str] = mapped_column(String(120))
    zone: Mapped[str] = mapped_column(String(120))
    rtsp_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    exit_zone: Mapped[list] = mapped_column(JSON, default=list)
    blocked_classes: Mapped[list] = mapped_column(JSON, default=list)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.35)
    minimum_overlap: Mapped[float] = mapped_column(Float, default=0.25)
    persistence_seconds: Mapped[float] = mapped_column(Float, default=5.0)
    alert_cooldown_seconds: Mapped[int] = mapped_column(Integer, default=300)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String, index=True)
    facility: Mapped[str] = mapped_column(String(120))
    zone: Mapped[str] = mapped_column(String(120))
    event_type: Mapped[str] = mapped_column(String(80), default="exit_blocked")
    object_type: Mapped[str] = mapped_column(String(80))
    confidence: Mapped[float] = mapped_column(Float)
    overlap: Mapped[float] = mapped_column(Float)
    duration_seconds: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(40), default="open", index=True)
    severity: Mapped[str] = mapped_column(String(40), default="high")
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    evidence_image: Mapped[str | None] = mapped_column(String(500), nullable=True)
    evidence_clip: Mapped[str | None] = mapped_column(String(500), nullable=True)
    incident_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[str] = mapped_column(Text, default="")
    recommended_action: Mapped[str] = mapped_column(Text, default="")
    sop_title: Mapped[str] = mapped_column(String(200), default="")
    sop_sources: Mapped[list] = mapped_column(JSON, default=list)
    telegram_status: Mapped[str] = mapped_column(String(80), default="not_configured")
    telegram_message_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    @property
    def spatial_method(self) -> str | None:
        return (self.incident_metadata or {}).get("spatial_method")

    @property
    def object_intrusion_ratio(self) -> float | None:
        value = (self.incident_metadata or {}).get("object_intrusion_ratio")
        return float(value) if value is not None else None

    @property
    def exit_blockage_ratio(self) -> float | None:
        value = (self.incident_metadata or {}).get("exit_blockage_ratio")
        return float(value) if value is not None else None

    @property
    def mask_zone_iou(self) -> float | None:
        value = (self.incident_metadata or {}).get("mask_zone_iou")
        return float(value) if value is not None else None

    @property
    def sam_polygon(self) -> list | None:
        return (self.incident_metadata or {}).get("sam_polygon")

    @property
    def sam_model(self) -> str | None:
        return (self.incident_metadata or {}).get("sam_model")

    @property
    def sam_score(self) -> float | None:
        value = (self.incident_metadata or {}).get("sam_score")
        return float(value) if value is not None else None

    @property
    def sam_inference_ms(self) -> float | None:
        value = (self.incident_metadata or {}).get("sam_inference_ms")
        return float(value) if value is not None else None

    @property
    def vehicle_identifier(self) -> str | None:
        value = (self.incident_metadata or {}).get("vehicle_identifier")
        text = str(value).strip() if value is not None else ""
        return text or None

    @property
    def vehicle_identifier_type(self) -> str | None:
        value = (self.incident_metadata or {}).get("vehicle_identifier_type")
        text = str(value).strip() if value is not None else ""
        return text or None

    @property
    def vehicle_identifier_confidence(self) -> float | None:
        value = (self.incident_metadata or {}).get("vehicle_identifier_confidence")
        return float(value) if value is not None else None

    @property
    def segmentation(self) -> dict | None:
        metadata = self.incident_metadata or {}
        if not metadata.get("sam_polygon"):
            return None
        return {
            "sam_polygon": metadata.get("sam_polygon"),
            "sam_model": metadata.get("sam_model"),
            "sam_score": metadata.get("sam_score"),
            "sam_inference_ms": metadata.get("sam_inference_ms"),
            "mask_area_pixels": metadata.get("mask_area_pixels"),
            "spatial_method": metadata.get("spatial_method"),
        }


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    media_type: Mapped[str] = mapped_column(String(20))
    filename: Mapped[str] = mapped_column(String(300))
    camera_id: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String(30), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0)
    message: Mapped[str] = mapped_column(Text, default="")
    incidents: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class TelegramSubscriber(Base):
    __tablename__ = "telegram_subscribers"

    chat_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    username: Mapped[str | None] = mapped_column(String(120), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
