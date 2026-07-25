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
