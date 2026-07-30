from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    track_id: int | None = None


@dataclass
class Obstruction:
    detection: Detection
    overlap: float
    object_intrusion_ratio: float
    exit_blockage_ratio: float
    is_blocking: bool
    blocked_duration_seconds: float = 0.0
