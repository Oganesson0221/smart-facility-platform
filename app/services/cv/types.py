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
    is_blocking: bool
