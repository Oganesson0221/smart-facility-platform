from dataclasses import dataclass

from app.services.cv.types import Detection


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1)


@dataclass
class Track:
    id: int
    box: tuple[int, int, int, int]
    label: str
    missed: int = 0


class IoUTracker:
    """Small dependency-free tracker; replaceable with ByteTrack/DeepStream."""

    def __init__(self, threshold: float = 0.25, max_missed: int = 8):
        self.threshold = threshold
        self.max_missed = max_missed
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched_tracks = set(self.tracks)
        for detection in detections:
            candidates = [
                (track_id, iou(detection.box, track.box))
                for track_id, track in self.tracks.items()
                if track.label == detection.label and track_id in unmatched_tracks
            ]
            track_id, score = max(candidates, key=lambda pair: pair[1], default=(None, 0))
            if track_id is not None and score >= self.threshold:
                detection.track_id = track_id
                self.tracks[track_id].box = detection.box
                self.tracks[track_id].missed = 0
                unmatched_tracks.remove(track_id)
            else:
                detection.track_id = self.next_id
                self.tracks[self.next_id] = Track(
                    self.next_id, detection.box, detection.label
                )
                self.next_id += 1
        for track_id in unmatched_tracks:
            self.tracks[track_id].missed += 1
            if self.tracks[track_id].missed > self.max_missed:
                del self.tracks[track_id]
        return detections
