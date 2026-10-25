"""Track state, lifecycle, and observation history."""

from __future__ import annotations

from collections import Counter, deque
from enum import StrEnum

from ..config import TrackerConfig
from ..types import BBox, Detection, TrackSnapshot
from .kalman import BoundingBoxKalmanFilter


class TrackState(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


class Track:
    """One physical-object hypothesis maintained across frames."""

    def __init__(self, track_id: int, detection: Detection, config: TrackerConfig) -> None:
        self.track_id = track_id
        self.config = config
        self.kalman = BoundingBoxKalmanFilter(detection.bbox, config.kalman)
        self.bbox = detection.bbox
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.state = TrackState.CONFIRMED if config.min_hits <= 1 else TrackState.TENTATIVE
        self.age = 1
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self._class_votes: deque[tuple[int, str]] = deque(
            [(detection.class_id, detection.class_name)], maxlen=config.class_vote_window
        )
        self.history: deque[tuple[float, float]] = deque(
            [detection.bbox.center], maxlen=config.history_size
        )

    def predict(self, dt: float) -> BBox:
        self.bbox = self.kalman.predict(dt)
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.bbox.center)
        return self.bbox

    def update(self, detection: Detection) -> None:
        self.bbox = self.kalman.update(detection.bbox)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        alpha = self.config.confidence_smoothing
        self.confidence = alpha * self.confidence + (1.0 - alpha) * detection.confidence
        self._class_votes.append((detection.class_id, detection.class_name))
        (self.class_id, self.class_name), _ = Counter(self._class_votes).most_common(1)[0]
        if self.history:
            self.history[-1] = self.bbox.center
        else:
            self.history.append(self.bbox.center)
        if self.hits >= self.config.min_hits:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        self.hit_streak = 0
        if self.state == TrackState.TENTATIVE:
            if self.time_since_update > self.config.max_tentative_age:
                self.state = TrackState.DELETED
            return
        if self.time_since_update > self.config.max_age:
            self.state = TrackState.DELETED
        else:
            self.state = TrackState.LOST

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def snapshot(self) -> TrackSnapshot:
        return TrackSnapshot(
            track_id=self.track_id,
            bbox=self.bbox,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            state=self.state.value,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            history=tuple(self.history),
        )
