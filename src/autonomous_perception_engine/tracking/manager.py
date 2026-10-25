"""Two-stage multi-object tracker orchestration."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import TrackerConfig
from ..types import Detection, TrackSnapshot
from .association import associate
from .track import Track, TrackState


class MultiObjectTracker:
    """Manage track prediction, association, confirmation, loss, and deletion."""

    def __init__(self, config: TrackerConfig, high_confidence_threshold: float) -> None:
        self.config = config
        self.high_confidence_threshold = high_confidence_threshold
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, detections: Sequence[Detection], dt: float) -> tuple[TrackSnapshot, ...]:
        if dt <= 0.0:
            raise ValueError("Tracker time step must be positive")

        active_tracks = [track for track in self.tracks if not track.is_deleted]
        for track in active_tracks:
            track.predict(dt)

        high = [
            detection
            for detection in detections
            if detection.confidence >= self.high_confidence_threshold
        ]
        low = [
            detection
            for detection in detections
            if detection.confidence < self.high_confidence_threshold
        ]

        first = associate(
            active_tracks,
            high,
            self.config.association,
            self.config.association.first_stage_max_cost,
        )
        matched_track_indices: set[int] = set()
        for track_index, detection_index in first.matches:
            active_tracks[track_index].update(high[detection_index])
            matched_track_indices.add(track_index)

        second_track_indices = [
            index
            for index in first.unmatched_tracks
            if active_tracks[index].state != TrackState.TENTATIVE
        ]
        second_tracks = [active_tracks[index] for index in second_track_indices]
        second = associate(
            second_tracks,
            low,
            self.config.association,
            self.config.association.second_stage_max_cost,
        )
        for relative_track_index, detection_index in second.matches:
            absolute_track_index = second_track_indices[relative_track_index]
            active_tracks[absolute_track_index].update(low[detection_index])
            matched_track_indices.add(absolute_track_index)

        for index, track in enumerate(active_tracks):
            if index not in matched_track_indices:
                track.mark_missed()

        for detection_index in first.unmatched_detections:
            self.tracks.append(Track(self.next_id, high[detection_index], self.config))
            self.next_id += 1

        self.tracks = [track for track in self.tracks if not track.is_deleted]
        return self.visible_tracks()

    def visible_tracks(self) -> tuple[TrackSnapshot, ...]:
        return tuple(
            track.snapshot()
            for track in self.tracks
            if track.state in (TrackState.CONFIRMED, TrackState.LOST)
            and track.time_since_update <= self.config.render_max_age
        )

    def all_tracks(self) -> tuple[TrackSnapshot, ...]:
        return tuple(track.snapshot() for track in self.tracks)
