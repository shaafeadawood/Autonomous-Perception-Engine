"""Unique-track, active-track, and directional line-crossing statistics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ..config import CountingConfig
from ..types import TrackSnapshot


@dataclass(frozen=True, slots=True)
class AnalyticsSnapshot:
    active_tracks: int
    active_by_class: dict[str, int]
    unique_tracks: int
    unique_by_class: dict[str, int]
    crossings_a_to_b: int
    crossings_b_to_a: int


class ObjectCounter:
    def __init__(self, config: CountingConfig) -> None:
        self.config = config
        self._seen_ids: dict[int, str] = {}
        self._previous_sides: dict[int, float] = {}
        self._crossings_a_to_b = 0
        self._crossings_b_to_a = 0

    def update(
        self,
        tracks: Sequence[TrackSnapshot],
        frame_width: int,
        frame_height: int,
    ) -> AnalyticsSnapshot:
        if not self.config.enabled:
            return AnalyticsSnapshot(0, {}, 0, {}, 0, 0)
        current = [track for track in tracks if track.time_since_update == 0]
        for track in current:
            self._seen_ids.setdefault(track.track_id, track.class_name)

        if self.config.line_enabled:
            start, end = self.line_points(frame_width, frame_height)
            for track in current:
                side = self._side(track.bbox.center, start, end)
                previous = self._previous_sides.get(track.track_id)
                if previous is not None and side != 0.0 and previous != 0.0:
                    if previous < 0.0 < side:
                        self._crossings_a_to_b += 1
                    elif previous > 0.0 > side:
                        self._crossings_b_to_a += 1
                if side != 0.0:
                    self._previous_sides[track.track_id] = side

        active_by_class = Counter(track.class_name for track in current)
        unique_by_class = Counter(self._seen_ids.values())
        return AnalyticsSnapshot(
            active_tracks=len(current),
            active_by_class=dict(sorted(active_by_class.items())),
            unique_tracks=len(self._seen_ids),
            unique_by_class=dict(sorted(unique_by_class.items())),
            crossings_a_to_b=self._crossings_a_to_b,
            crossings_b_to_a=self._crossings_b_to_a,
        )

    def line_points(
        self, frame_width: int, frame_height: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        start = self.config.line_start
        end = self.config.line_end
        if self.config.normalized_coordinates:
            return (
                (int(start[0] * frame_width), int(start[1] * frame_height)),
                (int(end[0] * frame_width), int(end[1] * frame_height)),
            )
        return ((int(start[0]), int(start[1])), (int(end[0]), int(end[1])))

    @staticmethod
    def _side(
        point: tuple[float, float],
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> float:
        return (end[0] - start[0]) * (point[1] - start[1]) - (
            end[1] - start[1]
        ) * (point[0] - start[0])
