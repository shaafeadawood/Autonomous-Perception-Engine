from __future__ import annotations

from autonomous_perception_engine.config import TrackerConfig
from autonomous_perception_engine.tracking.manager import MultiObjectTracker


def test_track_requires_confirmation_and_keeps_id(detection_factory) -> None:
    config = TrackerConfig(min_hits=2, max_age=2)
    tracker = MultiObjectTracker(config, high_confidence_threshold=0.5)
    assert tracker.update([detection_factory(x=10)], 1.0) == ()
    visible = tracker.update([detection_factory(x=12)], 1.0)
    assert len(visible) == 1
    assert visible[0].track_id == 1
    assert visible[0].state == "confirmed"

    low_confidence = tracker.update([detection_factory(x=14, confidence=0.3)], 1.0)
    assert len(low_confidence) == 1
    assert low_confidence[0].track_id == 1
    assert tracker.next_id == 2


def test_low_confidence_detection_cannot_start_track(detection_factory) -> None:
    tracker = MultiObjectTracker(TrackerConfig(min_hits=1), high_confidence_threshold=0.5)
    assert tracker.update([detection_factory(confidence=0.2)], 1.0) == ()
    assert tracker.all_tracks() == ()


def test_confirmed_track_expires_after_max_age(detection_factory) -> None:
    tracker = MultiObjectTracker(
        TrackerConfig(min_hits=1, max_age=2, render_max_age=2),
        high_confidence_threshold=0.5,
    )
    tracker.update([detection_factory()], 1.0)
    first_miss = tracker.update([], 1.0)
    second_miss = tracker.update([], 1.0)
    third_miss = tracker.update([], 1.0)
    assert first_miss[0].state == "lost"
    assert second_miss[0].track_id == 1
    assert third_miss == ()
    assert tracker.all_tracks() == ()
