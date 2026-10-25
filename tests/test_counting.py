from __future__ import annotations

from autonomous_perception_engine.analytics.counting import ObjectCounter
from autonomous_perception_engine.config import CountingConfig
from autonomous_perception_engine.types import BBox, TrackSnapshot


def snapshot(track_id: int, center_y: float) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=track_id,
        bbox=BBox.from_cxcywh(50, center_y, 20, 20),
        class_id=2,
        class_name="car",
        confidence=0.9,
        state="confirmed",
        age=3,
        hits=3,
        time_since_update=0,
        history=((50, center_y),),
    )


def test_unique_count_does_not_count_same_track_twice() -> None:
    counter = ObjectCounter(CountingConfig())
    first = counter.update([snapshot(7, 20)], 100, 100)
    second = counter.update([snapshot(7, 30)], 100, 100)
    assert first.unique_tracks == 1
    assert second.unique_tracks == 1
    assert second.active_by_class == {"car": 1}


def test_directional_line_crossing() -> None:
    config = CountingConfig(line_enabled=True, line_start=[0.0, 0.5], line_end=[1.0, 0.5])
    counter = ObjectCounter(config)
    counter.update([snapshot(1, 40)], 100, 100)
    result = counter.update([snapshot(1, 60)], 100, 100)
    assert result.crossings_a_to_b == 1
    assert result.crossings_b_to_a == 0


def test_disabled_counting_reports_zero() -> None:
    counter = ObjectCounter(CountingConfig(enabled=False))
    result = counter.update([snapshot(1, 40)], 100, 100)
    assert result.unique_tracks == 0
    assert result.active_tracks == 0
