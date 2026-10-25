from __future__ import annotations

from autonomous_perception_engine.config import AssociationConfig, TrackerConfig
from autonomous_perception_engine.tracking.association import associate
from autonomous_perception_engine.tracking.track import Track


def test_hungarian_assignment_finds_one_to_one_matches(detection_factory) -> None:
    config = TrackerConfig(min_hits=1)
    tracks = [
        Track(1, detection_factory(x=0), config),
        Track(2, detection_factory(x=100), config),
    ]
    detections = [detection_factory(x=102), detection_factory(x=2)]
    result = associate(tracks, detections, AssociationConfig(), max_cost=1.0)
    assert set(result.matches) == {(0, 1), (1, 0)}
    assert result.unmatched_tracks == ()
    assert result.unmatched_detections == ()


def test_class_gate_blocks_geometrically_identical_pair(detection_factory) -> None:
    tracker_config = TrackerConfig(min_hits=1)
    track = Track(1, detection_factory(class_id=2, class_name="car"), tracker_config)
    person = detection_factory(class_id=0, class_name="person")
    result = associate([track], [person], AssociationConfig(class_aware=True), max_cost=1.0)
    assert result.matches == ()
    assert result.unmatched_tracks == (0,)
    assert result.unmatched_detections == (0,)


def test_pre_assignment_gating_rejects_distant_detection(detection_factory) -> None:
    tracker_config = TrackerConfig(min_hits=1)
    track = Track(1, detection_factory(x=0, y=0), tracker_config)
    distant = detection_factory(x=10_000, y=10_000)
    result = associate([track], [distant], AssociationConfig(), max_cost=1.0)
    assert not result.matches
