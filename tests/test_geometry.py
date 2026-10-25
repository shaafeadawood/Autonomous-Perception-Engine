from __future__ import annotations

import pytest

from autonomous_perception_engine.types import BBox, Detection


def test_bbox_round_trip_and_iou() -> None:
    first = BBox(10, 20, 30, 60)
    rebuilt = BBox.from_cxcywh(*first.cxcywh)
    assert rebuilt == first
    assert first.iou(first) == pytest.approx(1.0)
    assert first.iou(BBox(100, 100, 120, 120)) == 0.0


def test_bbox_clip_stays_inside_frame() -> None:
    clipped = BBox(-10, -5, 120, 90).clip(100, 80)
    assert clipped == BBox(0, 0, 99, 79)


def test_invalid_bbox_and_confidence_are_rejected() -> None:
    with pytest.raises(ValueError, match="ordering"):
        BBox(5, 5, 4, 6)
    with pytest.raises(ValueError, match="confidence"):
        Detection(BBox(0, 0, 1, 1), 0, "object", 1.1)
