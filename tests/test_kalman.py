from __future__ import annotations

import numpy as np

from autonomous_perception_engine.config import KalmanConfig
from autonomous_perception_engine.tracking.kalman import BoundingBoxKalmanFilter
from autonomous_perception_engine.types import BBox


def test_kalman_uses_eight_state_variables_and_four_measurements() -> None:
    tracker = BoundingBoxKalmanFilter(BBox(0, 0, 20, 10), KalmanConfig())
    assert tracker.x.shape == (8, 1)
    assert tracker.H.shape == (4, 8)
    assert tracker.P.shape == (8, 8)


def test_kalman_learns_motion_and_preserves_positive_size() -> None:
    tracker = BoundingBoxKalmanFilter(BBox(0, 0, 20, 10), KalmanConfig())
    for offset in (2.0, 4.0, 6.0, 8.0):
        tracker.predict(1.0)
        tracker.update(BBox(offset, 0, offset + 20, 10))
    previous_center = tracker.bbox.center[0]
    predicted = tracker.predict(1.0)
    assert predicted.center[0] > previous_center
    assert predicted.width >= 1.0
    assert predicted.height >= 1.0
    assert np.allclose(tracker.P, tracker.P.T)


def test_mahalanobis_distance_prefers_nearby_measurement() -> None:
    tracker = BoundingBoxKalmanFilter(BBox(0, 0, 20, 10), KalmanConfig())
    near = tracker.mahalanobis_squared(BBox(1, 0, 21, 10))
    far = tracker.mahalanobis_squared(BBox(200, 200, 220, 210))
    assert near < far
