"""Gated multi-cue Hungarian detection-to-track association."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import AssociationConfig
from ..types import BBox, Detection
from .track import Track


@dataclass(frozen=True, slots=True)
class AssociationResult:
    matches: tuple[tuple[int, int], ...]
    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]


def normalized_center_distance(first: BBox, second: BBox) -> float:
    first_x, first_y = first.center
    second_x, second_y = second.center
    distance = math.hypot(first_x - second_x, first_y - second_y)
    first_diagonal = math.hypot(first.width, first.height)
    second_diagonal = math.hypot(second.width, second.height)
    scale = max((first_diagonal + second_diagonal) / 2.0, 1.0)
    return distance / scale


def _pair_cost(track: Track, detection: Detection, config: AssociationConfig) -> float | None:
    if config.class_aware and track.class_id != detection.class_id:
        return None

    iou = track.bbox.iou(detection.bbox)
    center_distance = normalized_center_distance(track.bbox, detection.bbox)
    mahalanobis = track.kalman.mahalanobis_squared(detection.bbox)

    if mahalanobis > config.max_mahalanobis:
        return None
    if iou < config.min_iou and center_distance > config.max_center_distance:
        return None

    weight_sum = config.iou_weight + config.center_distance_weight + config.mahalanobis_weight
    return (
        config.iou_weight * (1.0 - iou)
        + config.center_distance_weight * min(center_distance / config.max_center_distance, 1.0)
        + config.mahalanobis_weight * min(mahalanobis / config.max_mahalanobis, 1.0)
    ) / weight_sum


def associate(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    config: AssociationConfig,
    max_cost: float,
) -> AssociationResult:
    """Associate valid pairs globally and return sequence-relative indices."""

    if not tracks or not detections:
        return AssociationResult(
            matches=(),
            unmatched_tracks=tuple(range(len(tracks))),
            unmatched_detections=tuple(range(len(detections))),
        )

    invalid_cost = max(max_cost + 1.0, 1_000.0)
    costs = np.full((len(tracks), len(detections)), invalid_cost, dtype=np.float64)
    valid = np.zeros_like(costs, dtype=np.bool_)
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            cost = _pair_cost(track, detection, config)
            if cost is not None and cost <= max_cost:
                costs[track_index, detection_index] = cost
                valid[track_index, detection_index] = True

    row_indices, column_indices = linear_sum_assignment(costs)
    matches = tuple(
        (int(row), int(column))
        for row, column in zip(row_indices, column_indices, strict=True)
        if valid[row, column]
    )
    matched_tracks = {track_index for track_index, _ in matches}
    matched_detections = {detection_index for _, detection_index in matches}
    return AssociationResult(
        matches=matches,
        unmatched_tracks=tuple(
            index for index in range(len(tracks)) if index not in matched_tracks
        ),
        unmatched_detections=tuple(
            index for index in range(len(detections)) if index not in matched_detections
        ),
    )
