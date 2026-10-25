"""Shared immutable domain types and bounding-box geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in absolute ``xyxy`` pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bounding-box coordinates must be finite")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid bounding-box ordering: {values}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    @property
    def cxcywh(self) -> tuple[float, float, float, float]:
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, width: float, height: float) -> BBox:
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)
        return cls(
            float(cx) - width / 2.0,
            float(cy) - height / 2.0,
            float(cx) + width / 2.0,
            float(cy) + height / 2.0,
        )

    def clip(self, frame_width: int, frame_height: int) -> BBox:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        max_x = float(frame_width - 1)
        max_y = float(frame_height - 1)
        x1 = min(max(self.x1, 0.0), max_x)
        y1 = min(max(self.y1, 0.0), max_y)
        x2 = min(max(self.x2, x1), max_x)
        y2 = min(max(self.y2, y1), max_y)
        return BBox(x1, y1, x2, y2)

    def iou(self, other: BBox) -> float:
        left = max(self.x1, other.x1)
        top = max(self.y1, other.y1)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class Detection:
    """Normalized detector output consumed by the tracker."""

    bbox: BBox
    class_id: int
    class_name: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    """Read-only track state exposed to renderers, analytics, and exporters."""

    track_id: int
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float
    state: str
    age: int
    hits: int
    time_since_update: int
    history: tuple[tuple[float, float], ...]
