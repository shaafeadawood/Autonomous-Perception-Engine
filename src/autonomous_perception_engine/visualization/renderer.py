"""Readable, deterministic tracking overlays."""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from ..analytics.counting import AnalyticsSnapshot, ObjectCounter
from ..config import CountingConfig, VisualizationConfig
from ..types import TrackSnapshot


class Renderer:
    def __init__(
        self,
        config: VisualizationConfig,
        counting_config: CountingConfig,
        counter: ObjectCounter,
    ) -> None:
        self.config = config
        self.counting_config = counting_config
        self.counter = counter

    @staticmethod
    def color_for_id(track_id: int) -> tuple[int, int, int]:
        hue = (track_id * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
        return (int(blue * 255), int(green * 255), int(red * 255))

    def draw(
        self,
        frame: NDArray[np.uint8],
        tracks: Sequence[TrackSnapshot],
        analytics: AnalyticsSnapshot,
        fps: float,
    ) -> NDArray[np.uint8]:
        if not self.config.enabled:
            return frame
        height, width = frame.shape[:2]
        for track in tracks:
            box = track.bbox.clip(width, height)
            x1, y1, x2, y2 = (int(value) for value in (box.x1, box.y1, box.x2, box.y2))
            color = self.color_for_id(track.track_id)
            thickness = self.config.line_thickness if track.time_since_update == 0 else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            label = f"#{track.track_id} {track.class_name}"
            if self.config.show_confidence:
                label += f" {track.confidence:.2f}"
            if track.time_since_update > 0:
                label += " predicted"
            self._label(frame, label, x1, y1, color)

            if self.config.show_trails and len(track.history) > 1:
                points = list(track.history)[-self.config.trail_length :]
                clipped = np.asarray(
                    [
                        [
                            int(min(max(x, 0.0), width - 1)),
                            int(min(max(y, 0.0), height - 1)),
                        ]
                        for x, y in points
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(frame, [clipped], False, color, 2, cv2.LINE_AA)

        if self.counting_config.line_enabled:
            start, end = self.counter.line_points(width, height)
            cv2.line(frame, start, end, (255, 220, 0), 2, cv2.LINE_AA)

        if self.config.show_statistics:
            self._statistics_panel(frame, analytics, fps)
        return frame

    def _label(
        self,
        frame: NDArray[np.uint8],
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self.config.font_scale
        thickness = max(1, self.config.line_thickness - 1)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        top = max(0, y - text_height - baseline - 8)
        right = min(frame.shape[1] - 1, x + text_width + 8)
        cv2.rectangle(frame, (x, top), (right, y), color, -1)
        luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if luminance > 150 else (255, 255, 255)
        cv2.putText(
            frame,
            text,
            (x + 4, max(text_height + 2, y - baseline - 4)),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _statistics_panel(
        frame: NDArray[np.uint8], analytics: AnalyticsSnapshot, fps: float
    ) -> None:
        lines = [
            f"FPS: {fps:5.1f}",
            f"Active tracks: {analytics.active_tracks}",
            f"Unique tracks: {analytics.unique_tracks}",
        ]
        if analytics.active_by_class:
            classes = "  ".join(
                f"{name}:{count}" for name, count in analytics.active_by_class.items()
            )
            lines.append(classes)
        if analytics.crossings_a_to_b or analytics.crossings_b_to_a:
            lines.append(
                f"Crossings A>B: {analytics.crossings_a_to_b}  B>A: {analytics.crossings_b_to_a}"
            )

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        line_height = 24
        widths = [cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines]
        panel_width = min(max(widths, default=180) + 24, frame.shape[1])
        panel_height = min(len(lines) * line_height + 18, frame.shape[0])
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (12, 18, 28), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0.0, frame)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (12, 24 + index * line_height),
                font,
                scale,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
