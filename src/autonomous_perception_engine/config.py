"""Typed YAML configuration with explicit cross-field validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


@dataclass(slots=True)
class DetectorConfig:
    model: str = "yolo11n.pt"
    device: str = "auto"
    image_size: int = 640
    confidence_high: float = 0.50
    confidence_low: float = 0.15
    nms_iou: float = 0.70
    max_detections: int = 300
    half_precision: bool = False
    classes: list[str] = field(
        default_factory=lambda: [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "truck",
            "traffic light",
            "stop sign",
        ]
    )


@dataclass(slots=True)
class KalmanConfig:
    initial_position_variance: float = 25.0
    initial_velocity_variance: float = 400.0
    measurement_position_variance: float = 9.0
    measurement_size_variance: float = 16.0
    process_position_noise: float = 4.0
    process_size_noise: float = 2.0


@dataclass(slots=True)
class AssociationConfig:
    iou_weight: float = 0.55
    center_distance_weight: float = 0.25
    mahalanobis_weight: float = 0.20
    min_iou: float = 0.01
    max_center_distance: float = 2.50
    max_mahalanobis: float = 16.0
    first_stage_max_cost: float = 0.78
    second_stage_max_cost: float = 0.88
    class_aware: bool = True


@dataclass(slots=True)
class TrackerConfig:
    min_hits: int = 3
    max_age: int = 30
    max_tentative_age: int = 2
    render_max_age: int = 2
    history_size: int = 30
    class_vote_window: int = 12
    confidence_smoothing: float = 0.70
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)


@dataclass(slots=True)
class VideoConfig:
    codec: str = "mp4v"
    fallback_codecs: list[str] = field(default_factory=lambda: ["avc1", "XVID"])
    fallback_fps: float = 25.0
    overwrite: bool = False
    preserve_audio: bool = False
    progress_interval: int = 100


@dataclass(slots=True)
class VisualizationConfig:
    enabled: bool = True
    line_thickness: int = 2
    font_scale: float = 0.55
    show_confidence: bool = True
    show_trails: bool = True
    trail_length: int = 24
    show_statistics: bool = True


@dataclass(slots=True)
class CountingConfig:
    enabled: bool = True
    line_enabled: bool = False
    line_start: list[float] = field(default_factory=lambda: [0.10, 0.55])
    line_end: list[float] = field(default_factory=lambda: [0.90, 0.55])
    normalized_coordinates: bool = True


@dataclass(slots=True)
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)

    def validate(self) -> None:
        detector = self.detector
        if not 0.0 <= detector.confidence_low <= detector.confidence_high <= 1.0:
            raise ConfigurationError(
                "Detection thresholds must satisfy 0 <= confidence_low <= confidence_high <= 1"
            )
        if not 0.0 <= detector.nms_iou <= 1.0:
            raise ConfigurationError("detector.nms_iou must be in [0, 1]")
        if detector.image_size <= 0 or detector.max_detections <= 0:
            raise ConfigurationError("Detector image size and max detections must be positive")
        if not detector.classes:
            raise ConfigurationError("At least one detector class must be configured")

        tracker = self.tracker
        if tracker.min_hits < 1 or tracker.max_age < 1:
            raise ConfigurationError("tracker.min_hits and tracker.max_age must be positive")
        if tracker.max_tentative_age < 0 or tracker.render_max_age < 0:
            raise ConfigurationError("Tracker age limits cannot be negative")
        if tracker.history_size < 1 or tracker.class_vote_window < 1:
            raise ConfigurationError("Tracker history and vote windows must be positive")
        if not 0.0 <= tracker.confidence_smoothing < 1.0:
            raise ConfigurationError("tracker.confidence_smoothing must be in [0, 1)")

        association = tracker.association
        weights = (
            association.iou_weight,
            association.center_distance_weight,
            association.mahalanobis_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ConfigurationError("Association weights must be non-negative with a positive sum")
        if not 0.0 <= association.min_iou <= 1.0:
            raise ConfigurationError("association.min_iou must be in [0, 1]")
        if association.max_center_distance <= 0.0 or association.max_mahalanobis <= 0.0:
            raise ConfigurationError("Association gates must be positive")
        if association.first_stage_max_cost < 0.0 or association.second_stage_max_cost < 0.0:
            raise ConfigurationError("Association maximum costs cannot be negative")

        invalid_fallback = any(len(codec) != 4 for codec in self.video.fallback_codecs)
        if len(self.video.codec) != 4 or invalid_fallback:
            raise ConfigurationError("Every video codec must be a four-character code")
        if self.video.fallback_fps <= 0.0 or self.video.progress_interval < 1:
            raise ConfigurationError("Video FPS and progress interval must be positive")

        if len(self.counting.line_start) != 2 or len(self.counting.line_end) != 2:
            raise ConfigurationError("Counting line endpoints must contain exactly two coordinates")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: Any, section: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Configuration section '{section}' must be a mapping")
    return value


def _reject_unknown(data: Mapping[str, Any], cls: type[Any], section: str) -> None:
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown keys in '{section}': {', '.join(unknown)}")


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load an application configuration or return validated defaults."""

    raw: Mapping[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigurationError(f"Configuration file does not exist: {config_path}")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw = _mapping(loaded, "root")

    _reject_unknown(raw, AppConfig, "root")
    detector_raw = _mapping(raw.get("detector"), "detector")
    tracker_raw = _mapping(raw.get("tracker"), "tracker")
    video_raw = _mapping(raw.get("video"), "video")
    visualization_raw = _mapping(raw.get("visualization"), "visualization")
    counting_raw = _mapping(raw.get("counting"), "counting")
    kalman_raw = _mapping(tracker_raw.get("kalman"), "tracker.kalman")
    association_raw = _mapping(tracker_raw.get("association"), "tracker.association")

    for values, cls, section in (
        (detector_raw, DetectorConfig, "detector"),
        (tracker_raw, TrackerConfig, "tracker"),
        (kalman_raw, KalmanConfig, "tracker.kalman"),
        (association_raw, AssociationConfig, "tracker.association"),
        (video_raw, VideoConfig, "video"),
        (visualization_raw, VisualizationConfig, "visualization"),
        (counting_raw, CountingConfig, "counting"),
    ):
        _reject_unknown(values, cls, section)

    tracker_values = dict(tracker_raw)
    tracker_values.pop("kalman", None)
    tracker_values.pop("association", None)
    tracker = TrackerConfig(
        **tracker_values,
        kalman=KalmanConfig(**dict(kalman_raw)),
        association=AssociationConfig(**dict(association_raw)),
    )
    config = AppConfig(
        detector=DetectorConfig(**dict(detector_raw)),
        tracker=tracker,
        video=VideoConfig(**dict(video_raw)),
        visualization=VisualizationConfig(**dict(visualization_raw)),
        counting=CountingConfig(**dict(counting_raw)),
    )
    config.validate()
    return config
