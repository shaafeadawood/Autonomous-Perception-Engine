"""End-to-end perception pipeline and reproducible run manifest."""

from __future__ import annotations

import json
import logging
import platform
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2

from ..analytics.counting import AnalyticsSnapshot, ObjectCounter
from ..analytics.timing import TimingAccumulator
from ..config import AppConfig
from ..detection.base import Detector
from ..detection.yolo import YoloDetector
from ..evaluation.formats import MotPredictionWriter
from ..tracking.manager import MultiObjectTracker
from ..visualization.renderer import Renderer
from .video_io import AtomicVideoWriter, VideoReader, remux_source_audio

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunSummary:
    output: str
    manifest: str
    frames_processed: int
    detections: int
    unique_tracks: int
    unique_by_class: dict[str, int]
    stopped_early: bool
    codec: str
    audio_preserved: bool


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    packages = (
        "autonomous-perception-engine",
        "ultralytics",
        "opencv-python",
        "numpy",
        "scipy",
    )
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".run.json")


def run_video(
    source: str | int,
    output: str | Path,
    config: AppConfig,
    *,
    display: bool = False,
    mot_output: str | Path | None = None,
    detector: Detector | None = None,
) -> RunSummary:
    """Run detection, tracking, analytics, rendering, and safe video output."""

    config.validate()
    detector = detector or YoloDetector(config.detector)
    tracker = MultiObjectTracker(config.tracker, config.detector.confidence_high)
    counter = ObjectCounter(config.counting)
    renderer = Renderer(config.visualization, config.counting, counter)
    timings = TimingAccumulator()
    started_at = datetime.now(UTC)
    frame_count = 0
    detection_count = 0
    stopped_early = False
    latest_analytics = AnalyticsSnapshot(0, {}, 0, {}, 0, 0)
    output_path = Path(output).expanduser().resolve()

    mot_context = (
        MotPredictionWriter(mot_output, overwrite=config.video.overwrite)
        if mot_output
        else nullcontext(None)
    )
    with VideoReader(source, config.video.fallback_fps) as reader, mot_context as mot_writer:
        writer = AtomicVideoWriter(output_path, reader.info, config.video)
        dt = 1.0 / reader.info.fps
        try:
            while True:
                frame_started = time.perf_counter()
                read_started = time.perf_counter()
                frame = reader.read()
                timings.add("read", time.perf_counter() - read_started)
                if frame is None:
                    break

                detection_started = time.perf_counter()
                detections = detector.predict(frame)
                detector.synchronize()
                timings.add("detection", time.perf_counter() - detection_started)
                detection_count += len(detections)

                tracking_started = time.perf_counter()
                tracks = tracker.update(detections, dt)
                timings.add("tracking", time.perf_counter() - tracking_started)

                analytics_started = time.perf_counter()
                latest_analytics = counter.update(tracks, reader.info.width, reader.info.height)
                timings.add("analytics", time.perf_counter() - analytics_started)

                rendering_started = time.perf_counter()
                renderer.draw(frame, tracks, latest_analytics, timings.latest_fps())
                timings.add("rendering", time.perf_counter() - rendering_started)

                writing_started = time.perf_counter()
                writer.write(frame)
                timings.add("write", time.perf_counter() - writing_started)
                frame_count += 1
                if mot_writer is not None:
                    mot_writer.write(frame_count, tracks)

                timings.add("total", time.perf_counter() - frame_started)
                if frame_count % config.video.progress_interval == 0:
                    LOGGER.info(
                        "Processed %d frames | active=%d unique=%d rolling_fps=%.1f",
                        frame_count,
                        latest_analytics.active_tracks,
                        latest_analytics.unique_tracks,
                        timings.latest_fps(),
                    )
                if display:
                    cv2.imshow("Autonomous Perception Engine", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stopped_early = True
                        break
            writer.finalize()
        except BaseException:
            writer.abort()
            raise
        finally:
            if display:
                cv2.destroyAllWindows()

    audio_preserved = False
    if config.video.preserve_audio and isinstance(source, str):
        audio_preserved = remux_source_audio(source, output_path, overwrite=True)

    completed_at = datetime.now(UTC)
    timing_summary = {
        stage: asdict(stats) for stage, stats in timings.summary().items()
    }
    manifest_path = _manifest_path(output_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "source": str(source),
        "output": str(output_path),
        "frames_processed": frame_count,
        "reported_source_frames": reader.info.reported_frames,
        "resolution": [reader.info.width, reader.info.height],
        "source_fps": reader.info.fps,
        "detections": detection_count,
        "unique_tracks": latest_analytics.unique_tracks,
        "unique_by_class": latest_analytics.unique_by_class,
        "stopped_early": stopped_early,
        "output_codec": writer.codec,
        "audio_preserved": audio_preserved,
        "timings": timing_summary,
        "config": config.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "accuracy_notice": (
            "Runtime statistics are not accuracy metrics. Use labeled data and the evaluation "
            "commands before reporting detection or tracking quality."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return RunSummary(
        output=str(output_path),
        manifest=str(manifest_path),
        frames_processed=frame_count,
        detections=detection_count,
        unique_tracks=latest_analytics.unique_tracks,
        unique_by_class=latest_analytics.unique_by_class,
        stopped_early=stopped_early,
        codec=writer.codec,
        audio_preserved=audio_preserved,
    )
