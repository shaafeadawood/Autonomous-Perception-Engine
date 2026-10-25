"""Runtime benchmarking with warm-up exclusion and device synchronization."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..analytics.timing import TimingAccumulator
from ..config import AppConfig
from ..detection.base import Detector
from ..detection.yolo import YoloDetector
from ..pipeline.video_io import VideoReader
from ..tracking.manager import MultiObjectTracker


def benchmark_video(
    source: str | int,
    config: AppConfig,
    output: str | Path,
    *,
    max_frames: int = 300,
    warmup_frames: int = 10,
    detector: Detector | None = None,
) -> dict[str, Any]:
    if max_frames < 1 or warmup_frames < 0 or warmup_frames >= max_frames:
        raise ValueError("Require max_frames >= 1 and 0 <= warmup_frames < max_frames")
    detector = detector or YoloDetector(config.detector)
    tracker = MultiObjectTracker(config.tracker, config.detector.confidence_high)
    timings = TimingAccumulator()
    frames = 0

    with VideoReader(source, config.video.fallback_fps) as reader:
        dt = 1.0 / reader.info.fps
        while frames < max_frames:
            total_started = time.perf_counter()
            read_started = time.perf_counter()
            frame = reader.read()
            timings.add("read", time.perf_counter() - read_started)
            if frame is None:
                break
            detection_started = time.perf_counter()
            detections = detector.predict(frame)
            detector.synchronize()
            timings.add("detection", time.perf_counter() - detection_started)
            tracking_started = time.perf_counter()
            tracker.update(detections, dt)
            timings.add("tracking", time.perf_counter() - tracking_started)
            timings.add("total", time.perf_counter() - total_started)
            frames += 1

        video_info = asdict(reader.info)

    measured_frames = max(frames - warmup_frames, 0)
    summaries = {
        stage: asdict(stats)
        for stage, stats in timings.summary(skip_first=warmup_frames).items()
    }
    total_mean = summaries.get("total", {}).get("mean_ms", 0.0)
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmarked_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "frames_processed": frames,
        "warmup_frames_excluded": min(warmup_frames, frames),
        "measured_frames": measured_frames,
        "effective_fps_from_mean_latency": 1000.0 / total_mean if total_mean else None,
        "timings": summaries,
        "video": video_info,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": config.to_dict(),
        "accuracy_notice": "This benchmark measures runtime only and does not measure accuracy.",
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
