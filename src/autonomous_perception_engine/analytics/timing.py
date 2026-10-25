"""Per-stage timing aggregation without conflating throughput and accuracy."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TimingStats:
    samples: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float


class TimingAccumulator:
    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def add(self, stage: str, seconds: float) -> None:
        if math.isfinite(seconds) and seconds >= 0.0:
            self._samples[stage].append(seconds * 1000.0)

    def summary(self, skip_first: int = 0) -> dict[str, TimingStats]:
        result: dict[str, TimingStats] = {}
        for stage, all_values in sorted(self._samples.items()):
            values = np.asarray(all_values[skip_first:], dtype=np.float64)
            if values.size == 0:
                continue
            result[stage] = TimingStats(
                samples=int(values.size),
                mean_ms=float(values.mean()),
                median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)),
                minimum_ms=float(values.min()),
                maximum_ms=float(values.max()),
            )
        return result

    def latest_fps(self, window: int = 30) -> float:
        values = self._samples.get("total", [])[-window:]
        if not values:
            return 0.0
        mean_ms = sum(values) / len(values)
        return 1000.0 / mean_ms if mean_ms > 0.0 else 0.0
