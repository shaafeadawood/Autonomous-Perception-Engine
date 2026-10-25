"""Detector protocol used to decouple inference from the video pipeline."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..types import Detection


class Detector(Protocol):
    def predict(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Return normalized detections for one BGR frame."""

    def synchronize(self) -> None:
        """Synchronize asynchronous device work for honest timing."""
