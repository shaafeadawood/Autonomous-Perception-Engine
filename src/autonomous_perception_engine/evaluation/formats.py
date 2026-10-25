"""MOTChallenge-compatible prediction export."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO
from uuid import uuid4

from ..exceptions import EvaluationError
from ..types import TrackSnapshot


class MotPredictionWriter:
    def __init__(self, path: str | Path, *, overwrite: bool = False) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            raise EvaluationError(
                f"MOT output already exists: {self.path}. Use --overwrite to replace it."
            )
        self.partial_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.partial")
        self._handle: TextIO | None = None

    def __enter__(self) -> Self:
        self._handle = self.partial_path.open("w", encoding="utf-8", newline="")
        return self

    def write(self, frame_number: int, tracks: Sequence[TrackSnapshot]) -> None:
        if self._handle is None:
            raise RuntimeError("MOT writer is not open")
        for track in tracks:
            if track.time_since_update != 0:
                continue
            x, y, width, height = track.bbox.xywh
            self._handle.write(
                f"{frame_number},{track.track_id},{x:.3f},{y:.3f},{width:.3f},{height:.3f},"
                f"{track.confidence:.6f},{track.class_id},1\n"
            )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if exc_type is None:
            os.replace(self.partial_path, self.path)
        else:
            self.partial_path.unlink(missing_ok=True)
