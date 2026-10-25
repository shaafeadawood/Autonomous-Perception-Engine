"""Validated and failure-safe OpenCV video input/output."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self, cast
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import VideoConfig
from ..exceptions import VideoIOError


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    reported_frames: int


class VideoReader:
    def __init__(self, source: str | int, fallback_fps: float) -> None:
        self.source = source
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            self._capture.release()
            raise VideoIOError(f"Unable to open video source: {source}")

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            self._capture.release()
            raise VideoIOError(f"Video source reports invalid dimensions: {width}x{height}")
        if not math.isfinite(fps) or fps <= 0.0:
            fps = fallback_fps
        self.info = VideoInfo(width=width, height=height, fps=fps, reported_frames=max(frames, 0))

    def read(self) -> NDArray[np.uint8] | None:
        success, frame = self._capture.read()
        if not success:
            return None
        if frame is None or frame.size == 0:
            raise VideoIOError("Video decoder returned an empty frame")
        return cast(NDArray[np.uint8], frame)

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class AtomicVideoWriter:
    """Write beside the destination and publish only a successfully closed file."""

    def __init__(
        self,
        destination: str | Path,
        info: VideoInfo,
        config: VideoConfig,
    ) -> None:
        self.destination = Path(destination).expanduser().resolve()
        self.info = info
        self.config = config
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if self.destination.exists() and not config.overwrite:
            raise VideoIOError(
                f"Output already exists: {self.destination}. Use --overwrite to replace it."
            )
        suffix = self.destination.suffix or ".mp4"
        self.partial_path = self.destination.with_name(
            f".{self.destination.stem}.{uuid4().hex}.partial{suffix}"
        )
        self._writer: cv2.VideoWriter | None = None
        self.codec = self._open_writer([config.codec, *config.fallback_codecs])

    def _open_writer(self, codecs: list[str]) -> str:
        attempted: list[str] = []
        for codec in dict.fromkeys(codecs):
            attempted.append(codec)
            writer = cv2.VideoWriter(
                str(self.partial_path),
                cv2.VideoWriter.fourcc(*codec),
                self.info.fps,
                (self.info.width, self.info.height),
            )
            if writer.isOpened():
                self._writer = writer
                return codec
            writer.release()
            self.partial_path.unlink(missing_ok=True)
        raise VideoIOError(
            "Unable to initialize video writer; attempted codecs: " + ", ".join(attempted)
        )

    def write(self, frame: NDArray[np.uint8]) -> None:
        if self._writer is None:
            raise VideoIOError("Cannot write after the video writer has been closed")
        if frame.shape[:2] != (self.info.height, self.info.width):
            raise VideoIOError(
                f"Frame size {frame.shape[1]}x{frame.shape[0]} does not match "
                f"output {self.info.width}x{self.info.height}"
            )
        self._writer.write(frame)

    def finalize(self) -> Path:
        self._release()
        if not self.partial_path.is_file() or self.partial_path.stat().st_size == 0:
            self.abort()
            raise VideoIOError("Video writer produced an empty output file")
        if self.destination.exists() and not self.config.overwrite:
            self.abort()
            raise VideoIOError(f"Output appeared during processing: {self.destination}")
        os.replace(self.partial_path, self.destination)
        return self.destination

    def abort(self) -> None:
        self._release()
        self.partial_path.unlink(missing_ok=True)

    def _release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def remux_source_audio(source: str, processed_video: Path, overwrite: bool) -> bool:
    """Copy source audio into a completed video; return False when FFmpeg is absent."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    temporary = processed_video.with_name(
        f".{processed_video.stem}.{uuid4().hex}.audio{processed_video.suffix}"
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(processed_video),
        "-i",
        source,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise VideoIOError(f"FFmpeg audio remux failed: {completed.stderr.strip()}")
    os.replace(temporary, processed_video)
    return True
