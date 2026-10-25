from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autonomous_perception_engine.config import VideoConfig
from autonomous_perception_engine.exceptions import VideoIOError
from autonomous_perception_engine.pipeline.video_io import AtomicVideoWriter, VideoInfo, VideoReader


def test_atomic_video_writer_and_reader(tmp_path: Path) -> None:
    destination = tmp_path / "sample.avi"
    config = VideoConfig(codec="MJPG", fallback_codecs=["XVID"], overwrite=False)
    info = VideoInfo(width=64, height=48, fps=10.0, reported_frames=0)
    writer = AtomicVideoWriter(destination, info, config)
    for value in range(5):
        frame = np.full((48, 64, 3), value * 30, dtype=np.uint8)
        writer.write(frame)
    writer.finalize()
    assert destination.is_file()
    assert not writer.partial_path.exists()

    with VideoReader(str(destination), fallback_fps=25.0) as reader:
        frames = 0
        while reader.read() is not None:
            frames += 1
        assert frames == 5
        assert reader.info.width == 64


def test_writer_refuses_existing_output(tmp_path: Path) -> None:
    destination = tmp_path / "existing.avi"
    destination.write_bytes(b"occupied")
    config = VideoConfig(codec="MJPG", fallback_codecs=[], overwrite=False)
    with pytest.raises(VideoIOError, match="already exists"):
        AtomicVideoWriter(destination, VideoInfo(10, 10, 5.0, 0), config)
