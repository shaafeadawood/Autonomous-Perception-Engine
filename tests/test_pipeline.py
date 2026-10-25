from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from autonomous_perception_engine.config import load_config
from autonomous_perception_engine.pipeline.runner import run_video
from autonomous_perception_engine.types import BBox, Detection


class FakeDetector:
    def __init__(self) -> None:
        self.frame = 0

    def predict(self, frame: np.ndarray) -> list[Detection]:
        self.frame += 1
        x = 5.0 + self.frame * 2.0
        return [Detection(BBox(x, 10, x + 20, 30), 2, "car", 0.9)]

    def synchronize(self) -> None:
        return


def _write_input(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for index in range(6):
        writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    writer.release()


def test_end_to_end_pipeline_with_injected_detector(tmp_path: Path) -> None:
    source = tmp_path / "source.avi"
    output = tmp_path / "output.avi"
    mot_output = tmp_path / "tracks.txt"
    _write_input(source)
    config = load_config()
    config.tracker.min_hits = 1
    config.video.codec = "MJPG"
    config.video.fallback_codecs = ["XVID"]
    config.video.overwrite = False
    summary = run_video(
        str(source),
        output,
        config,
        mot_output=mot_output,
        detector=FakeDetector(),
    )
    assert summary.frames_processed == 6
    assert summary.unique_tracks == 1
    assert output.is_file()
    assert mot_output.read_text(encoding="utf-8").count("\n") == 6
    manifest = json.loads(Path(summary.manifest).read_text(encoding="utf-8"))
    assert manifest["frames_processed"] == 6
    assert "accuracy_notice" in manifest
