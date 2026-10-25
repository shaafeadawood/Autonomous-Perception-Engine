from __future__ import annotations

from pathlib import Path

import pytest

from autonomous_perception_engine.evaluation.formats import MotPredictionWriter
from autonomous_perception_engine.exceptions import EvaluationError
from autonomous_perception_engine.types import BBox, TrackSnapshot


def _track() -> TrackSnapshot:
    return TrackSnapshot(
        track_id=4,
        bbox=BBox(10, 20, 30, 50),
        class_id=2,
        class_name="car",
        confidence=0.8,
        state="confirmed",
        age=3,
        hits=3,
        time_since_update=0,
        history=((20, 35),),
    )


def test_mot_writer_publishes_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "tracks.txt"
    with MotPredictionWriter(destination) as writer:
        writer.write(1, [_track()])
        assert not destination.exists()
        assert writer.partial_path.exists()
    assert destination.read_text(encoding="utf-8").startswith("1,4,")


def test_mot_writer_removes_partial_file_after_failure(tmp_path: Path) -> None:
    destination = tmp_path / "tracks.txt"
    writer = MotPredictionWriter(destination)
    with pytest.raises(RuntimeError, match="simulated"):
        with writer:
            writer.write(1, [_track()])
            raise RuntimeError("simulated")
    assert not destination.exists()
    assert not writer.partial_path.exists()


def test_mot_writer_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "tracks.txt"
    destination.write_text("existing", encoding="utf-8")
    with pytest.raises(EvaluationError, match="already exists"):
        MotPredictionWriter(destination)
