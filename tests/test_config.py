from __future__ import annotations

from pathlib import Path

import pytest

from autonomous_perception_engine.config import load_config
from autonomous_perception_engine.exceptions import ConfigurationError


def test_default_configuration_is_valid() -> None:
    config = load_config()
    assert config.detector.model == "yolo11n.pt"
    assert config.detector.confidence_low < config.detector.confidence_high
    assert config.tracker.kalman.initial_velocity_variance > 0


def test_partial_nested_configuration_keeps_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("tracker:\n  min_hits: 5\n  association:\n    min_iou: 0.2\n", encoding="utf-8")
    config = load_config(path)
    assert config.tracker.min_hits == 5
    assert config.tracker.association.min_iou == pytest.approx(0.2)
    assert config.tracker.max_age == 30
    assert config.detector.model == "yolo11n.pt"


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("detector:\n  imaginary_option: true\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="imaginary_option"):
        load_config(path)


def test_inverted_confidence_thresholds_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "detector:\n  confidence_low: 0.8\n  confidence_high: 0.4\n", encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="confidence_low"):
        load_config(path)
