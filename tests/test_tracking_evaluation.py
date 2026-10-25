from __future__ import annotations

from pathlib import Path

import pytest

from autonomous_perception_engine.evaluation.tracking import evaluate_tracking


def test_perfect_mot_predictions_score_perfectly(tmp_path: Path) -> None:
    pytest.importorskip("motmetrics")
    content = "1,1,10,10,20,20,1,2,1\n2,1,12,10,20,20,1,2,1\n"
    ground_truth = tmp_path / "gt.txt"
    predictions = tmp_path / "predictions.txt"
    output = tmp_path / "metrics.json"
    ground_truth.write_text(content, encoding="utf-8")
    predictions.write_text(content, encoding="utf-8")
    result = evaluate_tracking(ground_truth, predictions, output)
    assert result["metrics"]["mota"] == pytest.approx(1.0)
    assert result["metrics"]["idf1"] == pytest.approx(1.0)
    assert output.is_file()
