"""Identity-aware MOTChallenge tracking evaluation."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import EvaluationError


def _load_mot(path: Path, ground_truth: bool) -> dict[int, list[tuple[int, list[float]]]]:
    frames: dict[int, list[tuple[int, list[float]]]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle), start=1):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                if len(row) < 6:
                    raise EvaluationError(f"{path}:{row_number} has fewer than six MOT columns")
                frame_id = int(float(row[0]))
                object_id = int(float(row[1]))
                box = [float(value) for value in row[2:6]]
                if box[2] <= 0.0 or box[3] <= 0.0:
                    continue
                confidence = float(row[6]) if len(row) > 6 else 1.0
                if ground_truth and confidence <= 0.0:
                    continue
                frames.setdefault(frame_id, []).append((object_id, box))
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Unable to parse MOT file '{path}': {exc}") from exc
    return frames


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _iou_distance_matrix(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return MOT distances without relying on motmetrics' removed NumPy 1.x APIs."""

    distances = np.full((len(ground_truth), len(predictions)), np.nan, dtype=np.float64)
    for ground_truth_index, first in enumerate(ground_truth):
        first_x1, first_y1, first_width, first_height = first
        first_x2 = first_x1 + first_width
        first_y2 = first_y1 + first_height
        first_area = first_width * first_height
        for prediction_index, second in enumerate(predictions):
            second_x1, second_y1, second_width, second_height = second
            second_x2 = second_x1 + second_width
            second_y2 = second_y1 + second_height
            intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
            intersection_height = max(
                0.0, min(first_y2, second_y2) - max(first_y1, second_y1)
            )
            intersection = intersection_width * intersection_height
            union = first_area + second_width * second_height - intersection
            iou = intersection / union if union > 0.0 else 0.0
            if iou >= iou_threshold:
                distances[ground_truth_index, prediction_index] = 1.0 - iou
    return distances


def evaluate_tracking(
    ground_truth: str | Path,
    predictions: str | Path,
    output: str | Path,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute MOT metrics from ground truth and generated prediction files."""

    if not 0.0 < iou_threshold <= 1.0:
        raise EvaluationError("Tracking IoU threshold must be in (0, 1]")
    gt_path = Path(ground_truth).expanduser().resolve()
    prediction_path = Path(predictions).expanduser().resolve()
    if not gt_path.is_file() or not prediction_path.is_file():
        raise EvaluationError("Both ground-truth and prediction MOT files must exist")
    try:
        import motmetrics as mm
    except ImportError as exc:
        raise EvaluationError(
            "motmetrics is required; install the project's 'evaluation' dependency group"
        ) from exc

    gt_frames = _load_mot(gt_path, ground_truth=True)
    prediction_frames = _load_mot(prediction_path, ground_truth=False)
    accumulator = mm.MOTAccumulator(auto_id=False)
    frame_ids = sorted(set(gt_frames) | set(prediction_frames))
    for frame_id in frame_ids:
        gt = gt_frames.get(frame_id, [])
        predictions_for_frame = prediction_frames.get(frame_id, [])
        gt_ids = [item[0] for item in gt]
        prediction_ids = [item[0] for item in predictions_for_frame]
        gt_boxes = np.asarray([item[1] for item in gt], dtype=np.float64).reshape(-1, 4)
        prediction_boxes = np.asarray(
            [item[1] for item in predictions_for_frame], dtype=np.float64
        ).reshape(-1, 4)
        distances = _iou_distance_matrix(gt_boxes, prediction_boxes, iou_threshold)
        accumulator.update(gt_ids, prediction_ids, distances, frameid=frame_id)

    metric_names = [
        "num_frames",
        "mota",
        "motp",
        "idf1",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "precision",
        "recall",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_fragmentations",
    ]
    summary = mm.metrics.create().compute(accumulator, metrics=metric_names, name="sequence")
    values = {name: _json_value(summary.loc["sequence", name]) for name in metric_names}
    result: dict[str, Any] = {
        "schema_version": 1,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "ground_truth": str(gt_path),
        "predictions": str(prediction_path),
        "iou_threshold": iou_threshold,
        "metrics": values,
        "motp_notice": "motmetrics reports MOTP as average assignment distance; lower is better.",
        "hota_notice": (
            "HOTA requires a separate TrackEval run; MOT export is compatible with that workflow."
        ),
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
