"""Ground-truth detection evaluation through Ultralytics validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..exceptions import EvaluationError


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_detection(
    config: AppConfig,
    data: str | Path,
    output: str | Path,
    split: str = "val",
) -> dict[str, Any]:
    """Evaluate against a labeled YOLO dataset and save machine-readable metrics."""

    data_path = Path(data).expanduser().resolve()
    if not data_path.is_file():
        raise EvaluationError(f"Dataset YAML does not exist: {data_path}")
    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except ImportError as exc:
        raise EvaluationError("Ultralytics is required for detection evaluation") from exc

    try:
        model = YOLO(config.detector.model)
        arguments: dict[str, Any] = {
            "data": str(data_path),
            "split": split,
            "imgsz": config.detector.image_size,
            "iou": config.detector.nms_iou,
            "max_det": config.detector.max_detections,
            "verbose": False,
        }
        if config.detector.device.lower() != "auto":
            arguments["device"] = config.detector.device
        metrics = model.val(**arguments)
    except Exception as exc:
        raise EvaluationError(f"Detection evaluation failed: {exc}") from exc

    box = getattr(metrics, "box", None)
    if box is None:
        raise EvaluationError("Ultralytics returned no bounding-box metrics")
    class_names = getattr(metrics, "names", {})
    per_class: list[dict[str, Any]] = []
    maps = getattr(box, "maps", None)
    if maps is not None:
        for class_id, map_value in enumerate(maps):
            name = (
                class_names.get(class_id, str(class_id))
                if isinstance(class_names, dict)
                else str(class_id)
            )
            per_class.append(
                {"class_id": class_id, "class_name": name, "map50_95": float(map_value)}
            )

    result: dict[str, Any] = {
        "schema_version": 1,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "model": config.detector.model,
        "dataset": str(data_path),
        "split": split,
        "metrics": {
            "precision": _optional_float(getattr(box, "mp", None)),
            "recall": _optional_float(getattr(box, "mr", None)),
            "map50": _optional_float(getattr(box, "map50", None)),
            "map50_95": _optional_float(getattr(box, "map", None)),
            "map75": _optional_float(getattr(box, "map75", None)),
        },
        "per_class": per_class,
        "config": config.to_dict(),
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
