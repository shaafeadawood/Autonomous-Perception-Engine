"""Ultralytics YOLO11 detector adapter."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import DetectorConfig
from ..exceptions import DetectorError
from ..types import BBox, Detection

LOGGER = logging.getLogger(__name__)


class YoloDetector:
    """Convert Ultralytics-specific results into project domain objects."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        try:
            from ultralytics import YOLO  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover - exercised in installation failures.
            raise DetectorError(
                "Ultralytics is not installed. Install the project dependencies before inference."
            ) from exc

        try:
            self.model = YOLO(config.model)
        except Exception as exc:
            raise DetectorError(f"Unable to load YOLO model '{config.model}': {exc}") from exc

        names = self.model.names
        self.names: dict[int, str] = (
            {int(index): str(name) for index, name in names.items()}
            if isinstance(names, dict)
            else {index: str(name) for index, name in enumerate(names)}
        )
        wanted = set(config.classes)
        self.class_ids = sorted(index for index, name in self.names.items() if name in wanted)
        missing = sorted(wanted - set(self.names.values()))
        if missing:
            raise DetectorError(
                "Configured classes are absent from the model: " + ", ".join(missing)
            )
        if not self.class_ids:
            raise DetectorError("No configured target classes are available in the model")
        LOGGER.info("Loaded %s with %d target classes", config.model, len(self.class_ids))

    def predict(self, frame: NDArray[np.uint8]) -> list[Detection]:
        arguments: dict[str, Any] = {
            "source": frame,
            "conf": self.config.confidence_low,
            "iou": self.config.nms_iou,
            "imgsz": self.config.image_size,
            "classes": self.class_ids,
            "max_det": self.config.max_detections,
            "verbose": False,
        }
        if self.config.half_precision:
            arguments["half"] = True
        if self.config.device.lower() != "auto":
            arguments["device"] = self.config.device

        try:
            results: Any = self.model.predict(**arguments)
            result: Any = results[0] if isinstance(results, list) else next(iter(results))
        except Exception as exc:
            raise DetectorError(f"YOLO inference failed: {exc}") from exc

        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = np.asarray(result.boxes.xyxy.cpu(), dtype=np.float64)
        class_ids = np.asarray(result.boxes.cls.cpu(), dtype=np.int64)
        confidences = np.asarray(result.boxes.conf.cpu(), dtype=np.float64)
        detections: list[Detection] = []
        for coordinates, class_id, confidence in zip(
            xyxy, class_ids, confidences, strict=True
        ):
            x1, y1, x2, y2 = (float(value) for value in coordinates)
            detections.append(
                Detection(
                    bbox=BBox(x1, y1, x2, y2),
                    class_id=int(class_id),
                    class_name=self.names[int(class_id)],
                    confidence=float(confidence),
                )
            )
        return detections

    def synchronize(self) -> None:
        if self.config.device.lower() == "cpu":
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            return
