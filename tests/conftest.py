from __future__ import annotations

import pytest

from autonomous_perception_engine.types import BBox, Detection


@pytest.fixture
def detection_factory():
    def make(
        x: float = 10.0,
        y: float = 20.0,
        width: float = 40.0,
        height: float = 30.0,
        confidence: float = 0.9,
        class_id: int = 2,
        class_name: str = "car",
    ) -> Detection:
        return Detection(
            BBox(x, y, x + width, y + height),
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
        )

    return make
