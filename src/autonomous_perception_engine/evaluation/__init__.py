"""Detection, tracking, and runtime evaluation entry points."""

from .benchmark import benchmark_video
from .detection import evaluate_detection
from .tracking import evaluate_tracking

__all__ = ["benchmark_video", "evaluate_detection", "evaluate_tracking"]
