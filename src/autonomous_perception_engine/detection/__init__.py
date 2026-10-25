"""Object-detector interfaces and implementations."""

from .base import Detector
from .yolo import YoloDetector

__all__ = ["Detector", "YoloDetector"]
