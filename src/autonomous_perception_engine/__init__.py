"""Autonomous Perception Engine public package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("autonomous-perception-engine")
except PackageNotFoundError:  # Running directly from a source checkout.
    __version__ = "0.1.0"

__all__ = ["__version__"]
