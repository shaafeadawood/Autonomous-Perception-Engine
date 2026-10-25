"""Project-specific exceptions with actionable user-facing messages."""


class PerceptionEngineError(RuntimeError):
    """Base class for expected application failures."""


class ConfigurationError(PerceptionEngineError):
    """Raised when configuration values are missing or inconsistent."""


class DetectorError(PerceptionEngineError):
    """Raised when the detector cannot be initialized or executed."""


class VideoIOError(PerceptionEngineError):
    """Raised when a video source or output cannot be opened safely."""


class EvaluationError(PerceptionEngineError):
    """Raised when evaluation inputs are invalid or dependencies are absent."""
