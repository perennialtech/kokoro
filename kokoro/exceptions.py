class TensorRTShapeError(RuntimeError):
    """Raised when TensorRT rejects or cannot resolve tensor shapes."""


class TensorRTExecutionError(RuntimeError):
    """Raised when TensorRT execution fails."""


class TensorRTDeserializationError(RuntimeError):
    """Raised when a serialized TensorRT engine cannot be loaded."""
