"""Custom exceptions for pipeline boundaries."""


class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails in a controlled way."""
