"""Errors raised while assembling or running a consolidation pipeline."""


class PipelineError(Exception):
    """A pipeline was mis-assembled or a stage broke its contract."""
