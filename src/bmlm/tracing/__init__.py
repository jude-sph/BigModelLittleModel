"""Tracing module for Phoenix observability."""

from bmlm.tracing.setup import init_tracing, shutdown_tracing, is_tracing_enabled
from bmlm.tracing.spans import (
    trace_big_model,
    trace_small_model,
    trace_action,
    trace_task,
    trace_verification,
    set_task_metadata,
)

__all__ = [
    "init_tracing",
    "shutdown_tracing",
    "is_tracing_enabled",
    "trace_big_model",
    "trace_small_model",
    "trace_action",
    "trace_task",
    "trace_verification",
    "set_task_metadata",
]
