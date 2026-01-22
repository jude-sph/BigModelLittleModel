"""Phoenix tracing setup and configuration."""

import atexit
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global state
_tracer_provider = None
_tracing_enabled = False


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    return _tracing_enabled


def init_tracing(
    project_name: str = "bmlm",
    endpoint: Optional[str] = None,
) -> bool:
    """Initialize Phoenix tracing.

    Args:
        project_name: Name for the Phoenix project
        endpoint: Phoenix endpoint (default: local Phoenix server)

    Returns:
        True if tracing was initialized successfully
    """
    global _tracer_provider, _tracing_enabled

    if _tracing_enabled:
        logger.info("Tracing already initialized")
        return True

    try:
        import phoenix as px
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from phoenix.otel import register

        # Launch Phoenix in the background (opens browser by default)
        # Use launch_app=False to just start the server without opening browser
        session = px.launch_app()
        logger.info(f"Phoenix UI available at: {session.url}")

        # Register the tracer provider
        _tracer_provider = register(project_name=project_name)

        _tracing_enabled = True
        logger.info(f"Tracing initialized for project: {project_name}")

        # Register shutdown handler
        atexit.register(shutdown_tracing)

        return True

    except ImportError as e:
        logger.warning(
            f"Phoenix not installed. Install with: uv sync --extra tracing\n"
            f"Error: {e}"
        )
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize tracing: {e}")
        return False


def shutdown_tracing() -> None:
    """Shutdown tracing and flush remaining spans."""
    global _tracer_provider, _tracing_enabled

    if not _tracing_enabled:
        return

    try:
        if _tracer_provider and hasattr(_tracer_provider, "force_flush"):
            logger.info("Flushing traces...")
            _tracer_provider.force_flush(timeout_millis=5000)
            logger.info("Traces flushed")
    except Exception as e:
        logger.warning(f"Error flushing traces: {e}")

    _tracing_enabled = False


def get_tracer(name: str = "bmlm"):
    """Get a tracer instance.

    Args:
        name: Tracer name (typically module name)

    Returns:
        Tracer instance or NoOpTracer if tracing disabled
    """
    if not _tracing_enabled:
        from opentelemetry.trace import NoOpTracer
        return NoOpTracer()

    from opentelemetry import trace
    return trace.get_tracer(name)
