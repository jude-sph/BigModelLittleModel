"""Phoenix tracing setup and configuration."""

import atexit
import logging
import os
import socket
from typing import Optional

logger = logging.getLogger(__name__)

# Global state
_tracer_provider = None
_tracing_enabled = False
_launched_server = False  # Track if we launched the server (vs connecting to existing)

# Default Phoenix endpoint
PHOENIX_HOST = "localhost"
PHOENIX_PORT = 6006
PHOENIX_GRPC_PORT = 4317


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    return _tracing_enabled


def _is_phoenix_running(host: str = PHOENIX_HOST, port: int = PHOENIX_PORT) -> bool:
    """Check if Phoenix server is already running."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect((host, port))
            return True
    except (socket.error, socket.timeout):
        return False


def init_tracing(
    project_name: str = "bmlm",
    endpoint: Optional[str] = None,
    launch_server: bool = True,
) -> bool:
    """Initialize Phoenix tracing.

    Connects to an existing Phoenix server if running, or launches one if not.
    Data persists in ~/.phoenix between runs.

    To run a persistent Phoenix server separately:
        uv run phoenix serve

    Args:
        project_name: Name for the Phoenix project
        endpoint: Phoenix endpoint (default: localhost:4317)
        launch_server: If True, launch Phoenix if not running. If False, only connect.

    Returns:
        True if tracing was initialized successfully
    """
    global _tracer_provider, _tracing_enabled, _launched_server

    if _tracing_enabled:
        logger.info("Tracing already initialized")
        return True

    try:
        import phoenix as px
        from phoenix.otel import register

        phoenix_running = _is_phoenix_running()

        if phoenix_running:
            logger.info(f"Connecting to existing Phoenix server at http://{PHOENIX_HOST}:{PHOENIX_PORT}")
        elif launch_server:
            # Launch Phoenix server
            logger.info("Starting Phoenix server...")
            px.launch_app()
            _launched_server = True
            logger.info(f"Phoenix UI available at http://{PHOENIX_HOST}:{PHOENIX_PORT}")
        else:
            logger.warning("Phoenix server not running. Start it with: uv run phoenix serve")
            return False

        # Register the tracer provider (connects to Phoenix's OTLP endpoint)
        _tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint or f"http://{PHOENIX_HOST}:{PHOENIX_GRPC_PORT}",
        )

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
    """Shutdown tracing and flush remaining spans.

    Only shuts down the Phoenix server if we launched it.
    If connecting to an external server, just flushes traces.
    """
    global _tracer_provider, _tracing_enabled, _launched_server

    if not _tracing_enabled:
        return

    try:
        if _tracer_provider and hasattr(_tracer_provider, "force_flush"):
            logger.info("Flushing traces...")
            _tracer_provider.force_flush(timeout_millis=5000)
            logger.info("Traces flushed")
    except Exception as e:
        logger.warning(f"Error flushing traces: {e}")

    # Only log shutdown if we launched the server
    if _launched_server:
        logger.info("Phoenix server will shut down with process")
    else:
        logger.info("Disconnected from Phoenix (server still running)")

    _tracing_enabled = False
    _launched_server = False


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
