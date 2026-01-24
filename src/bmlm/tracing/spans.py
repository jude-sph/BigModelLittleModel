"""Span helpers for tracing different components."""

from contextlib import contextmanager
from typing import Any, Generator, Optional

from bmlm.tracing.setup import get_tracer, is_tracing_enabled

# Task-level metadata that gets attached to all spans
_current_task_metadata: dict[str, Any] = {}


def set_task_metadata(
    task_name: str,
    goal: str,
    **extra: Any,
) -> None:
    """Set metadata for the current task (attached to all subsequent spans).

    Args:
        task_name: Name of the task being run
        goal: Task goal description
        **extra: Additional metadata
    """
    global _current_task_metadata
    _current_task_metadata = {
        "task.name": task_name,
        "task.goal": goal,
        **extra,
    }


def _add_task_metadata(span) -> None:
    """Add current task metadata to a span."""
    for key, value in _current_task_metadata.items():
        if value is not None:
            span.set_attribute(key, str(value) if not isinstance(value, (int, float, bool)) else value)


@contextmanager
def trace_big_model(
    task: str,
    ui_elements_count: int,
    model_path: str,
    previous_actions_count: int = 0,
) -> Generator[dict, None, None]:
    """Trace a big model (planner) generation call.

    Args:
        task: The task/goal being planned
        ui_elements_count: Number of UI elements in context
        model_path: Model being used
        previous_actions_count: Number of previous actions provided

    Yields:
        Dict to populate with results (plan_steps, generation_time_ms, etc.)
    """
    if not is_tracing_enabled():
        yield {}
        return

    tracer = get_tracer("bmlm.big_model")
    result: dict[str, Any] = {}

    with tracer.start_as_current_span("big_model.generate") as span:
        _add_task_metadata(span)
        span.set_attribute("llm.model", model_path)
        span.set_attribute("llm.role", "planner")
        span.set_attribute("input.task", task)
        span.set_attribute("input.ui_elements_count", ui_elements_count)
        span.set_attribute("input.previous_actions_count", previous_actions_count)

        yield result

        # Set readable output attributes
        if "plan_steps" in result:
            span.set_attribute("output.plan_steps", result["plan_steps"])
        if "plan_goal" in result:
            span.set_attribute("output.plan_goal", result["plan_goal"])
        if "expects_completion" in result:
            span.set_attribute("output.expects_completion", result["expects_completion"])
        if "generation_time_ms" in result:
            span.set_attribute("metrics.generation_time_ms", result["generation_time_ms"])
        if "tokens" in result:
            span.set_attribute("metrics.tokens", result["tokens"])

        # Create human-readable plan summary
        if "plan_steps_detail" in result:
            steps = result["plan_steps_detail"]
            expects = result.get("expects_completion", False)
            plan_type = "FINAL" if expects else "INTERMEDIATE"
            current_screen = result.get("current_screen", "")
            summary_lines = [f"Goal: {result.get('plan_goal', '?')} [{plan_type}]"]
            if current_screen:
                summary_lines.append(f"Screen: {current_screen}")
            for i, step in enumerate(steps[:10], 1):  # Max 10 steps
                action = step.get("action", "?")
                direction = step.get("direction")
                target = step.get("target_description", step.get("target_index", "?"))
                # Include direction for swipe/scroll actions
                if direction and action in ("swipe", "scroll"):
                    summary_lines.append(f"  {i}. {action} {direction} → {target}")
                else:
                    summary_lines.append(f"  {i}. {action} → {target}")
            span.set_attribute("output.plan_summary", "\n".join(summary_lines))

        # Store raw for debugging
        if "raw_output" in result:
            span.set_attribute("debug.raw_response", result["raw_output"][:1500])


@contextmanager
def trace_verification(
    goal: str,
    model_path: str,
) -> Generator[dict, None, None]:
    """Trace a goal verification call.

    Args:
        goal: The goal being verified
        model_path: Model being used

    Yields:
        Dict to populate with results (goal_achieved, reason, etc.)
    """
    if not is_tracing_enabled():
        yield {}
        return

    tracer = get_tracer("bmlm.big_model")
    result: dict[str, Any] = {}

    with tracer.start_as_current_span("big_model.verify") as span:
        _add_task_metadata(span)
        span.set_attribute("llm.model", model_path)
        span.set_attribute("llm.role", "verifier")
        span.set_attribute("input.goal", goal)

        yield result

        if "goal_achieved" in result:
            span.set_attribute("output.goal_achieved", result["goal_achieved"])
        if "reason" in result:
            span.set_attribute("output.reason", result["reason"])
        if "next_steps" in result:
            span.set_attribute("output.next_steps", result["next_steps"])
        if "generation_time_ms" in result:
            span.set_attribute("metrics.generation_time_ms", result["generation_time_ms"])
        if "raw_output" in result:
            span.set_attribute("debug.raw_response", result["raw_output"][:1500])

        # Summary
        achieved = "✓ ACHIEVED" if result.get("goal_achieved") else "✗ NOT ACHIEVED"
        reason = result.get("reason", "")
        span.set_attribute("output.summary", f"{achieved}: {reason}")


@contextmanager
def trace_small_model(
    current_step: str,
    step_index: int,
    ui_elements_count: int,
    model_path: str,
    recent_actions_count: int = 0,
    ui_elements: Optional[list[dict]] = None,
) -> Generator[dict, None, None]:
    """Trace a small model (executor) generation call.

    Args:
        current_step: The current plan step being executed
        step_index: Index of current step in plan
        ui_elements_count: Number of UI elements in context
        model_path: Model being used
        recent_actions_count: Number of recent actions in context
        ui_elements: List of UI elements visible to the model

    Yields:
        Dict to populate with results (action, confidence, etc.)
    """
    if not is_tracing_enabled():
        yield {}
        return

    tracer = get_tracer("bmlm.small_model")
    result: dict[str, Any] = {}

    with tracer.start_as_current_span("small_model.generate") as span:
        _add_task_metadata(span)
        span.set_attribute("llm.model", model_path)
        span.set_attribute("llm.role", "executor")
        span.set_attribute("input.current_step", current_step)
        span.set_attribute("input.step_index", step_index)
        span.set_attribute("input.ui_elements_count", ui_elements_count)
        span.set_attribute("input.recent_actions_count", recent_actions_count)

        # Format UI elements for readable display
        if ui_elements:
            elements_summary = []
            for elem in ui_elements[:20]:  # Limit to 20 elements
                idx = elem.get("index", "?")
                text = elem.get("text", "")
                desc = elem.get("content_desc", "")
                elem_type = elem.get("type", "")
                label = text or desc or elem_type or "unknown"
                elements_summary.append(f"[{idx}] {label}")
            span.set_attribute("input.ui_elements", "\n".join(elements_summary))

        yield result

        # Set readable output attributes
        if "action" in result:
            span.set_attribute("output.action", result["action"])
        if "target_index" in result:
            idx = result["target_index"]
            span.set_attribute("output.target_index", idx if idx is not None else -1)
        if "confidence" in result:
            span.set_attribute("output.confidence", result["confidence"])
        if "needs_replanning" in result:
            span.set_attribute("output.needs_replanning", result["needs_replanning"])
        if "reasoning" in result:
            span.set_attribute("output.reasoning", result["reasoning"])
        if "direction" in result and result["direction"]:
            span.set_attribute("output.direction", result["direction"])
        if "input_text" in result and result["input_text"]:
            span.set_attribute("output.input_text", result["input_text"])
        if "generation_time_ms" in result:
            span.set_attribute("metrics.generation_time_ms", result["generation_time_ms"])

        # Create human-readable summary based on action type
        action = result.get("action", "?")
        target = result.get("target_index")
        direction = result.get("direction")
        input_text = result.get("input_text")
        conf = result.get("confidence", "?")
        reasoning = result.get("reasoning", "")

        # Build action-specific summary
        if action in ("tap", "long_press"):
            summary = f"{action} → element [{target}]"
        elif action in ("swipe", "scroll"):
            summary = f"{action} {direction or '?'}"
            if target is not None:
                summary += f" (from element [{target}])"
        elif action == "type":
            text_preview = (input_text[:30] + "...") if input_text and len(input_text) > 30 else input_text
            summary = f"type \"{text_preview or ''}\""
        elif action in ("navigate_home", "navigate_back", "wait"):
            summary = action
        else:
            summary = f"{action} → element [{target}]"

        summary += f" ({conf} confidence)"
        if reasoning:
            summary += f"\nReason: {reasoning}"
        span.set_attribute("output.summary", summary)

        # Store raw for debugging (collapsed in UI)
        if "raw_output" in result:
            span.set_attribute("debug.raw_response", result["raw_output"][:1500])


@contextmanager
def trace_action(
    action_type: str,
    target_id: Optional[str] = None,
    coords: Optional[tuple[int, int]] = None,
) -> Generator[dict, None, None]:
    """Trace an action execution on the Android device.

    Args:
        action_type: Type of action (tap, swipe, type, etc.)
        target_id: Target element ID if applicable
        coords: Target coordinates if applicable

    Yields:
        Dict to populate with results (success, duration_ms, etc.)
    """
    if not is_tracing_enabled():
        yield {}
        return

    tracer = get_tracer("bmlm.actions")
    result: dict[str, Any] = {}

    with tracer.start_as_current_span(f"action.{action_type}") as span:
        _add_task_metadata(span)
        span.set_attribute("action.type", action_type)
        if target_id:
            span.set_attribute("action.target_id", target_id)
        if coords:
            span.set_attribute("action.coords", f"{coords[0]},{coords[1]}")

        yield result

        if "success" in result:
            span.set_attribute("action.success", result["success"])
        if "duration_ms" in result:
            span.set_attribute("metrics.duration_ms", result["duration_ms"])
        if "error" in result:
            span.set_attribute("action.error", result["error"])


@contextmanager
def trace_task(
    task_name: str,
    goal: str,
    max_steps: int,
) -> Generator[dict, None, None]:
    """Trace an entire task execution.

    Args:
        task_name: Name of the AndroidWorld task
        goal: Task goal description
        max_steps: Maximum steps allowed

    Yields:
        Dict to populate with results (success, steps, score, etc.)
    """
    if not is_tracing_enabled():
        yield {}
        return

    # Set task metadata for child spans
    set_task_metadata(task_name, goal, max_steps=max_steps)

    tracer = get_tracer("bmlm.benchmark")
    result: dict[str, Any] = {}

    with tracer.start_as_current_span(f"task.{task_name}") as span:
        span.set_attribute("task.name", task_name)
        span.set_attribute("task.goal", goal)
        span.set_attribute("task.max_steps", max_steps)

        yield result

        if "success" in result:
            span.set_attribute("task.success", result["success"])
        if "score" in result:
            span.set_attribute("task.score", result["score"])
        if "steps" in result:
            span.set_attribute("task.steps_taken", result["steps"])
        if "replans" in result:
            span.set_attribute("task.replans", result["replans"])
        if "elapsed_s" in result:
            span.set_attribute("metrics.elapsed_s", result["elapsed_s"])
        if "error" in result:
            span.set_attribute("task.error", result["error"])
        if "termination_reason" in result:
            span.set_attribute("task.termination_reason", result["termination_reason"])


@contextmanager
def trace_orchestrator_step(
    step_number: int,
    plan_step: str,
    steps_since_replan: int,
) -> Generator[dict, None, None]:
    """Trace an orchestrator step.

    Args:
        step_number: Overall step number
        plan_step: Current plan step description
        steps_since_replan: Steps since last replan

    Yields:
        Dict to populate with results
    """
    if not is_tracing_enabled():
        yield {}
        return

    tracer = get_tracer("bmlm.orchestrator")
    result: dict[str, Any] = {}

    with tracer.start_as_current_span("orchestrator.step") as span:
        _add_task_metadata(span)
        span.set_attribute("orchestrator.step_number", step_number)
        span.set_attribute("orchestrator.plan_step", plan_step)
        span.set_attribute("orchestrator.steps_since_replan", steps_since_replan)

        yield result

        if "triggered_replan" in result:
            span.set_attribute("orchestrator.triggered_replan", result["triggered_replan"])
        if "trigger_reason" in result:
            span.set_attribute("orchestrator.trigger_reason", result["trigger_reason"])
        if "action_success" in result:
            span.set_attribute("orchestrator.action_success", result["action_success"])
