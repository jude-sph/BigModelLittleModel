"""Main orchestrator for coordinating big and small models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

import structlog

from bmlm.models.small_model import Confidence
from bmlm.orchestrator.plan import Plan

if TYPE_CHECKING:
    from bmlm.models.big_model import BigModel
    from bmlm.models.small_model import SmallModel

log = structlog.get_logger()


class TriggerReason(Enum):
    """Reasons for calling the big model."""
    TASK_START = "task_start"
    PLAN_COMPLETE = "plan_complete"
    LOW_CONFIDENCE = "low_confidence"
    NEEDS_REPLANNING = "needs_replanning"
    STEP_FAILED = "step_failed"
    MAX_STEPS_REACHED = "max_steps_reached"
    MANUAL = "manual"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    confidence_threshold: Confidence = Confidence.MEDIUM
    max_steps_without_replan: int = 10
    max_retries_per_step: int = 3
    wait_after_action_ms: int = 500


@dataclass
class StepResult:
    """Result of executing a single step."""

    success: bool
    action_taken: str
    target_id: str | None
    duration_ms: float
    triggered_replan: bool = False
    trigger_reason: TriggerReason | None = None


@dataclass
class OrchestratorState:
    """Current state of the orchestrator."""

    current_plan: Plan | None = None
    steps_since_replan: int = 0
    total_steps: int = 0
    total_replans: int = 0
    action_history: list[dict] = field(default_factory=list)


class Orchestrator:
    """Coordinates big and small models for task execution."""

    def __init__(
        self,
        big_model: BigModel,
        small_model: SmallModel,
        config: OrchestratorConfig | None = None,
    ):
        self.big_model = big_model
        self.small_model = small_model
        self.config = config or OrchestratorConfig()
        self.state = OrchestratorState()

        # Callbacks for integration
        self._get_ui_elements: Callable[[], list[dict]] | None = None
        self._execute_action: Callable[[dict], bool] | None = None

    def set_callbacks(
        self,
        get_ui_elements: Callable[[], list[dict]],
        execute_action: Callable[[dict], bool],
    ) -> None:
        """Set callbacks for UI interaction.

        Args:
            get_ui_elements: Function that returns current UI elements
            execute_action: Function that executes an action, returns success
        """
        self._get_ui_elements = get_ui_elements
        self._execute_action = execute_action

    def start_task(self, task: str) -> Plan:
        """Start a new task by generating initial plan.

        Args:
            task: The task description

        Returns:
            The generated plan
        """
        log.info("starting_task", task=task)
        self.state = OrchestratorState()

        ui_elements = self._get_ui_elements() if self._get_ui_elements else []
        result = self.big_model.generate(task=task, ui_elements=ui_elements)

        self.state.current_plan = result.plan
        self.state.total_replans += 1

        log.info(
            "plan_generated",
            goal=result.plan.goal,
            steps=len(result.plan.steps),
            generation_ms=result.generation.generation_time_ms,
        )

        return result.plan

    def step(self) -> StepResult:
        """Execute one step of the current plan.

        Returns:
            StepResult with details of what happened
        """
        if not self.state.current_plan:
            raise RuntimeError("No active plan. Call start_task first.")

        if not self._get_ui_elements or not self._execute_action:
            raise RuntimeError("Callbacks not set. Call set_callbacks first.")

        plan = self.state.current_plan
        start_time = time.perf_counter()

        # Check if plan is complete
        if plan.is_complete:
            return StepResult(
                success=True,
                action_taken="none",
                target_id=None,
                duration_ms=0,
                triggered_replan=False,
            )

        current_step = plan.current_step
        if not current_step:
            plan.mark_complete()
            return StepResult(
                success=True,
                action_taken="plan_complete",
                target_id=None,
                duration_ms=0,
            )

        # Get current UI state
        ui_elements = self._get_ui_elements()

        # Ask small model what to do
        decision = self.small_model.generate(
            current_step=current_step,
            plan=plan,
            ui_elements=ui_elements,
            recent_actions=self.state.action_history[-5:],
        )

        # Check if we need to replan
        trigger_reason = self._check_replan_triggers(decision)

        if trigger_reason:
            log.info("triggering_replan", reason=trigger_reason.value)
            self._replan(trigger_reason)
            duration_ms = (time.perf_counter() - start_time) * 1000
            return StepResult(
                success=True,
                action_taken="replan",
                target_id=None,
                duration_ms=duration_ms,
                triggered_replan=True,
                trigger_reason=trigger_reason,
            )

        # Execute the action
        action_dict = {
            "action": decision.action,
            "target_id": decision.target_id,
            "target_coords": decision.target_coords,
            "input_text": decision.input_text,
            "direction": decision.direction,
        }

        success = self._execute_action(action_dict)

        # Record action
        self.state.action_history.append({
            **action_dict,
            "step_index": current_step.index,
            "success": success,
            "confidence": decision.confidence.value,
        })

        # Update state
        self.state.steps_since_replan += 1
        self.state.total_steps += 1

        if success:
            plan.advance()

        duration_ms = (time.perf_counter() - start_time) * 1000

        log.info(
            "step_executed",
            action=decision.action,
            target=decision.target_id,
            success=success,
            confidence=decision.confidence.value,
            duration_ms=round(duration_ms, 2),
        )

        return StepResult(
            success=success,
            action_taken=decision.action,
            target_id=decision.target_id,
            duration_ms=duration_ms,
        )

    def _check_replan_triggers(self, decision) -> TriggerReason | None:
        """Check if any condition triggers a replan."""
        # Explicit request from small model
        if decision.needs_replanning:
            return TriggerReason.NEEDS_REPLANNING

        # Low confidence
        if decision.confidence == Confidence.LOW:
            return TriggerReason.LOW_CONFIDENCE

        # Too many steps without replan
        if self.state.steps_since_replan >= self.config.max_steps_without_replan:
            return TriggerReason.MAX_STEPS_REACHED

        return None

    def _replan(self, reason: TriggerReason) -> None:
        """Generate a new plan from the big model."""
        if not self.state.current_plan:
            return

        ui_elements = self._get_ui_elements() if self._get_ui_elements else []

        result = self.big_model.generate(
            task=self.state.current_plan.goal,
            ui_elements=ui_elements,
            previous_actions=self.state.action_history[-10:],
        )

        self.state.current_plan = result.plan
        self.state.steps_since_replan = 0
        self.state.total_replans += 1

        log.info(
            "replanned",
            reason=reason.value,
            new_steps=len(result.plan.steps),
            generation_ms=result.generation.generation_time_ms,
        )

    def run_until_complete(self, task: str, max_steps: int = 50) -> bool:
        """Run a task until completion or max steps reached.

        Args:
            task: The task to complete
            max_steps: Maximum number of steps before giving up

        Returns:
            True if task completed successfully
        """
        self.start_task(task)

        for _ in range(max_steps):
            if self.state.current_plan and self.state.current_plan.is_complete:
                log.info(
                    "task_complete",
                    total_steps=self.state.total_steps,
                    total_replans=self.state.total_replans,
                )
                return True

            result = self.step()
            if not result.success and result.action_taken != "replan":
                log.warning("step_failed", result=result)

            # Small delay between actions
            time.sleep(self.config.wait_after_action_ms / 1000)

        log.warning("max_steps_reached", max_steps=max_steps)
        return False

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "total_steps": self.state.total_steps,
            "total_replans": self.state.total_replans,
            "steps_since_replan": self.state.steps_since_replan,
            "plan_progress": self.state.current_plan.progress if self.state.current_plan else 0,
            "actions_taken": len(self.state.action_history),
        }
