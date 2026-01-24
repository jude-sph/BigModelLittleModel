"""Main orchestrator for coordinating big and small models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

import structlog
from PIL import Image

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
    REPEATED_FAILURE = "repeated_failure"
    MANUAL = "manual"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    confidence_threshold: Confidence = Confidence.MEDIUM
    max_steps_without_replan: int = 10
    max_retries_per_step: int = 3
    max_repeated_actions: int = 2  # Trigger replan after N identical actions
    wait_after_action_ms: int = 500


@dataclass
class StepResult:
    """Result of executing a single step."""

    success: bool
    action_taken: str
    target_index: int | None  # Jeeves element index
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
    # Track repeated actions to detect loops
    last_action_signature: str | None = None
    consecutive_same_action: int = 0
    failure_context: str | None = None  # Passed to big model on replan


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
        self._get_screenshot: Callable[[], Optional[Image.Image]] | None = None

    def set_callbacks(
        self,
        get_ui_elements: Callable[[], list[dict]],
        execute_action: Callable[[dict], bool],
        get_screenshot: Callable[[], Optional[Image.Image]] | None = None,
    ) -> None:
        """Set callbacks for UI interaction.

        Args:
            get_ui_elements: Function that returns current UI elements
            execute_action: Function that executes an action, returns success
            get_screenshot: Function that returns current screenshot (optional)
        """
        self._get_ui_elements = get_ui_elements
        self._execute_action = execute_action
        self._get_screenshot = get_screenshot

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
        screenshot = self._get_screenshot() if self._get_screenshot else None
        result = self.big_model.generate(task=task, ui_elements=ui_elements, screenshot=screenshot)

        self.state.current_plan = result.plan
        self.state.total_replans += 1

        # Check for empty or failed plan
        if not result.plan.steps:
            log.warning(
                "empty_plan_generated",
                raw_response=result.raw_response[:500] if result.raw_response else "none",
                parse_error=getattr(result.plan, "parse_error", False),
            )

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
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Only verify if the plan was expected to complete the task
            if plan.expects_completion:
                verification = self._verify_completion()
                duration_ms = (time.perf_counter() - start_time) * 1000

                if verification.goal_achieved:
                    log.info("goal_verified_complete", reason=verification.reason)
                    return StepResult(
                        success=True,
                        action_taken="goal_complete",
                        target_index=None,
                        duration_ms=duration_ms,
                        triggered_replan=False,
                    )
                else:
                    # Goal not achieved - use new plan from verification
                    log.info("goal_not_complete", reason=verification.reason)
                    if verification.next_plan and verification.next_plan.steps:
                        self.state.current_plan = verification.next_plan
                        self.state.steps_since_replan = 0
                        self.state.total_replans += 1
                        log.info("continuing_with_new_plan", steps=len(verification.next_plan.steps))
                    else:
                        # Verification didn't provide next steps, do full replan
                        self._replan(TriggerReason.PLAN_COMPLETE)
                    return StepResult(
                        success=True,
                        action_taken="verify_replan",
                        target_index=None,
                        duration_ms=duration_ms,
                        triggered_replan=True,
                        trigger_reason=TriggerReason.PLAN_COMPLETE,
                    )
            else:
                # Intermediate plan complete - just get more steps
                log.info("intermediate_plan_complete", requesting_more_steps=True)
                self._replan(TriggerReason.PLAN_COMPLETE)
                return StepResult(
                    success=True,
                    action_taken="continue_planning",
                    target_index=None,
                    duration_ms=duration_ms,
                    triggered_replan=True,
                    trigger_reason=TriggerReason.PLAN_COMPLETE,
                )

        current_step = plan.current_step
        if not current_step:
            plan.mark_complete()
            return StepResult(
                success=True,
                action_taken="plan_complete",
                target_index=None,
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
                target_index=None,
                duration_ms=duration_ms,
                triggered_replan=True,
                trigger_reason=trigger_reason,
            )

        # Execute the action
        action_dict = {
            "action": decision.action,
            "target_index": decision.target_index,
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

        # Track repeated actions to detect loops
        action_signature = f"{decision.action}:{decision.target_index}:{decision.direction}"
        if action_signature == self.state.last_action_signature:
            self.state.consecutive_same_action += 1
        else:
            self.state.last_action_signature = action_signature
            self.state.consecutive_same_action = 1

        # Check for repeated action loop
        if self.state.consecutive_same_action >= self.config.max_repeated_actions:
            failure_context = (
                f"The action '{decision.action}' on element {decision.target_index} "
                f"(direction: {decision.direction}) has been attempted "
                f"{self.state.consecutive_same_action} times without progress. "
                f"This approach is not working. Try a different strategy."
            )
            self.state.failure_context = failure_context
            log.warning(
                "repeated_action_detected",
                action=decision.action,
                target_index=decision.target_index,
                direction=decision.direction,
                count=self.state.consecutive_same_action,
            )
            self._replan(TriggerReason.REPEATED_FAILURE)
            duration_ms = (time.perf_counter() - start_time) * 1000
            return StepResult(
                success=False,
                action_taken=decision.action,
                target_index=decision.target_index,
                duration_ms=duration_ms,
                triggered_replan=True,
                trigger_reason=TriggerReason.REPEATED_FAILURE,
            )

        # Update state
        self.state.steps_since_replan += 1
        self.state.total_steps += 1

        if success:
            plan.advance()

        duration_ms = (time.perf_counter() - start_time) * 1000

        log.info(
            "step_executed",
            action=decision.action,
            target_index=decision.target_index,
            success=success,
            confidence=decision.confidence.value,
            duration_ms=round(duration_ms, 2),
        )

        return StepResult(
            success=success,
            action_taken=decision.action,
            target_index=decision.target_index,
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

    def _verify_completion(self):
        """Verify if the goal has been achieved using the big model."""
        screenshot = self._get_screenshot() if self._get_screenshot else None

        return self.big_model.verify_completion(
            goal=self.state.current_plan.goal,
            screenshot=screenshot,
            previous_actions=self.state.action_history[-10:],
        )

    def _replan(self, reason: TriggerReason) -> None:
        """Generate a new plan from the big model."""
        if not self.state.current_plan:
            return

        ui_elements = self._get_ui_elements() if self._get_ui_elements else []
        screenshot = self._get_screenshot() if self._get_screenshot else None

        result = self.big_model.generate(
            task=self.state.current_plan.goal,
            ui_elements=ui_elements,
            screenshot=screenshot,
            previous_actions=self.state.action_history[-10:],
            failure_context=self.state.failure_context,
        )

        had_failure_context = self.state.failure_context is not None

        self.state.current_plan = result.plan
        self.state.steps_since_replan = 0
        self.state.total_replans += 1
        # Reset failure tracking after replan
        self.state.last_action_signature = None
        self.state.consecutive_same_action = 0
        self.state.failure_context = None

        log.info(
            "replanned",
            reason=reason.value,
            new_steps=len(result.plan.steps),
            generation_ms=result.generation.generation_time_ms,
            had_failure_context=had_failure_context,
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
