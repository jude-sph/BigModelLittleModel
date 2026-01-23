"""Big model (planner) for creating action plans."""

import json
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from bmlm.models.base import GenerationResult, ModelConfig, VisionModel
from bmlm.orchestrator.plan import Plan, PlanStep
from bmlm.tracing import trace_big_model


@dataclass
class PlanningResult:
    """Result from plan generation."""

    plan: Plan
    raw_response: str
    generation: GenerationResult


PLANNING_SYSTEM_PROMPT = """You are an Android GUI automation planner. You can see the screen with numbered UI elements.

Look at the screenshot to understand:
1. What screen am I on? What app?
2. What elements are visible? (Numbers show clickable elements)
3. What sequence of actions reaches the goal?

Output ONLY a JSON plan:
{"goal": "...", "steps": [
  {"action": "tap", "target_index": 5, "target_description": "Settings icon", "expected_result": "Settings opens"},
  ...
], "success_indicator": "..."}

Use target_index numbers from the screenshot. Each step should advance toward the goal.

Action types: tap, type, swipe, scroll, long_press, navigate_home, navigate_back, wait
Plan 3-7 steps. For swipe/scroll, include "direction": "up/down/left/right"."""


class BigModel(VisionModel):
    """Vision-language planner model that creates multi-step action plans."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.system_prompt = PLANNING_SYSTEM_PROMPT

    def generate(
        self,
        task: str,
        ui_elements: list[dict],
        screenshot: Optional[Image.Image] = None,
        previous_actions: list[dict] | None = None,
    ) -> PlanningResult:
        """Generate a plan for completing a task.

        Args:
            task: The task to complete
            ui_elements: List of UI elements with id, text, type, bounds
            screenshot: Screenshot of current screen with Jeeves overlays
            previous_actions: Optional list of actions already taken

        Returns:
            PlanningResult with the generated plan
        """
        with trace_big_model(
            task=task,
            ui_elements_count=len(ui_elements),
            model_path=self.config.model_path,
            previous_actions_count=len(previous_actions) if previous_actions else 0,
        ) as trace_result:
            # Build the prompt for vision model
            prompt_parts = [
                self.system_prompt,
                f"\nTask: {task}",
            ]

            if previous_actions:
                actions_str = json.dumps(previous_actions[-5:], indent=2)
                prompt_parts.append(f"\nRecent actions taken:\n{actions_str}")

            prompt_parts.append("\nLook at the screenshot and create a JSON plan. Output ONLY JSON starting with {")

            prompt = "\n".join(prompt_parts)

            # Generate with vision if screenshot available, else fall back to text
            if screenshot is not None:
                result = self._generate_with_image(prompt, screenshot)
            else:
                # Fallback: include UI elements as text if no screenshot
                prompt_parts.insert(2, f"\nUI Elements:\n{json.dumps(ui_elements, indent=2)}")
                prompt = "\n".join(prompt_parts)
                # Use parent's text generation - need to load as text model
                from mlx_lm import generate, load
                if not self._loaded:
                    self.load()
                import time
                start = time.perf_counter()
                # This is a fallback, won't work well - vision model needs image
                result = GenerationResult(
                    text="{}",
                    tokens_generated=0,
                    generation_time_ms=0,
                    tokens_per_second=0,
                )

            # Parse the plan from response
            plan = self._parse_plan(result.text, task)

            # Record trace data
            trace_result["plan_steps"] = len(plan.steps)
            trace_result["plan_goal"] = plan.goal
            trace_result["generation_time_ms"] = result.generation_time_ms
            trace_result["raw_output"] = result.text
            # Pass step details for readable summary
            trace_result["plan_steps_detail"] = [
                {
                    "action": step.action,
                    "target_description": step.target_description,
                    "target_index": step.target_index,
                }
                for step in plan.steps
            ]

            return PlanningResult(plan=plan, raw_response=result.text, generation=result)

    def _parse_plan(self, response: str, task: str) -> Plan:
        """Parse a Plan from the model's JSON response."""
        try:
            # Find JSON in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                steps = []
                for i, step_data in enumerate(data.get("steps", [])):
                    # Parse target_index as integer
                    target_index = step_data.get("target_index")
                    if target_index is not None:
                        try:
                            target_index = int(target_index)
                        except (ValueError, TypeError):
                            target_index = None

                    steps.append(
                        PlanStep(
                            index=i,
                            action=step_data.get("action", "tap"),
                            target_index=target_index,
                            target_description=step_data.get("target_description", ""),
                            input_text=step_data.get("input_text"),
                            direction=step_data.get("direction"),
                            expected_result=step_data.get("expected_result", ""),
                        )
                    )

                return Plan(
                    goal=data.get("goal", task),
                    steps=steps,
                    success_indicator=data.get("success_indicator", ""),
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: return empty plan if parsing fails
        return Plan(goal=task, steps=[], success_indicator="", parse_error=True)
