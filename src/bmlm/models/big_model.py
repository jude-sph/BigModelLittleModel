"""Big model (planner) for creating action plans."""

import json
from dataclasses import dataclass

from bmlm.models.base import BaseModel, GenerationResult, ModelConfig
from bmlm.orchestrator.plan import Plan, PlanStep
from bmlm.tracing import trace_big_model


@dataclass
class PlanningResult:
    """Result from plan generation."""

    plan: Plan
    raw_response: str
    generation: GenerationResult


PLANNING_SYSTEM_PROMPT = """You are an expert Android GUI automation agent. Create step-by-step plans for completing tasks on Android devices.

You will receive:
1. A task description
2. A list of UI elements with INDEX numbers (matching numbered overlays on screen)

Output ONLY valid JSON. Example:
{"goal": "Open Settings app", "steps": [{"action": "tap", "target_index": 3, "target_description": "Settings icon", "input_text": null, "direction": null, "expected_result": "Settings app opens"}], "success_indicator": "Settings screen is visible"}

Action types (choose ONE per step):
- tap: Tap element at target_index
- type: Type input_text into focused field
- swipe/scroll: Move in direction (up/down/left/right)
- long_press: Long press element at target_index
- navigate_home: Go to home screen
- navigate_back: Press back button
- wait: Wait for screen to load

Keep plans concise (3-7 steps). Set target_index to null if element must be found dynamically."""


class BigModel(BaseModel):
    """Planner model that creates multi-step action plans."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.system_prompt = PLANNING_SYSTEM_PROMPT

    def generate(
        self,
        task: str,
        ui_elements: list[dict],
        screenshot_description: str | None = None,
        previous_actions: list[dict] | None = None,
    ) -> PlanningResult:
        """Generate a plan for completing a task.

        Args:
            task: The task to complete
            ui_elements: List of UI elements with id, text, type, bounds
            screenshot_description: Optional description of the current screen
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
            # Build the prompt
            prompt_parts = [
                f"<|im_start|>system\n{self.system_prompt}<|im_end|>",
                "<|im_start|>user",
                f"Task: {task}",
            ]

            if screenshot_description:
                prompt_parts.append(f"\nCurrent screen: {screenshot_description}")

            # Format UI elements concisely
            elements_str = json.dumps(ui_elements, indent=2)
            prompt_parts.append(f"\nUI Elements:\n{elements_str}")

            if previous_actions:
                actions_str = json.dumps(previous_actions, indent=2)
                prompt_parts.append(f"\nActions already taken:\n{actions_str}")

            prompt_parts.append("\nCreate a plan to complete this task.<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n")

            prompt = "\n".join(prompt_parts)

            # Generate
            result = self._generate(prompt)

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
