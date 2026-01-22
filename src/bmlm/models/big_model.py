"""Big model (planner) for creating action plans."""

import json
from dataclasses import dataclass

from bmlm.models.base import BaseModel, GenerationResult, ModelConfig
from bmlm.orchestrator.plan import Plan, PlanStep


@dataclass
class PlanningResult:
    """Result from plan generation."""

    plan: Plan
    raw_response: str
    generation: GenerationResult


PLANNING_SYSTEM_PROMPT = """You are an expert Android GUI automation agent. Your role is to create step-by-step plans for completing tasks on Android devices.

You will receive:
1. A task description
2. A screenshot of the current screen (if available)
3. A list of UI elements with their IDs and properties

Output a JSON plan with this structure:
{
    "goal": "Brief description of what we're trying to achieve",
    "steps": [
        {
            "action": "tap|type|swipe|scroll|long_press|navigate_home|navigate_back|wait",
            "target_id": "element_id or null if needs to be found",
            "target_description": "Human-readable description of the target",
            "input_text": "Text to type (only for type action)",
            "direction": "up|down|left|right (only for swipe/scroll)",
            "expected_result": "What should happen after this action"
        }
    ],
    "success_indicator": "How to know the task is complete"
}

Be specific but flexible. If an exact element ID isn't visible, describe what to look for.
Keep plans concise - typically 3-7 steps."""


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
                    steps.append(
                        PlanStep(
                            index=i,
                            action=step_data.get("action", "tap"),
                            target_id=step_data.get("target_id"),
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
