"""Big model (planner) for creating action plans."""

import json
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from bmlm.models.base import GenerationResult, ModelConfig, VisionModel
from bmlm.orchestrator.plan import Plan, PlanStep
from bmlm.tracing import trace_big_model, trace_verification


@dataclass
class PlanningResult:
    """Result from plan generation."""

    plan: Plan
    raw_response: str
    generation: GenerationResult


@dataclass
class VerificationResult:
    """Result from goal verification."""

    goal_achieved: bool
    reason: str
    next_plan: Optional[Plan]  # New plan if goal not achieved
    raw_response: str
    generation: GenerationResult


PLANNING_SYSTEM_PROMPT_TEMPLATE = """You are an Android GUI planner. Create a plan based on what you can see NOW.

STEP 1: Look at the UI elements list below. What screen is this?
STEP 2: Search the list - is your TARGET element there? (e.g., "brightness", "slider", etc.)
STEP 3: If target is NOT in list, plan to navigate. If target IS in list, plan to interact with it.

RULES:
- You MUST include "current_screen" in your output
- CRITICAL: Search the UI elements list for your target. If it's NOT there, navigate first!
- Output 1-{max_steps} steps
- For swipe/scroll: specify direction (up/down/left/right)
- SLIDER DIRECTIONS: To INCREASE/MAXIMIZE a horizontal slider, swipe RIGHT. To DECREASE/MINIMIZE, swipe LEFT.
- expects_completion: Set TRUE when your planned steps will achieve the goal. Set FALSE only when you need to navigate somewhere first before the goal can be achieved.
- Output raw JSON only

Example (home screen - "brightness" NOT in element list):
{{"current_screen": "home screen with app icons", "goal": "Set brightness max", "expects_completion": false, "steps": [{{"action": "swipe", "direction": "down", "target_index": null, "target_description": "top of screen", "expected_result": "Quick settings opens"}}], "success_indicator": "Brightness slider in list"}}

Example (compact quick settings - "brightness" still NOT in list, need to expand):
{{"current_screen": "compact quick settings", "goal": "Set brightness max", "expects_completion": false, "steps": [{{"action": "swipe", "direction": "down", "target_index": null, "target_description": "quick settings panel", "expected_result": "Panel expands"}}], "success_indicator": "Brightness slider in list"}}

Example (expanded quick settings - "brightness" IS in list at index 8, swipe RIGHT to maximize):
{{"current_screen": "expanded quick settings with brightness slider", "goal": "Set brightness max", "expects_completion": true, "steps": [{{"action": "swipe", "direction": "right", "target_index": 8, "target_description": "brightness slider", "expected_result": "Slider moved to maximum"}}], "success_indicator": "Slider at right edge"}}

Actions: tap, type, swipe, scroll, long_press, navigate_home, navigate_back, wait, open_app

For open_app: set target_index to null and put app name in input_text field.
Example: {{"action": "open_app", "target_index": null, "input_text": "Settings", "target_description": "Settings app", "expected_result": "Settings app opens"}}"""


VERIFICATION_PROMPT_TEMPLATE = """Look at the screenshot. Has this goal been achieved: "{goal}"?

Output JSON only:
{{"goal_achieved": true/false, "reason": "brief explanation", "next_steps": []}}

If goal_achieved is false, include 1-{max_steps} next_steps to continue.
Example next_steps: [{{"action": "tap", "target_index": 5, "target_description": "Save button", "expected_result": "Settings saved"}}]"""


class BigModel(VisionModel):
    """Vision-language planner model that creates multi-step action plans."""

    def __init__(self, config: ModelConfig, max_plan_steps: int = 3):
        super().__init__(config)
        self.max_plan_steps = max_plan_steps
        self.system_prompt = PLANNING_SYSTEM_PROMPT_TEMPLATE.format(max_steps=max_plan_steps)

    def generate(
        self,
        task: str,
        ui_elements: list[dict],
        screenshot: Optional[Image.Image] = None,
        previous_actions: list[dict] | None = None,
        failure_context: str | None = None,
    ) -> PlanningResult:
        """Generate a plan for completing a task.

        Args:
            task: The task to complete
            ui_elements: List of UI elements with id, text, type, bounds
            screenshot: Screenshot of current screen with Jeeves overlays
            previous_actions: Optional list of actions already taken
            failure_context: Optional context about why the previous approach failed

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

            # Include failure context prominently if present
            if failure_context:
                prompt_parts.append(f"\n⚠️ IMPORTANT - Previous approach failed:\n{failure_context}")

            if previous_actions:
                actions_str = json.dumps(previous_actions[-5:], indent=2)
                prompt_parts.append(f"\nRecent actions taken:\n{actions_str}")

            # Include UI elements so model knows exactly what's available
            if ui_elements:
                elements_summary = []
                for elem in ui_elements[:25]:  # Limit to avoid prompt bloat
                    idx = elem.get("index", "?")
                    text = elem.get("text", "")
                    desc = elem.get("content_desc", "")
                    elem_type = elem.get("type", "")
                    label = text or desc or elem_type or "unknown"
                    elements_summary.append(f"  [{idx}] {label}")
                prompt_parts.append(f"\n=== VISIBLE UI ELEMENTS (this is what's on screen NOW) ===\n" + "\n".join(elements_summary))
                prompt_parts.append("\n=== END OF VISIBLE ELEMENTS ===")
            else:
                prompt_parts.append("\n=== NO UI ELEMENTS DETECTED ===")

            prompt_parts.append("\nCheck the elements list above. Is your target there? If NOT, navigate first. Output JSON starting with {")

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
            plan, current_screen = self._parse_plan(result.text, task)

            # Record trace data
            trace_result["plan_steps"] = len(plan.steps)
            trace_result["plan_goal"] = plan.goal
            trace_result["expects_completion"] = plan.expects_completion
            trace_result["generation_time_ms"] = result.generation_time_ms
            trace_result["raw_output"] = result.text
            if current_screen:
                trace_result["current_screen"] = current_screen
            if failure_context:
                trace_result["failure_context"] = failure_context
            # Pass step details for readable summary
            trace_result["plan_steps_detail"] = [
                {
                    "action": step.action,
                    "target_description": step.target_description,
                    "target_index": step.target_index,
                    "direction": step.direction,
                }
                for step in plan.steps
            ]

            return PlanningResult(plan=plan, raw_response=result.text, generation=result)

    def verify_completion(
        self,
        goal: str,
        screenshot: Optional[Image.Image] = None,
        previous_actions: list[dict] | None = None,
    ) -> VerificationResult:
        """Verify if the goal has been achieved and get next steps if not.

        Args:
            goal: The goal to verify
            screenshot: Current screenshot
            previous_actions: Actions taken so far

        Returns:
            VerificationResult with achievement status and optional new plan
        """
        with trace_verification(goal=goal, model_path=self.config.model_path) as trace_result:
            prompt = VERIFICATION_PROMPT_TEMPLATE.format(goal=goal, max_steps=self.max_plan_steps)

            if previous_actions:
                actions_summary = ", ".join(a.get("action", "?") for a in previous_actions[-5:])
                prompt += f"\n\nRecent actions: {actions_summary}"

            if screenshot is not None:
                result = self._generate_with_image(prompt, screenshot)
            else:
                result = GenerationResult(
                    text='{"goal_achieved": false, "reason": "No screenshot", "next_steps": []}',
                    tokens_generated=0,
                    generation_time_ms=0,
                    tokens_per_second=0,
                )

            # Parse verification response
            goal_achieved = False
            reason = ""
            next_plan = None

            try:
                # Strip markdown if present
                response = result.text
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]

                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    data = json.loads(response[json_start:json_end])
                    goal_achieved = data.get("goal_achieved", False)
                    reason = data.get("reason", "")

                    # Parse next steps if goal not achieved
                    if not goal_achieved and "next_steps" in data:
                        steps = []
                        for i, step_data in enumerate(data["next_steps"]):
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
                        if steps:
                            next_plan = Plan(goal=goal, steps=steps, success_indicator="")
            except (json.JSONDecodeError, KeyError, TypeError):
                reason = "Failed to parse verification response"

            # Record trace data
            trace_result["goal_achieved"] = goal_achieved
            trace_result["reason"] = reason
            trace_result["next_steps"] = len(next_plan.steps) if next_plan else 0
            trace_result["generation_time_ms"] = result.generation_time_ms
            trace_result["raw_output"] = result.text

            return VerificationResult(
                goal_achieved=goal_achieved,
                reason=reason,
                next_plan=next_plan,
                raw_response=result.text,
                generation=result,
            )

    def _parse_plan(self, response: str, task: str) -> tuple[Plan, str | None]:
        """Parse a Plan from the model's JSON response.

        Returns:
            Tuple of (Plan, current_screen description or None)
        """
        current_screen = None
        try:
            # Strip markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            # Find JSON in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                # Extract current_screen if present
                current_screen = data.get("current_screen")

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
                    expects_completion=data.get("expects_completion", False),
                ), current_screen
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: return empty plan if parsing fails
        return Plan(goal=task, steps=[], success_indicator="", parse_error=True), None
