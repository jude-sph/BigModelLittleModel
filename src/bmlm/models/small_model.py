"""Small model (executor) for executing plan steps."""

import json
from dataclasses import dataclass
from enum import Enum

from bmlm.models.base import BaseModel, GenerationResult, ModelConfig
from bmlm.orchestrator.plan import Plan, PlanStep


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ExecutionDecision:
    """Decision from the small model about what action to take."""

    action: str
    target_id: str | None
    target_coords: tuple[int, int] | None  # Fallback if ID not found
    input_text: str | None
    direction: str | None
    confidence: Confidence
    reasoning: str
    needs_replanning: bool
    raw_response: str
    generation: GenerationResult


EXECUTOR_SYSTEM_PROMPT = """You are an Android GUI executor agent. Your role is to execute specific steps from a plan.

You will receive:
1. The current plan step to execute
2. The overall goal
3. A list of current UI elements with their IDs

Your job is to:
1. Find the best matching UI element for the current step
2. Decide the exact action to take
3. Report your confidence level

Output JSON:
{
    "action": "tap|type|swipe|scroll|long_press|navigate_home|navigate_back|wait",
    "target_id": "element_id or null",
    "target_coords": [x, y] or null,
    "input_text": "text to type or null",
    "direction": "up|down|left|right or null",
    "confidence": "high|medium|low",
    "reasoning": "Brief explanation",
    "needs_replanning": false
}

Set needs_replanning=true if:
- You cannot find any matching element
- The screen state doesn't match expectations
- You're very uncertain about what to do

Use wait action if you need more time for the screen to load."""


class SmallModel(BaseModel):
    """Executor model that maps plan steps to concrete actions."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.system_prompt = EXECUTOR_SYSTEM_PROMPT

    def generate(
        self,
        current_step: PlanStep,
        plan: Plan,
        ui_elements: list[dict],
        recent_actions: list[dict] | None = None,
    ) -> ExecutionDecision:
        """Decide how to execute the current plan step.

        Args:
            current_step: The step to execute
            plan: The full plan for context
            ui_elements: Current UI elements on screen
            recent_actions: Recent actions taken (for context)

        Returns:
            ExecutionDecision with the concrete action to take
        """
        prompt_parts = [
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>",
            "<|im_start|>user",
            f"Goal: {plan.goal}",
            f"\nCurrent step ({current_step.index + 1}/{len(plan.steps)}):",
            f"  Action: {current_step.action}",
            f"  Target: {current_step.target_description}",
        ]

        if current_step.target_id:
            prompt_parts.append(f"  Expected ID: {current_step.target_id}")
        if current_step.input_text:
            prompt_parts.append(f"  Text to type: {current_step.input_text}")
        if current_step.expected_result:
            prompt_parts.append(f"  Expected result: {current_step.expected_result}")

        # Format UI elements
        elements_str = json.dumps(ui_elements, indent=2)
        prompt_parts.append(f"\nCurrent UI Elements:\n{elements_str}")

        if recent_actions:
            prompt_parts.append(f"\nRecent actions: {json.dumps(recent_actions[-3:])}")

        prompt_parts.append("\nDecide the exact action to execute.<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")

        prompt = "\n".join(prompt_parts)
        result = self._generate(prompt)

        return self._parse_decision(result)

    def _parse_decision(self, result: GenerationResult) -> ExecutionDecision:
        """Parse an ExecutionDecision from the model response."""
        response = result.text
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])

                coords = data.get("target_coords")
                if coords and isinstance(coords, list) and len(coords) == 2:
                    coords = tuple(coords)
                else:
                    coords = None

                confidence_str = data.get("confidence", "medium").lower()
                confidence = Confidence(confidence_str) if confidence_str in ["high", "medium", "low"] else Confidence.MEDIUM

                return ExecutionDecision(
                    action=data.get("action", "wait"),
                    target_id=data.get("target_id"),
                    target_coords=coords,
                    input_text=data.get("input_text"),
                    direction=data.get("direction"),
                    confidence=confidence,
                    reasoning=data.get("reasoning", ""),
                    needs_replanning=data.get("needs_replanning", False),
                    raw_response=response,
                    generation=result,
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: safe wait action
        return ExecutionDecision(
            action="wait",
            target_id=None,
            target_coords=None,
            input_text=None,
            direction=None,
            confidence=Confidence.LOW,
            reasoning="Failed to parse response, waiting for replanning",
            needs_replanning=True,
            raw_response=response,
            generation=result,
        )
