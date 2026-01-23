"""Small model (executor) for executing plan steps."""

import json
from dataclasses import dataclass
from enum import Enum

from bmlm.models.base import BaseModel, GenerationResult, ModelConfig
from bmlm.orchestrator.plan import Plan, PlanStep
from bmlm.tracing import trace_small_model


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ExecutionDecision:
    """Decision from the small model about what action to take."""

    action: str
    target_index: int | None  # Jeeves element index (matches overlay numbers)
    input_text: str | None
    direction: str | None
    confidence: Confidence
    reasoning: str
    needs_replanning: bool
    raw_response: str
    generation: GenerationResult


EXECUTOR_SYSTEM_PROMPT = """You are an Android GUI executor agent. Your role is to execute specific steps from a plan.

CRITICAL: Follow the plan step closely. Use the action type and find an element matching the target description.
- The element's label must SEMANTICALLY match what you're controlling
- "open calculator" → find "Calculator" (NOT Calendar, NOT Clock)
- "send message" → find "Send" button (NOT Delete, NOT Cancel)
- If no element matches the plan's target, set needs_replanning=true

You will receive:
1. The current plan step to execute
2. The overall goal
3. A list of UI elements with their INDEX numbers

Your job is to:
1. READ the step carefully - what are you trying to control?
2. FIND an element whose label/description matches that function
3. If no matching element exists, set needs_replanning=true

Output ONLY valid JSON. Example:
{"action": "tap", "target_index": 5, "input_text": null, "direction": null, "confidence": "high", "reasoning": "Tapping Settings button", "needs_replanning": false}

Action types:
- tap: Tap element at target_index
- type: Type input_text into focused field
- swipe/scroll: Swipe in direction (up/down/left/right)
- long_press: Long press element at target_index
- navigate_home: Go to home screen
- navigate_back: Press back button
- wait: Wait for screen to load

IMPORTANT: Do NOT pick a random element. If the step says "adjust brightness" and you only see WiFi/Bluetooth elements, you MUST set needs_replanning=true."""


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
            ui_elements: Current UI elements with Jeeves indices
            recent_actions: Recent actions taken (for context)

        Returns:
            ExecutionDecision with the concrete action to take
        """
        with trace_small_model(
            current_step=current_step.target_description or current_step.action,
            step_index=current_step.index,
            ui_elements_count=len(ui_elements),
            model_path=self.config.model_path,
            recent_actions_count=len(recent_actions) if recent_actions else 0,
        ) as trace_result:
            prompt_parts = [
                f"<|im_start|>system\n{self.system_prompt}<|im_end|>",
                "<|im_start|>user",
                f"Goal: {plan.goal}",
                f"\nCurrent step ({current_step.index + 1}/{len(plan.steps)}):",
                f"  Action: {current_step.action}",
                f"  Target: {current_step.target_description}",
            ]

            if current_step.target_index is not None:
                prompt_parts.append(f"  Expected element index: {current_step.target_index}")
            if current_step.input_text:
                prompt_parts.append(f"  Text to type: {current_step.input_text}")
            if current_step.expected_result:
                prompt_parts.append(f"  Expected result: {current_step.expected_result}")

            # Format UI elements with indices prominently
            prompt_parts.append("\nUI Elements (index: description):")
            for elem in ui_elements:
                idx = elem.get("index", "?")
                text = elem.get("text", "")
                desc = elem.get("content_desc", "")
                elem_type = elem.get("type", "Unknown")
                label = text or desc or elem_type
                prompt_parts.append(f"  [{idx}] {label} ({elem_type})")

            if recent_actions:
                prompt_parts.append(f"\nRecent actions: {json.dumps(recent_actions[-3:])}")

            prompt_parts.append("\nFind the element that matches the step's intent. If no element logically matches, set needs_replanning=true.<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n")

            prompt = "\n".join(prompt_parts)
            result = self._generate(prompt)

            decision = self._parse_decision(result)

            # Record trace data
            trace_result["action"] = decision.action
            trace_result["target_index"] = decision.target_index
            trace_result["confidence"] = decision.confidence.value
            trace_result["needs_replanning"] = decision.needs_replanning
            trace_result["reasoning"] = decision.reasoning
            trace_result["generation_time_ms"] = result.generation_time_ms
            trace_result["raw_output"] = result.text

            return decision

    def _parse_decision(self, result: GenerationResult) -> ExecutionDecision:
        """Parse an ExecutionDecision from the model response."""
        response = result.text
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])

                # Parse target_index as integer
                target_index = data.get("target_index")
                if target_index is not None:
                    try:
                        target_index = int(target_index)
                    except (ValueError, TypeError):
                        target_index = None

                confidence_str = data.get("confidence", "medium").lower()
                confidence = Confidence(confidence_str) if confidence_str in ["high", "medium", "low"] else Confidence.MEDIUM

                return ExecutionDecision(
                    action=data.get("action", "wait"),
                    target_index=target_index,
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
            target_index=None,
            input_text=None,
            direction=None,
            confidence=Confidence.LOW,
            reasoning="Failed to parse response, waiting for replanning",
            needs_replanning=True,
            raw_response=response,
            generation=result,
        )
