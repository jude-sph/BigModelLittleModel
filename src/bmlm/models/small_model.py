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


EXECUTOR_SYSTEM_PROMPT = """You execute plan steps. The plan specifies the action - you find the target element.

COPY THE ACTION AND DIRECTION FROM THE PLAN STEP. Do not change them.
- If the plan says "swipe" with direction "right", output action="swipe" and direction="right"
- If the plan says "tap", output action="tap"

Your only job: find the UI element that matches the plan's target description.

For actions that don't need an element (open_app, navigate_home, navigate_back, wait):
- Set target_index to null
- Copy input_text from plan if present (e.g., app name for open_app)

Output JSON with the SAME action and direction as the plan step:
{"action": "<copy from plan>", "target_index": <element number or null>, "direction": "<copy from plan if specified>", "input_text": "<copy from plan if specified>", "confidence": "high/medium/low", "reasoning": "<why this element>", "needs_replanning": false}

If no element matches (for actions that need one), set needs_replanning=true."""


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
            ui_elements=ui_elements,
        ) as trace_result:
            prompt_parts = [
                f"<|im_start|>system\n{self.system_prompt}<|im_end|>",
                "<|im_start|>user",
                f"Goal: {plan.goal}",
                f"\nPlan step {current_step.index + 1}/{len(plan.steps)}:",
                f"  ACTION TO EXECUTE: {current_step.action}",
            ]

            # Explicitly show direction if specified (for swipe/scroll)
            if current_step.direction:
                prompt_parts.append(f"  DIRECTION: {current_step.direction} (use this exactly)")

            prompt_parts.append(f"  TARGET TO FIND: {current_step.target_description}")

            if current_step.target_index is not None:
                prompt_parts.append(f"  Expected element index: {current_step.target_index}")
            if current_step.input_text:
                if current_step.action == "open_app":
                    prompt_parts.append(f"  APP NAME: {current_step.input_text} (use this in input_text)")
                else:
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

            # Build final instruction based on action type
            no_element_actions = ("open_app", "navigate_home", "navigate_back", "wait")
            if current_step.action in no_element_actions:
                if current_step.action == "open_app" and current_step.input_text:
                    prompt_parts.append(f"\nOutput action=\"{current_step.action}\" with input_text=\"{current_step.input_text}\" and target_index=null.<|im_end|>")
                else:
                    prompt_parts.append(f"\nOutput action=\"{current_step.action}\" with target_index=null.<|im_end|>")
            elif current_step.direction:
                prompt_parts.append(f"\nOutput action=\"{current_step.action}\" with direction=\"{current_step.direction}\" and find the matching element.<|im_end|>")
            else:
                prompt_parts.append(f"\nOutput action=\"{current_step.action}\" and find the matching element.<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n{")

            prompt = "\n".join(prompt_parts)
            result = self._generate(prompt)

            decision = self._parse_decision(result)

            # Record trace data
            trace_result["action"] = decision.action
            trace_result["target_index"] = decision.target_index
            trace_result["direction"] = decision.direction
            trace_result["input_text"] = decision.input_text
            trace_result["confidence"] = decision.confidence.value
            trace_result["needs_replanning"] = decision.needs_replanning
            trace_result["reasoning"] = decision.reasoning
            trace_result["generation_time_ms"] = result.generation_time_ms
            trace_result["raw_output"] = result.text

            return decision

    def _parse_decision(self, result: GenerationResult) -> ExecutionDecision:
        """Parse an ExecutionDecision from the model response."""
        # We prepend { in the prompt, so add it back for parsing
        response = "{" + result.text

        # Strip markdown code blocks if present
        if "```json" in response:
            response = "{" + response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = "{" + response.split("```")[1].split("```")[0]

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
