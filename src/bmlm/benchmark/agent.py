"""AndroidWorld agent implementation using BMLM architecture."""

from dataclasses import dataclass
from typing import Any

import structlog

from android_world.agents import base_agent

from bmlm.android.actions import ActionExecutor
from bmlm.models.base import ModelConfig
from bmlm.models.big_model import BigModel
from bmlm.models.small_model import SmallModel
from bmlm.orchestrator.controller import Orchestrator, OrchestratorConfig

log = structlog.get_logger()


@dataclass
class BMLMAgentConfig:
    """Configuration for the BMLM agent."""

    # Model paths (MLX format)
    big_model_path: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    small_model_path: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"

    # Generation settings
    big_model_max_tokens: int = 1024
    small_model_max_tokens: int = 256
    temperature: float = 0.7

    # Orchestrator settings
    max_steps_without_replan: int = 10
    wait_after_action_ms: int = 500


class BMLMAgent(base_agent.EnvironmentInteractingAgent):
    """BMLM agent that integrates with AndroidWorld benchmark.

    This agent uses a hierarchical architecture with a big model (planner)
    and small model (executor) coordinated by an orchestrator.
    """

    def __init__(self, env: Any, config: BMLMAgentConfig | None = None):
        """Initialize the BMLM agent.

        Args:
            env: AndroidWorld environment instance
            config: Agent configuration
        """
        super().__init__(env)
        self.config = config or BMLMAgentConfig()

        # Initialize models
        big_model_config = ModelConfig(
            model_path=self.config.big_model_path,
            max_tokens=self.config.big_model_max_tokens,
            temperature=self.config.temperature,
        )
        small_model_config = ModelConfig(
            model_path=self.config.small_model_path,
            max_tokens=self.config.small_model_max_tokens,
            temperature=self.config.temperature,
        )

        self.big_model = BigModel(big_model_config)
        self.small_model = SmallModel(small_model_config)

        # Initialize orchestrator
        orch_config = OrchestratorConfig(
            max_steps_without_replan=self.config.max_steps_without_replan,
            wait_after_action_ms=self.config.wait_after_action_ms,
        )
        self.orchestrator = Orchestrator(
            big_model=self.big_model,
            small_model=self.small_model,
            config=orch_config,
        )

        # Initialize action executor
        self.action_executor = ActionExecutor(env)

        # Wire up callbacks
        self.orchestrator.set_callbacks(
            get_ui_elements=self.action_executor.get_ui_elements,
            execute_action=self.action_executor.execute,
        )

        self._task_started = False
        self._current_task: str = ""

    def set_task(self, task: str) -> None:
        """Set the current task to execute.

        Args:
            task: Task description
        """
        self._current_task = task
        self._task_started = False
        log.info("task_set", task=task)

    def step(self) -> base_agent.AgentInteractionResult:
        """Execute one step of the agent.

        This is called by AndroidWorld in a loop until the task is complete
        or max steps are reached.

        Returns:
            AgentInteractionResult with action taken and completion status
        """
        # Start task on first step
        if not self._task_started:
            if not self._current_task:
                return base_agent.AgentInteractionResult(
                    done=True,
                    output="No task set",
                )
            self.orchestrator.start_task(self._current_task)
            self._task_started = True

        # Check if plan is complete
        if self.orchestrator.state.current_plan and self.orchestrator.state.current_plan.is_complete:
            stats = self.orchestrator.get_stats()
            return base_agent.AgentInteractionResult(
                done=True,
                output=f"Task completed. Steps: {stats['total_steps']}, Replans: {stats['total_replans']}",
            )

        # Execute one step
        result = self.orchestrator.step()

        # Build output message
        output = f"Action: {result.action_taken}"
        if result.target_id:
            output += f" on {result.target_id}"
        if result.triggered_replan:
            output += f" (triggered replan: {result.trigger_reason.value if result.trigger_reason else 'unknown'})"

        return base_agent.AgentInteractionResult(
            done=False,
            output=output,
        )

    def reset(self) -> None:
        """Reset the agent state."""
        self._task_started = False
        self._current_task = ""
        self.orchestrator.state = type(self.orchestrator.state)()
        log.info("agent_reset")

    def load_models(self) -> None:
        """Pre-load both models into memory."""
        log.info("loading_models")
        self.big_model.load()
        self.small_model.load()
        log.info("models_loaded")

    def unload_models(self) -> None:
        """Unload models to free memory."""
        self.big_model.unload()
        self.small_model.unload()
        log.info("models_unloaded")


def create_agent(env: Any, **kwargs: Any) -> BMLMAgent:
    """Factory function to create a BMLM agent.

    This function signature matches what AndroidWorld expects.

    Args:
        env: AndroidWorld environment
        **kwargs: Additional configuration options

    Returns:
        Configured BMLMAgent instance
    """
    config = BMLMAgentConfig(**kwargs)
    return BMLMAgent(env, config)
