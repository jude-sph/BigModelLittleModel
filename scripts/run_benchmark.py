#!/usr/bin/env python3
"""Run BMLM agent on AndroidWorld benchmark."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import structlog
import yaml

# Add project root to path for jeeves import
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_jeeves(config: dict) -> bool:
    """Set up Jeeves accessibility service on the emulator.

    Args:
        config: Configuration dictionary

    Returns:
        True if Jeeves is ready, False otherwise
    """
    from jeeves.setup_jeeves import JeevesAutoSetup

    log.info("setting_up_jeeves")

    # Configure logging for jeeves module
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device_serial = config["android"].get("adb_serial", "emulator-5554")
    setup = JeevesAutoSetup(device_serial=device_serial)

    if setup.ensure_jeeves_ready(force_reinstall=False):
        log.info("jeeves_ready")
        # Enable overlay so agent can see numbered elements on screen
        setup.enable_overlay_visibility()
        return True
    else:
        log.warning("jeeves_setup_failed", message="Continuing without Jeeves")
        return False


def setup_environment(config: dict):
    """Set up AndroidWorld environment.

    Returns:
        Tuple of (env, task_registry)
    """
    from android_world.env import env_launcher
    from android_world.registry import TaskRegistry

    log.info("setting_up_environment")

    # Create AsyncEnv using env_launcher (this is what tasks expect)
    env = env_launcher.load_and_setup_env(
        console_port=5554,  # Default emulator console port
        adb_path=config["android"].get("adb_path", "~/Library/Android/sdk/platform-tools/adb"),
        grpc_port=config["android"]["grpc_port"],
        emulator_setup=False,  # Don't run setup again
    )

    # Get task registry
    task_reg = TaskRegistry()

    return env, task_reg


def create_agent(env, config: dict):
    """Create BMLM agent with configuration."""
    from bmlm.benchmark.agent import BMLMAgent, BMLMAgentConfig

    agent_config = BMLMAgentConfig(
        big_model_path=config["models"]["big"]["path"],
        small_model_path=config["models"]["small"]["path"],
        big_model_max_tokens=config["models"]["big"]["max_tokens"],
        small_model_max_tokens=config["models"]["small"]["max_tokens"],
        temperature=config["models"]["big"]["temperature"],
        max_plan_steps=config["orchestrator"]["max_plan_steps"],
        max_steps_without_replan=config["orchestrator"]["max_steps_without_replan"],
        wait_after_action_ms=config["orchestrator"]["wait_after_action_ms"],
    )

    agent = BMLMAgent(env, agent_config)
    return agent


def run_task(agent, task_name: str, task_class, env, max_steps: int) -> dict:
    """Run a single task and return results.

    Args:
        agent: BMLM agent
        task_name: Name of the task
        task_class: AndroidWorld task class
        env: AndroidWorld environment/controller
        max_steps: Maximum steps allowed

    Returns:
        Dict with task results
    """
    log.info("starting_task", task=task_name)

    # Instantiate task with random params
    params = task_class.generate_random_params()
    task = task_class(params)

    # Initialize the task on device
    log.info("initializing_task", task=task_name)
    task.initialize_task(env)

    start_time = time.perf_counter()

    # Get the goal and set it on the agent
    goal = task.goal
    log.info("task_goal", goal=goal)
    agent.set_task(goal)

    # Run until done or max steps
    steps = 0
    done = False

    while not done and steps < max_steps:
        result = agent.step(goal)
        steps += 1
        done = result.done

        if steps % 10 == 0:
            log.info("task_progress", task=task_name, steps=steps)

    elapsed = time.perf_counter() - start_time

    # Check success using task's evaluation
    success = False
    try:
        success_score = task.is_successful(env)
        success = success_score > 0.5  # Threshold for success
        log.info("task_evaluated", score=success_score)
    except Exception as e:
        log.warning("evaluation_failed", error=str(e))

    stats = agent.orchestrator.get_stats()

    # Determine why the task ended
    if done:
        termination_reason = "plan_complete"
    elif steps >= max_steps:
        termination_reason = "max_steps_reached"
    else:
        termination_reason = "unknown"

    result = {
        "task": task_name,
        "goal": goal,
        "success": success,
        "steps": steps,
        "replans": stats["total_replans"],
        "elapsed_s": round(elapsed, 2),
        "done": done,
        "termination_reason": termination_reason,
    }

    log.info("task_complete", **result)
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run BMLM on AndroidWorld benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task to run (default: all)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per task",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just load models and exit (test setup)",
    )
    parser.add_argument(
        "--skip-jeeves",
        action="store_true",
        help="Skip Jeeves setup (for testing without bounding boxes)",
    )
    parser.add_argument(
        "--no-tracing",
        action="store_true",
        help="Disable Phoenix tracing (tracing is ON by default)",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        log.error("config_not_found", path=str(config_path))
        return 1

    config = load_config(config_path)
    log.info("config_loaded", path=str(config_path))

    # Initialize tracing (enabled by default)
    tracing_enabled = False
    if not args.no_tracing:
        try:
            from bmlm.tracing import init_tracing
            if init_tracing(project_name="bmlm"):
                log.info("tracing_enabled", url="http://localhost:6006")
                tracing_enabled = True
            else:
                log.warning("tracing_init_failed")
        except ImportError:
            log.warning("tracing_not_available", message="Install with: uv sync --extra tracing")

    # Set up environment FIRST (AndroidWorld installs its own accessibility forwarder)
    try:
        env, task_registry = setup_environment(config)
    except Exception as e:
        log.error("environment_setup_failed", error=str(e))
        log.info("hint", message="Make sure the Android emulator is running with: emulator -avd AndroidWorldAvd -grpc 8554")
        return 1

    # Set up Jeeves AFTER AndroidWorld (so we can add it alongside the forwarder)
    if not args.skip_jeeves:
        try:
            setup_jeeves(config)
        except Exception as e:
            log.warning("jeeves_setup_error", error=str(e))
            log.info("continuing_without_jeeves")

    # Create agent
    agent = create_agent(env, config)

    # Load models
    log.info("loading_models")
    agent.load_models()
    log.info("models_loaded")

    if args.dry_run:
        log.info("dry_run_complete", message="Models loaded successfully")
        agent.unload_models()
        return 0

    # Get tasks to run
    # Get the task registry dictionary for AndroidWorld tasks
    task_dict = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

    if args.task:
        if args.task not in task_dict:
            log.error("task_not_found", task=args.task, available=list(task_dict.keys())[:10])
            return 1
        tasks = [(args.task, task_dict[args.task])]
    else:
        task_filter = config["benchmark"].get("task_filter", [])
        if task_filter:
            tasks = [(t, task_dict[t]) for t in task_filter if t in task_dict]
        else:
            tasks = list(task_dict.items())

    # Import trace_task if tracing is enabled
    trace_task_cm = None
    if tracing_enabled:
        try:
            from bmlm.tracing import trace_task
            trace_task_cm = trace_task
        except ImportError:
            pass

    # Run tasks
    results = []
    for task_name, task_class in tasks:
        try:
            agent.reset()

            # Wrap task in tracing if enabled
            if trace_task_cm:
                # Get goal early for tracing
                params = task_class.generate_random_params()
                temp_task = task_class(params)
                goal = temp_task.goal

                with trace_task_cm(task_name, goal, args.max_steps) as trace_result:
                    result = run_task(agent, task_name, task_class, env, args.max_steps)
                    trace_result["success"] = result.get("success", False)
                    trace_result["score"] = 1.0 if result.get("success") else 0.0
                    trace_result["steps"] = result.get("steps", 0)
                    trace_result["replans"] = result.get("replans", 0)
                    trace_result["elapsed_s"] = result.get("elapsed_s", 0)
                    trace_result["termination_reason"] = result.get("termination_reason", "unknown")
            else:
                result = run_task(agent, task_name, task_class, env, args.max_steps)

            results.append(result)
        except Exception as e:
            log.error("task_failed", task=task_name, error=str(e))
            results.append({
                "task": task_name,
                "success": False,
                "error": str(e),
            })

    # Summary
    success_count = sum(1 for r in results if r.get("success"))
    total = len(results)

    log.info(
        "benchmark_complete",
        total_tasks=total,
        successful=success_count,
        success_rate=f"{success_count/total*100:.1f}%" if total > 0 else "N/A",
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info("results_saved", path=str(output_path))

    # Cleanup
    agent.unload_models()

    # Shutdown tracing
    if tracing_enabled:
        try:
            from bmlm.tracing import shutdown_tracing
            shutdown_tracing()
        except ImportError:
            pass

    return 0 if success_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
