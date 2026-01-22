#!/usr/bin/env python3
"""Run BMLM agent on AndroidWorld benchmark."""

import argparse
import json
import sys
import time
from pathlib import Path

import structlog
import yaml

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


def setup_environment(config: dict):
    """Set up AndroidWorld environment.

    Returns:
        Tuple of (controller, task_registry)
    """
    from android_world.env.android_world_controller import get_controller
    from android_world.registry import TaskRegistry

    log.info("setting_up_environment")

    # Create controller using get_controller helper
    controller = get_controller(
        console_port=5554,  # Default emulator console port
        adb_path=config["android"].get("adb_path", "~/Library/Android/sdk/platform-tools/adb"),
        grpc_port=config["android"]["grpc_port"],
    )

    # Get task registry
    task_reg = TaskRegistry()

    return controller, task_reg


def create_agent(env, config: dict):
    """Create BMLM agent with configuration."""
    from bmlm.benchmark.agent import BMLMAgent, BMLMAgentConfig

    agent_config = BMLMAgentConfig(
        big_model_path=config["models"]["big"]["path"],
        small_model_path=config["models"]["small"]["path"],
        big_model_max_tokens=config["models"]["big"]["max_tokens"],
        small_model_max_tokens=config["models"]["small"]["max_tokens"],
        temperature=config["models"]["big"]["temperature"],
        max_steps_without_replan=config["orchestrator"]["max_steps_without_replan"],
        wait_after_action_ms=config["orchestrator"]["wait_after_action_ms"],
    )

    agent = BMLMAgent(env, agent_config)
    return agent


def run_task(agent, task, max_steps: int) -> dict:
    """Run a single task and return results.

    Args:
        agent: BMLM agent
        task: AndroidWorld task
        max_steps: Maximum steps allowed

    Returns:
        Dict with task results
    """
    task_name = task.name if hasattr(task, "name") else str(task)
    log.info("starting_task", task=task_name)

    start_time = time.perf_counter()

    # Set task on agent
    agent.set_task(task.goal if hasattr(task, "goal") else str(task))

    # Run until done or max steps
    steps = 0
    done = False

    while not done and steps < max_steps:
        result = agent.step()
        steps += 1
        done = result.done

        if steps % 10 == 0:
            log.info("task_progress", task=task_name, steps=steps)

    elapsed = time.perf_counter() - start_time

    # Check success (task-specific evaluation)
    success = False
    if hasattr(task, "evaluate"):
        try:
            success = task.evaluate()
        except Exception as e:
            log.warning("evaluation_failed", error=str(e))

    stats = agent.orchestrator.get_stats()

    result = {
        "task": task_name,
        "success": success,
        "steps": steps,
        "replans": stats["total_replans"],
        "elapsed_s": round(elapsed, 2),
        "done": done,
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

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        log.error("config_not_found", path=str(config_path))
        return 1

    config = load_config(config_path)
    log.info("config_loaded", path=str(config_path))

    # Set up environment
    try:
        env, task_registry = setup_environment(config)
    except Exception as e:
        log.error("environment_setup_failed", error=str(e))
        log.info("hint", message="Make sure the Android emulator is running with: emulator -avd AndroidWorldAvd -grpc 8554")
        return 1

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
    if args.task:
        tasks = [task_registry.get(args.task)]
    else:
        task_filter = config["benchmark"].get("task_filter", [])
        if task_filter:
            tasks = [task_registry.get(t) for t in task_filter]
        else:
            tasks = list(task_registry.get_all())

    # Run tasks
    results = []
    for task in tasks:
        try:
            agent.reset()
            result = run_task(agent, task, args.max_steps)
            results.append(result)
        except Exception as e:
            log.error("task_failed", task=str(task), error=str(e))
            results.append({
                "task": str(task),
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

    return 0 if success_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
