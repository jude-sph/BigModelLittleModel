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

    result = {
        "task": task_name,
        "goal": goal,
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

    # Run tasks
    results = []
    for task_name, task_class in tasks:
        try:
            agent.reset()
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

    return 0 if success_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
