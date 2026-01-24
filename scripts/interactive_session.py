#!/usr/bin/env python3
"""Interactive testing session that keeps models loaded in memory.

Usage:
    python scripts/interactive_session.py

Commands in session:
    <task>          - Run a task (e.g., "Turn brightness to max")
    /reload         - Reload models
    /status         - Show current status
    /quit or /exit  - Exit session
"""

import readline  # Enables arrow keys and history in input()
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    print("=" * 60)
    print("BMLM Interactive Testing Session")
    print("=" * 60)
    print("\nModels will stay loaded between tasks for fast iteration.")
    print("Commands: /reload, /status, /quit\n")

    # Import after path setup
    from bmlm.android.actions import ActionExecutor
    from bmlm.benchmark.agent import BMLMAgentConfig
    from bmlm.models.base import ModelConfig
    from bmlm.models.big_model import BigModel
    from bmlm.models.small_model import SmallModel
    from bmlm.orchestrator.controller import Orchestrator, OrchestratorConfig

    config = BMLMAgentConfig()

    # Initialize models
    print("Loading big model...")
    big_model = BigModel(
        ModelConfig(
            model_path=config.big_model_path,
            max_tokens=config.big_model_max_tokens,
            temperature=config.temperature,
        ),
        max_plan_steps=config.max_plan_steps,
    )
    big_model.load()
    print(f"  ✓ {config.big_model_path}")

    print("Loading small model...")
    small_model = SmallModel(
        ModelConfig(
            model_path=config.small_model_path,
            max_tokens=config.small_model_max_tokens,
            temperature=config.temperature,
        )
    )
    small_model.load()
    print(f"  ✓ {config.small_model_path}")

    # Initialize orchestrator
    orchestrator = Orchestrator(
        big_model=big_model,
        small_model=small_model,
        config=OrchestratorConfig(
            max_steps_without_replan=config.max_steps_without_replan,
            wait_after_action_ms=config.wait_after_action_ms,
        ),
    )

    # Check if we have an Android environment
    env = None
    action_executor = None
    try:
        from android_world.env import AndroidWorldEnv
        print("\nConnecting to Android environment...")
        env = AndroidWorldEnv.create()
        action_executor = ActionExecutor(env)
        orchestrator.set_callbacks(
            get_ui_elements=action_executor.get_ui_elements,
            execute_action=action_executor.execute,
            get_screenshot=action_executor.get_screenshot,
        )
        print("  ✓ Connected to Android emulator")
    except Exception as e:
        print(f"\n⚠ No Android environment available: {e}")
        print("  Running in dry-run mode (planning only, no execution)")

        # Mock callbacks for dry-run mode
        def mock_ui_elements():
            return [
                {"index": 0, "text": "Phone", "type": "AppIcon", "content_desc": ""},
                {"index": 1, "text": "Messages", "type": "AppIcon", "content_desc": ""},
                {"index": 2, "text": "Camera", "type": "AppIcon", "content_desc": ""},
                {"index": 3, "text": "Settings", "type": "AppIcon", "content_desc": ""},
                {"index": 4, "text": "Chrome", "type": "AppIcon", "content_desc": ""},
            ]

        def mock_execute(action_dict):
            print(f"    [DRY-RUN] Would execute: {action_dict}")
            return True

        def mock_screenshot():
            return None

        orchestrator.set_callbacks(
            get_ui_elements=mock_ui_elements,
            execute_action=mock_execute,
            get_screenshot=mock_screenshot,
        )

    print("\n" + "=" * 60)
    print("Ready! Enter a task or command.")
    print("=" * 60 + "\n")

    # Main loop
    while True:
        try:
            user_input = input("\033[96mtask>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            break
        elif user_input.lower() == "/reload":
            print("Reloading models...")
            big_model.unload()
            small_model.unload()
            big_model.load()
            small_model.load()
            print("  ✓ Models reloaded")
            continue
        elif user_input.lower() == "/status":
            stats = orchestrator.get_stats()
            print(f"  Total steps: {stats['total_steps']}")
            print(f"  Total replans: {stats['total_replans']}")
            print(f"  Big model: {config.big_model_path}")
            print(f"  Small model: {config.small_model_path}")
            continue
        elif user_input.startswith("/"):
            print(f"  Unknown command: {user_input}")
            print("  Commands: /reload, /status, /quit")
            continue

        # Run task
        print()
        try:
            success = orchestrator.run_until_complete(user_input, max_steps=30)
            print()
            if success:
                print("\033[92m✓ Task completed successfully\033[0m")
            else:
                print("\033[91m✗ Task did not complete\033[0m")
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")
            import traceback
            traceback.print_exc()

        # Reset orchestrator state for next task
        from bmlm.orchestrator.controller import OrchestratorState
        orchestrator.state = OrchestratorState()
        print()

    # Cleanup
    print("Unloading models...")
    big_model.unload()
    small_model.unload()
    if env:
        try:
            env.close()
        except:
            pass
    print("Goodbye!")


if __name__ == "__main__":
    main()
