# Big Model Little Model (BMLM)

Hierarchical agent architecture for Android GUI automation, using a big model (planner) and small model (executor) coordinated by an orchestrator.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator                           │
│  - Decides when to call big vs small model                  │
│  - Tracks plan progress and state                           │
│  - Triggers replanning on errors/low confidence             │
└──────────────┬────────────────────────┬─────────────────────┘
               │                        │
    ┌──────────▼──────────┐  ┌──────────▼──────────┐
    │     Big Model       │  │    Small Model      │
    │  (Qwen2.5-VL-7B)    │  │  (Qwen2.5-3B)       │
    │                     │  │                     │
    │  Creates multi-step │  │  Executes single    │
    │  plans from UI      │  │  steps, maps plan   │
    │  state + task       │  │  to actions         │
    └─────────────────────┘  └──────────┬──────────┘
                                        │
                             ┌──────────▼──────────┐
                             │   Action Executor   │
                             │                     │
                             │  Executes on        │
                             │  Android emulator   │
                             └──────────┬──────────┘
                                        │
                             ┌──────────▼──────────┐
                             │  AndroidWorld Env   │
                             └─────────────────────┘
```

## Jeeves Accessibility Service

The project includes **Jeeves**, a custom Android app that provides:
- Bounding box overlays for UI elements
- Accessibility service for element detection
- ContentProvider API for querying UI state

Jeeves is **automatically installed and configured** when you run the benchmark - no manual setup required in the emulator UI. The setup script handles:
1. APK installation via ADB
2. Accessibility service enablement via `adb shell settings put`
3. Overlay permission granting via `adb shell appops`

## Quick Start

### 1. Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Android Studio with emulator installed
- [uv](https://github.com/astral-sh/uv) package manager
- CMake (required for building some dependencies)

```bash
# Install CMake via Homebrew
brew install cmake
```

### 2. Create Android Virtual Device

1. Open Android Studio → Virtual Device Manager
2. Create new device:
   - **Device**: Pixel 6
   - **System Image**: Tiramisu (API 33)
   - **AVD Name**: `AndroidWorldAvd`

### 3. Install Dependencies

```bash
cd BigModelLittleModel

# Create virtual environment and install dependencies
uv sync

# This will download MLX, mlx-lm, android-world, and other dependencies
```

### 4. Download Models

Models are downloaded automatically on first use, but you can pre-download:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Pre-download models (optional)
python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-VL-7B-Instruct-4bit')"
python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-3B-Instruct-4bit')"
```

### 5. Test Models

```bash
# Test that models load and run correctly
uv run python scripts/test_models.py
```

### 6. Start the Emulator

**Important**: Launch from command line, not Android Studio UI:

```bash
# Find your emulator path (usually in ~/Library/Android/sdk/emulator/)
~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -grpc 8554
```

### 7. Run Benchmark

```bash
# Dry run (test setup without running tasks)
uv run python scripts/run_benchmark.py --dry-run

# Run specific task
uv run python scripts/run_benchmark.py --task SystemBrightnessMax --output results.json

# Run all tasks
uv run python scripts/run_benchmark.py --output results.json

# Skip Jeeves setup (for debugging)
uv run python scripts/run_benchmark.py --task SystemBrightnessMax --skip-jeeves
```

On first run, the benchmark script will automatically:
1. Install and configure Jeeves on the emulator
2. Enable the accessibility service
3. Grant overlay permissions

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
models:
  big:
    path: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    max_tokens: 1024
  small:
    path: "mlx-community/Qwen2.5-3B-Instruct-4bit"
    max_tokens: 256

orchestrator:
  max_steps_without_replan: 10
  confidence_threshold: medium
```

## Project Structure

```
BigModelLittleModel/
├── src/bmlm/
│   ├── models/
│   │   ├── base.py          # Abstract model interface
│   │   ├── big_model.py     # Planner (creates plans)
│   │   └── small_model.py   # Executor (runs steps)
│   ├── orchestrator/
│   │   ├── controller.py    # Main coordination logic
│   │   └── plan.py          # Plan data structures
│   ├── android/
│   │   └── actions.py       # Action execution
│   ├── tracing/             # Phoenix observability
│   │   ├── setup.py         # Phoenix initialization
│   │   └── spans.py         # Span helpers
│   └── benchmark/
│       └── agent.py         # AndroidWorld integration
├── jeeves/                   # Android accessibility app
│   ├── app/src/main/        # Android source code
│   └── setup_jeeves.py      # Auto-setup script
├── configs/
│   └── default.yaml         # Configuration
└── scripts/
    ├── test_models.py       # Test model loading
    ├── profile_latency.py   # Measure inference latency
    └── run_benchmark.py     # Run AndroidWorld benchmark
```

## Profiling Latency

```bash
# Profile small model latency
uv run python scripts/profile_latency.py mlx-community/Qwen2.5-3B-Instruct-4bit

# Profile with specific token counts
uv run python scripts/profile_latency.py --tokens 64,128,256 --runs 10
```

## Replan Triggers

The orchestrator calls the big model when:

1. **Task Start**: Initial plan generation
2. **Low Confidence**: Small model uncertain about action
3. **Needs Replanning**: Small model explicitly requests replan
4. **Max Steps Reached**: After N steps without replanning
5. **Step Failed**: Action execution failed

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run linter
uv run ruff check src/

# Run tests
uv run pytest
```

## Troubleshooting

### Emulator won't start
- Ensure you have enough disk space (8GB+)
- Try cold boot: `emulator -avd AndroidWorldAvd -grpc 8554 -no-snapshot-load`

### Model loading fails
- Check you have enough RAM (24GB recommended)
- Try smaller models: `Qwen2.5-1.5B-Instruct-4bit`

### AndroidWorld connection fails
- Verify emulator is running with `-grpc 8554` flag
- Check ADB connection: `adb devices`

### Slow inference
- Ensure you're using 4-bit quantized models
- Close other GPU-intensive applications
- Profile with `scripts/profile_latency.py`

### Jeeves setup fails
- Check ADB connection: `adb devices`
- Manual test: `uv run python jeeves/setup_jeeves.py`
- Force reinstall: `uv run python jeeves/setup_jeeves.py --force`
- If APK not built, build it manually:
  ```bash
  cd jeeves
  ./gradlew assembleDebug
  ```

## Tracing with Phoenix

The project includes optional Phoenix tracing to visualize what the models are thinking at each step.

### Setup

```bash
# Install tracing dependencies
uv sync --extra tracing
```

### Usage

```bash
# Run with tracing enabled
uv run python scripts/run_benchmark.py --task SystemBrightnessMax --enable-tracing
```

This will:
1. Start a local Phoenix server
2. Open Phoenix UI at http://localhost:6006
3. Record traces for all model calls and actions

### What Gets Traced

| Component | Attributes |
|-----------|------------|
| Big Model (planner) | prompt, plan output, generation_time_ms |
| Small Model (executor) | current_step, action chosen, confidence |
| Actions | action_type, target_id, success/failure, duration |
| Tasks | goal, steps taken, final score, elapsed time |

### Standalone Phoenix

You can also run Phoenix standalone to view past traces:

```bash
uv run phoenix serve
# Then open http://localhost:6006
```
