# Setup

## First-Time Setup

```bash
# 1. Install dependencies (includes tracing)
cd BigModelLittleModel
uv sync --extra tracing

# 2. Create Android Virtual Device
#    Open Android Studio → Virtual Device Manager → Create device:
#    - Device: Pixel 6
#    - System Image: Tiramisu (API 33)
#    - AVD Name: AndroidWorldAvd

# 3. (Optional) Pre-download models to avoid first-run delay
uv run python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-VL-7B-Instruct-4bit')"
uv run python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-3B-Instruct-4bit')"
```

## Every-Time Setup

**Terminal 1: Start emulator**
```bash
~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -grpc 8554
```

**Terminal 2: Start Phoenix server (keeps data between runs)**
```bash
uv run phoenix serve
# UI available at http://localhost:6006
```

**Terminal 3: Run benchmark**
```bash
uv run python scripts/run_benchmark.py --task SystemBrightnessMax --output results.json
```

## Common Commands

```bash
# Dry run (test setup)
uv run python scripts/run_benchmark.py --dry-run

# Run specific task
uv run python scripts/run_benchmark.py --task SystemBrightnessMax

# Run without tracing
uv run python scripts/run_benchmark.py --no-tracing

# Manual Jeeves setup
uv run python jeeves/setup_jeeves.py

# Clear Phoenix data (if corrupted)
rm -rf ~/.phoenix
```

## Notes

- **Phoenix data** persists in `~/.phoenix/` between runs
- **Jeeves setup** is automatic (runs after AndroidWorld environment setup)
- If Phoenix server isn't running, benchmark will start one (but it dies when script ends)
