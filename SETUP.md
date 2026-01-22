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

```bash
# 1. Start emulator with gRPC flag
~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -grpc 8554

# 2. Wait for emulator to boot, then run benchmark
#    - Jeeves setup is automatic
#    - Phoenix tracing starts automatically at http://localhost:6006
uv run python scripts/run_benchmark.py --dry-run
```

## Common Commands

```bash
# Run specific task
uv run python scripts/run_benchmark.py --task SystemBrightnessMax

# Run without tracing
uv run python scripts/run_benchmark.py --task SystemBrightnessMax --no-tracing

# Manual Jeeves setup (if needed)
uv run python jeeves/setup_jeeves.py
```
