# BMLM Makefile - Convenience commands for running the benchmark

# ==================== Screen Viewing ====================

# Start ws-scrcpy web interface (view emulator in browser)
# Access at http://localhost:8000
scrcpy-web:
	npx ws-scrcpy

# Start Phoenix tracing UI
# Access at http://localhost:6006
phoenix:
	phoenix serve

# ==================== Emulator ====================

# Start Android emulator with gRPC
emulator-start:
	emulator -avd AndroidWorldAvd -grpc 8554 &
	@echo "Waiting for emulator to boot..."
	adb wait-for-device
	@echo "Emulator ready"

# Enable ADB over TCP (needed for ws-scrcpy)
emulator-tcpip:
	adb tcpip 5555
	adb connect localhost:5555
	@echo "ADB connected on localhost:5555"

# ==================== Benchmark ====================

# Run benchmark with tracing
run:
	python scripts/run_benchmark.py --config configs/default.yaml

# Run specific task
run-task:
	@read -p "Task name: " task; \
	python scripts/run_benchmark.py --config configs/default.yaml --task $$task

# Dry run (test model loading)
dry-run:
	python scripts/run_benchmark.py --config configs/default.yaml --dry-run

# Run without tracing
run-no-trace:
	python scripts/run_benchmark.py --config configs/default.yaml --no-tracing

# ==================== Development ====================

# Install dependencies (run this first)
install:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	pip install -e ".[dev,tracing]"

# Run linter
lint:
	ruff check src/

# Format code
format:
	ruff format src/

# ==================== Utilities ====================

# Check GPU status
gpu-status:
	nvidia-smi

# Clear CUDA cache
clear-cuda:
	python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"

# Take screenshot of emulator
screenshot:
	adb exec-out screencap -p > screenshot.png
	@echo "Screenshot saved to screenshot.png"

.PHONY: scrcpy-web phoenix emulator-start emulator-tcpip run run-task dry-run run-no-trace install lint format gpu-status clear-cuda screenshot
