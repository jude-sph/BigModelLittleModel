# Ubuntu/NVIDIA Setup Guide

This guide covers setting up BMLM on Ubuntu with an NVIDIA GPU (tested on RTX 3090).

## Prerequisites

- Ubuntu 20.04 or later
- NVIDIA GPU with CUDA support (24GB+ VRAM recommended for 7B model)
- CUDA 12.1+ and cuDNN installed
- Python 3.11+
- Node.js (for ws-scrcpy)
- Android Studio with emulator

## 1. Install CUDA and cuDNN

```bash
# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-535

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 2. Clone and Set Up Project

```bash
git clone -b ubuntu-nvidia https://github.com/jude-sph/BigModelLittleModel
cd BigModelLittleModel

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
make install

# Or manually:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install -e ".[dev,tracing]"
```

## 3. Verify CUDA Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## 4. Set Up Android Emulator

```bash
# Start emulator with gRPC (required for AndroidWorld)
emulator -avd AndroidWorldAvd -grpc 8554

# Enable ADB over TCP (in another terminal, needed for ws-scrcpy)
adb tcpip 5555
adb connect localhost:5555
```

## 5. Remote Screen Viewing (from Mac)

To view the Android emulator from your Mac:

**On Ubuntu (start these in separate terminals):**

```bash
# Terminal 1: Start ws-scrcpy web interface
npx ws-scrcpy

# Terminal 2: Start Phoenix tracing (optional)
phoenix serve
```

**On Mac (SSH with port forwarding):**

```bash
ssh -L 8000:localhost:8000 -L 6006:localhost:6006 user@ubuntu-machine
```

**Open in browser on Mac:**
- Android screen: http://localhost:8000
- Phoenix tracing: http://localhost:6006

## 6. Run Benchmark

```bash
# Activate environment
source .venv/bin/activate

# Run with tracing
make run

# Or manually:
python scripts/run_benchmark.py --config configs/default.yaml

# Run specific task
python scripts/run_benchmark.py --task ContactsAddContact

# Dry run to test model loading
make dry-run
```

## Memory Usage

With 4-bit quantization (default):
- Big model (Qwen2.5-VL-7B): ~5-6 GB VRAM
- Small model (Qwen2.5-3B): ~2-3 GB VRAM
- Total: ~8-10 GB VRAM

## Quick Reference

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make run` | Run benchmark with tracing |
| `make dry-run` | Test model loading |
| `make scrcpy-web` | Start web-based screen viewer |
| `make phoenix` | Start tracing UI |
| `make gpu-status` | Check GPU usage |

## Troubleshooting

### CUDA out of memory
```bash
make clear-cuda
# Or use smaller models in configs/default.yaml
```

### bitsandbytes issues
```bash
pip uninstall bitsandbytes
pip install bitsandbytes --no-cache-dir
```

### ws-scrcpy can't connect
```bash
# Make sure ADB is in TCP mode
adb tcpip 5555
adb connect localhost:5555

# Check connection
adb devices
```

### Flash attention installation fails
```bash
pip install packaging ninja
CUDA_HOME=/usr/local/cuda-12.1 pip install flash-attn --no-build-isolation
```
