# Ubuntu/NVIDIA Setup Guide

This guide covers setting up BMLM on Ubuntu with an NVIDIA GPU (tested on RTX 3090).

## Prerequisites

- Ubuntu 20.04 or later
- NVIDIA GPU with CUDA support (24GB+ VRAM recommended for 7B model)
- CUDA 12.1+ and cuDNN installed
- Python 3.11+
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
git clone <repository-url>
cd BigModelLittleModel
git checkout ubuntu-nvidia

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -e .

# Optional: Install flash-attention for faster inference
pip install flash-attn --no-build-isolation
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
# Install Android SDK (if not already installed)
sudo apt install android-sdk

# Or use Android Studio's SDK manager
# Typical Linux SDK path: ~/Android/Sdk

# Create AVD for AndroidWorld
# Follow AndroidWorld setup instructions

# Start emulator with gRPC
emulator -avd AndroidWorldAvd -grpc 8554
```

## 5. Configure ADB Path

Edit `configs/default.yaml` if your ADB path differs:

```yaml
android:
  adb_path: "~/Android/Sdk/platform-tools/adb"
```

## 6. Run Benchmark

```bash
# Activate environment
source .venv/bin/activate

# Run with tracing (recommended)
python scripts/run_benchmark.py --config configs/default.yaml

# Run specific task
python scripts/run_benchmark.py --task ContactsAddContact

# Dry run to test model loading
python scripts/run_benchmark.py --dry-run
```

## Memory Usage

With 4-bit quantization (default):
- Big model (Qwen2.5-VL-7B): ~5-6 GB VRAM
- Small model (Qwen2.5-3B): ~2-3 GB VRAM
- Total: ~8-10 GB VRAM

For GPUs with less memory, consider:
- Using smaller models (Qwen2.5-1.5B for small model)
- Running models sequentially instead of simultaneously

## Troubleshooting

### CUDA out of memory
```bash
# Clear cache between runs
python -c "import torch; torch.cuda.empty_cache()"

# Use smaller batch sizes or models
```

### bitsandbytes issues
```bash
# Reinstall bitsandbytes
pip uninstall bitsandbytes
pip install bitsandbytes --no-cache-dir
```

### Flash attention installation fails
```bash
# Install build dependencies
pip install packaging ninja

# Install with specific CUDA version
CUDA_HOME=/usr/local/cuda-12.1 pip install flash-attn --no-build-isolation
```
