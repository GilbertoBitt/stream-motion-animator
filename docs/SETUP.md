# Setup Guide

## Prerequisites

### System Requirements

**Minimum:**
- GPU: NVIDIA RTX 2060 or better (6GB+ VRAM)
- CPU: Intel i5-8400 / AMD Ryzen 5 2600
- RAM: 16GB
- OS: Windows 10/11, Linux (Ubuntu 20.04+)
- Python: 3.8 or higher
- CUDA: 11.8 or higher (for GPU acceleration)

**Recommended (RTX 3080):**
- GPU: NVIDIA RTX 3080 (10GB VRAM)
- CPU: Intel i7-9700K / AMD Ryzen 7 3700X
- RAM: 32GB
- SSD for model storage
- Webcam: 1080p @ 30fps or higher

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/GilbertoBitt/stream-motion-animator.git
cd stream-motion-animator
```

### 2. Create Virtual Environment

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA

Visit [pytorch.org](https://pytorch.org/get-started/locally/) and install the appropriate version for your system.

**Example for CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 5. Download AI Models

```bash
python scripts/download_models.py --model live-portrait
```

**Note:** Currently, you need to manually set up Live Portrait:

1. Clone Live Portrait repository:
   ```bash
   git clone https://github.com/lylalabs/live-portrait
   ```

2. Follow their setup instructions

3. Copy or symlink model weights to `models/` directory

### 6. Optional: Install Spout (Windows Only)

For Spout output support:

```bash
pip install SpoutGL
```

Download Spout from: https://spout.zeal.co/

### 7. Optional: Install NDI

For NDI output support:

1. Download and install NDI SDK from: https://www.ndi.tv/sdk/
2. Install ndi-python:
   ```bash
   pip install ndi-python
   ```

## Configuration

### Basic Configuration

Copy and edit the default configuration:

```bash
cp config.yaml config_local.yaml
# Edit config_local.yaml with your settings
```

### Key Settings to Adjust

**For RTX 3080:**
```yaml
animation:
  gpu_id: 0
  batch_size: 4
  inference_precision: fp16
  use_half_precision: true

performance:
  target_fps: 60
  max_latency_ms: 15
```

**For RTX 2060:**
```yaml
animation:
  gpu_id: 0
  batch_size: 2
  inference_precision: fp16
  use_half_precision: true

performance:
  target_fps: 30
  max_latency_ms: 25
```

## Verify Installation

### 1. Check GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 2. Check Dependencies

```bash
python -c "import cv2, mediapipe, numpy; print('All core dependencies installed')"
```

### 3. Run Tests

```bash
pytest tests/ -v
```

## First Run

### Prepare Test Image

1. Use a high-quality portrait photo (512x512 to 2048x2048)
2. Face should be clearly visible and well-lit
3. Supported formats: JPG, PNG, BMP

### Run the Animator

```bash
# Basic test with display output
python main.py --image examples/sample_portrait.jpg

# With custom config
python main.py --image portrait.jpg --config examples/config_rtx3080.yaml

# With Spout output (Windows)
python main.py --image portrait.jpg --output display spout

# With NDI output
python main.py --image portrait.jpg --output display ndi
```

## Benchmark Your System

```bash
# Run 60-second benchmark
python scripts/benchmark.py --image portrait.jpg --duration 60

# Auto-tune to find optimal batch size
python scripts/benchmark.py --image portrait.jpg --auto-tune
```

## Troubleshooting

### "CUDA out of memory"

**Solution:** Reduce batch size in config:
```yaml
animation:
  batch_size: 2  # or 1
```

### "Camera not found"

**Solution:** List available cameras:
```bash
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

Then specify the correct camera:
```bash
python main.py --image portrait.jpg --camera 1
```

### Low FPS

**Solutions:**
1. Enable FP16: `--fp16`
2. Reduce batch size
3. Lower capture resolution
4. Close background applications
5. Update GPU drivers

### Spout/NDI Not Working

**Spout (Windows):**
- Install SpoutGL: `pip install SpoutGL`
- Ensure Spout is installed system-wide
- Check Windows firewall

**NDI:**
- Install NDI SDK from NewTek
- Set NDI_RUNTIME_DIR environment variable
- Restart after installation

## Next Steps

- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions
- See [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) for performance tuning
- See [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md) for adding new models

## Getting Help

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share tips
- Wiki: Community guides and examples
