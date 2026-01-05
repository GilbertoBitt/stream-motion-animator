# Installation Guide

Complete setup instructions for Stream Motion Animator.

## Table of Contents

- [System Requirements](#system-requirements)
- [Python Environment Setup](#python-environment-setup)
- [PyTorch CUDA Installation](#pytorch-cuda-installation)
- [Dependencies Installation](#dependencies-installation)
- [Model Download](#model-download)
- [Spout Setup (Windows)](#spout-setup-windows)
- [NDI Setup](#ndi-setup)
- [OBS Integration](#obs-integration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10 64-bit or newer
- **GPU**: NVIDIA GTX 1660 or better with 6GB+ VRAM
- **CPU**: Intel i5 or AMD Ryzen 5 (4+ cores recommended)
- **RAM**: 8GB
- **Storage**: 10GB free space (models + dependencies)
- **Webcam**: Any USB webcam (720p or higher)
- **Python**: 3.8, 3.9, 3.10, or 3.11

### Recommended Requirements
- **OS**: Windows 11 64-bit
- **GPU**: NVIDIA RTX 3080 with 10GB+ VRAM
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **RAM**: 16GB or more
- **Storage**: SSD with 20GB+ free space
- **Webcam**: 1080p webcam with good lighting
- **Python**: 3.10

## Python Environment Setup

### 1. Install Python

Download Python from [python.org](https://www.python.org/downloads/):
- Python 3.10 recommended for best compatibility
- During installation, check "Add Python to PATH"

Verify installation:
```bash
python --version
# Should output: Python 3.10.x
```

### 2. Create Virtual Environment

```bash
cd stream-motion-animator
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows**:
```bash
venv\Scripts\activate
```

**Linux/Mac**:
```bash
source venv/bin/activate
```

Your prompt should now show `(venv)`.

## PyTorch CUDA Installation

PyTorch with CUDA support is required for GPU acceleration.

### For CUDA 11.8 (RTX 30/40 series)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### For CUDA 12.1 (Latest RTX 40 series)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify CUDA Installation

```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA Available: True
CUDA Version: 11.8
GPU: NVIDIA GeForce RTX 3080
```

## Dependencies Installation

### 1. Install Core Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch and torchvision (if not already installed)
- OpenCV for video processing
- MediaPipe for face tracking
- Other required packages

### 2. Install Development Dependencies (Optional)

For development and testing:
```bash
pip install -r requirements-dev.txt
```

## Model Download

### Automatic Download

Run the model downloader script:
```bash
python tools/download_models.py
```

This will:
1. Check available disk space
2. Download Live Portrait model (~500MB)
3. Extract to `models/liveportrait/`
4. Verify integrity

### Manual Download

If automatic download fails:

1. Visit [Live Portrait Releases](https://github.com/KwaiVGI/LivePortrait/releases)
2. Download `liveportrait_models.zip`
3. Extract to `models/liveportrait/`

## Spout Setup (Windows)

Spout allows low-latency video sharing with OBS.

### 1. Install Spout Library

The Python Spout library should be installed via requirements.txt:
```bash
pip install SpoutGL
```

### 2. Install Spout2 Plugin for OBS

1. Download Spout2 plugin from [OBS Spout2 Plugin](https://github.com/Off-World-Live/obs-spout2-plugin/releases)
2. Extract to OBS plugins folder:
   - Default: `C:\Program Files\obs-studio\obs-plugins\64bit\`
3. Restart OBS

## NDI Setup

NDI enables network video streaming.

### 1. Install NDI Runtime

1. Download NDI Tools from [NDI Website](https://ndi.tv/tools/)
2. Install NDI Runtime
3. Restart your computer

### 2. Install Python NDI Library

```bash
# Note: NDI Python bindings require manual setup
# See https://github.com/buresu/ndi-python for instructions
```

### 3. Install NDI Plugin for OBS

1. Download NDI plugin from [OBS NDI Plugin](https://github.com/obs-ndi/obs-ndi/releases)
2. Run installer
3. Restart OBS

## OBS Integration

### Using Spout (Recommended for local streaming)

1. Open OBS Studio
2. Add new source: **Spout2 Capture**
3. Select **StreamMotionAnimator** from dropdown
4. Adjust size and position as needed

### Using NDI

1. Open OBS Studio
2. Add new source: **NDI Source**
3. Select **Stream Motion Animator** from dropdown
4. Configure bandwidth settings if needed

### Transparency in OBS

For transparent backgrounds:
1. Right-click the source
2. Filters → Add → Color Key or Chroma Key
3. Or use the built-in alpha channel if supported

## Verification

### 1. Test Webcam

```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'Webcam FAILED'); cap.release()"
```

### 2. Test GPU

```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not Available')"
```

### 3. Run Benchmark

```bash
python tools/benchmark.py --duration 10
```

Expected output should show:
- System information with your GPU
- FPS measurements for each pipeline stage
- Performance recommendations

### 4. Test Character Image

Place a test image in `assets/characters/` and run:
```bash
python tools/test_character.py assets/characters/your_image.png
```

### 5. Run Application

```bash
python src/main.py
```

You should see:
- Console output with initialization messages
- Preview window showing animated character
- FPS statistics

Press **Q** to quit.

## Troubleshooting

### "CUDA not available"

**Solution**:
1. Verify NVIDIA drivers are installed (run `nvidia-smi`)
2. Reinstall PyTorch with CUDA support
3. Check CUDA version compatibility

### "Failed to open webcam"

**Solution**:
1. Check webcam is connected
2. Close other apps using webcam (Zoom, Teams, etc.)
3. Try different camera index in config: `video.source: 1`

### "Model not found"

**Solution**:
1. Run `python tools/download_models.py`
2. Check `models/liveportrait/` exists
3. Verify internet connection

### "Spout not working"

**Solution**:
1. Ensure SpoutGL is installed: `pip install SpoutGL`
2. Verify OBS Spout2 plugin is installed
3. Check source name matches: "StreamMotionAnimator"
4. Restart OBS after installing plugin

### "Low FPS"

**Solution**:
1. Enable FP16 in config: `ai_model.fp16: true`
2. Enable async pipeline: `performance.async_pipeline: true`
3. Reduce output resolution
4. Close background applications
5. Run benchmark to identify bottleneck

### Import Errors

**Solution**:
```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Next Steps

- Read [Performance Guide](PERFORMANCE.md) for optimization tips
- Check [Character Guide](CHARACTER_GUIDE.md) for preparing images
- See [Troubleshooting](TROUBLESHOOTING.md) for common issues

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/GilbertoBitt/stream-motion-animator/issues)
- **Discussions**: [Ask questions](https://github.com/GilbertoBitt/stream-motion-animator/discussions)

---

**Installation complete!** You're ready to start animating. 🎉
