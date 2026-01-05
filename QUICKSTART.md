# Quick Start Guide

Get up and running with the AI Motion Animator in 5 minutes!

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- Webcam
- Portrait image

## Installation

### 1. Clone and Setup

```bash
git clone https://github.com/GilbertoBitt/stream-motion-animator.git
cd stream-motion-animator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA

```bash
# For CUDA 11.8 (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Prepare Your Image

Use a clear portrait photo:
- Face clearly visible
- Good lighting
- 512x512 to 2048x2048 pixels
- JPG or PNG format

Optional: Convert/prepare your image:
```bash
python scripts/convert_image.py --input photo.jpg --square --size 1024 1024
```

## Run Your First Animation

### Basic Run (Display Only)

```bash
python main.py --image your_portrait.jpg
```

**Controls:**
- `Q` or `ESC`: Quit
- `F`: Toggle FPS display
- `R`: Show performance report
- `P`: Pause/Resume

### With Performance Optimization

For RTX 3080 users:
```bash
python main.py \
    --image your_portrait.jpg \
    --config examples/config_rtx3080.yaml \
    --batch-size 4 \
    --fp16
```

For other GPUs:
```bash
# RTX 2060/3060
python main.py --image your_portrait.jpg --batch-size 2 --fp16

# RTX 3070
python main.py --image your_portrait.jpg --batch-size 3 --fp16

# RTX 4080/4090
python main.py --image your_portrait.jpg --batch-size 6 --fp16
```

## Troubleshooting

### "CUDA not available"

Install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### "Camera not found"

List available cameras:
```bash
python -c "import cv2; [print(f'Camera {i}') for i in range(5) if cv2.VideoCapture(i).isOpened()]"
```

Use specific camera:
```bash
python main.py --image portrait.jpg --camera 1
```

### Low FPS

1. Reduce batch size: `--batch-size 1`
2. Enable FP16: `--fp16`
3. Lower resolution: `--resolution 1280 720`
4. Close background apps

### "Out of memory"

Reduce batch size:
```bash
python main.py --image portrait.jpg --batch-size 1
```

## Next Steps

### Enable Streaming Outputs

**Spout (Windows only):**
```bash
pip install SpoutGL
python main.py --image portrait.jpg --output display spout
```

Add "Spout2 Capture" source in OBS and select "AI_Avatar"

**NDI:**
1. Install NDI SDK from https://www.ndi.tv/sdk/
2. Install ndi-python: `pip install ndi-python`
3. Run: `python main.py --image portrait.jpg --output display ndi`
4. Add NDI source in OBS and select "AI_Avatar"

### Optimize Performance

Run benchmark to find optimal settings:
```bash
python scripts/benchmark.py --image portrait.jpg --auto-tune
```

### Use Multiple Images

Switch between images during runtime:
```bash
python main.py --image img1.jpg img2.jpg img3.jpg
```

Press `1`, `2`, `3` to switch between images while running.

## Getting Help

- **Documentation**: See `docs/` folder
- **Issues**: https://github.com/GilbertoBitt/stream-motion-animator/issues
- **Discussions**: https://github.com/GilbertoBitt/stream-motion-animator/discussions

## Common Use Cases

### Streaming Setup

```bash
# Start animator with Spout/NDI
python main.py \
    --image avatar.jpg \
    --config examples/config_rtx3080.yaml \
    --output spout ndi

# In OBS:
# 1. Add Spout2 Capture source
# 2. Or add NDI source
# 3. Enjoy smooth 60+ FPS animation!
```

### Content Creation

```bash
# High quality for recording
python main.py \
    --image portrait.jpg \
    --batch-size 4 \
    --fp16 \
    --target-fps 60
```

### Development/Testing

```bash
# Debug mode with metrics
python main.py \
    --image test.jpg \
    --log-level DEBUG \
    --enable-metrics
```

## Performance Targets

| GPU | Resolution | Expected FPS |
|-----|-----------|-------------|
| RTX 4090 | 1080p | 120+ |
| RTX 3090 | 1080p | 75-85 |
| RTX 3080 | 1080p | **65-70** |
| RTX 3070 | 1080p | 50-60 |
| RTX 3060 | 1080p | 40-50 |
| RTX 2060 | 1080p | 30-40 |

## Tips for Best Results

1. **Good Source Image**
   - High resolution (1024x1024+)
   - Clear face, well-lit
   - Neutral expression
   - Centered composition

2. **Camera Setup**
   - Good lighting on your face
   - Eye-level camera position
   - Clean background
   - Stable camera mount

3. **Performance**
   - Start with recommended batch size for your GPU
   - Always enable FP16 (--fp16)
   - Monitor GPU temperature
   - Close background applications

4. **Quality vs Speed**
   - Higher batch size = better GPU utilization but higher latency
   - Lower batch size = lower latency but may not fully use GPU
   - Use benchmark to find sweet spot

Enjoy your AI-powered avatar! 🎉
