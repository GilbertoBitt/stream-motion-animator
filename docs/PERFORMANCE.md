# Performance Optimization Guide

Detailed guide for optimizing Stream Motion Animator on RTX 3080 and achieving 60-70 FPS.

## Table of Contents

- [Performance Targets](#performance-targets)
- [RTX 3080 Optimization](#rtx-3080-optimization)
- [Configuration Tuning](#configuration-tuning)
- [Async Pipeline](#async-pipeline)
- [GPU Optimization](#gpu-optimization)
- [Monitoring Performance](#monitoring-performance)
- [Troubleshooting Performance](#troubleshooting-performance)
- [Advanced Optimizations](#advanced-optimizations)

## Performance Targets

### Target Specifications

| Component | Target | Acceptable | Issue |
|-----------|--------|------------|-------|
| **Total FPS** | 60-70 | 45-60 | <45 |
| **Inference Time** | <15ms | 15-20ms | >20ms |
| **GPU Usage** | 70-80% | 60-90% | >95% |
| **GPU Memory** | <8GB | 8-9GB | >9GB |
| **Latency** | <20ms | 20-30ms | >30ms |

### Expected Performance by GPU

| GPU Model | FP32 | FP16 | TensorRT |
|-----------|------|------|----------|
| RTX 3060 (12GB) | 30-35 | 40-50 | 50-60 |
| RTX 3070 (8GB) | 35-45 | 50-60 | 60-70 |
| RTX 3080 (10GB) | 45-55 | 60-70 | 70-85 |
| RTX 3090 (24GB) | 50-60 | 65-80 | 80-100 |
| RTX 4070 (12GB) | 50-60 | 70-85 | 85-100 |
| RTX 4080 (16GB) | 60-75 | 80-95 | 95-120 |

*Based on 1080p output, 720p input, single character*

## RTX 3080 Optimization

### Optimal Configuration

Edit `assets/config.yaml`:

```yaml
video:
  source: 0
  width: 1280        # 720p input
  height: 720
  fps: 60

ai_model:
  type: "liveportrait"
  device: "cuda"
  fp16: true         # Essential for RTX 3080
  batch_size: 1
  use_tensorrt: false  # Enable after TensorRT setup
  warmup_frames: 10

output:
  width: 1920        # 1080p output
  height: 1080

performance:
  target_fps: 60
  async_pipeline: true  # Essential for parallel processing
  frame_skip_threshold: 45
```

### Driver Configuration

1. **Update GPU Drivers**:
   - Download latest from [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)
   - Use "Game Ready" drivers for best performance

2. **NVIDIA Control Panel Settings**:
   - Power Management: "Prefer Maximum Performance"
   - Texture Filtering: "High Performance"
   - Low Latency Mode: "Ultra"

3. **Windows Settings**:
   - Power Plan: "High Performance"
   - Game Mode: Enabled
   - Hardware-accelerated GPU scheduling: Enabled

## Configuration Tuning

### Resolution Trade-offs

Lower input resolution for higher FPS:

```yaml
# High Performance (720p input)
video:
  width: 1280
  height: 720
# Expected: 60-70 FPS on RTX 3080

# Balanced (900p input)
video:
  width: 1600
  height: 900
# Expected: 50-60 FPS on RTX 3080

# Quality (1080p input)
video:
  width: 1920
  height: 1080
# Expected: 35-45 FPS on RTX 3080
```

### Character Management

Optimize character loading:

```yaml
character:
  preload_all: true      # Faster switching
  auto_crop: true
  target_size: [512, 512]  # Lower = faster
```

Reduce `target_size` for better performance:
- `[256, 256]`: Fastest, lower quality
- `[512, 512]`: Balanced (recommended)
- `[1024, 1024]`: Highest quality, slower

### Frame Skip Threshold

```yaml
performance:
  frame_skip_threshold: 45  # Skip frames if FPS drops below this
```

Adjustments:
- Set to `55` for stricter quality
- Set to `30` for maximum smoothness

## Async Pipeline

The async pipeline is critical for performance.

### How It Works

```
Thread 1: Webcam Capture (120+ FPS possible)
    ↓
Thread 2: Face Tracking (MediaPipe, ~100 FPS)
    ↓
Thread 3: AI Inference (GPU, 60-70 FPS) ← Bottleneck
    ↓
Thread 4: Output (Spout/NDI, 180+ FPS)
```

Each thread runs independently, maximizing throughput.

### Enable Async Pipeline

```yaml
performance:
  async_pipeline: true  # Must be true for 60+ FPS
```

### Sync Mode (Debugging Only)

Use sync mode only for debugging:
```yaml
performance:
  async_pipeline: false
```

Sync mode runs all stages sequentially, resulting in lower FPS but easier debugging.

## GPU Optimization

### FP16 (Half Precision)

**Essential for RTX 3080!**

```yaml
ai_model:
  fp16: true  # 2x speed boost
```

Benefits:
- 2x faster inference
- 50% less VRAM usage
- Minimal quality loss

### GPU Memory Management

Monitor memory usage:
```bash
nvidia-smi -l 1  # Update every second
```

If memory is high:
1. Reduce character preloading:
   ```yaml
   character:
     preload_all: false
   ```

2. Lower character target size:
   ```yaml
   character:
     target_size: [256, 256]
   ```

3. Reduce batch size (if using batching):
   ```yaml
   ai_model:
     batch_size: 1
   ```

### TensorRT Acceleration

For 10-15% additional speedup:

1. Install TensorRT:
   ```bash
   pip install tensorrt
   ```

2. Enable in config:
   ```yaml
   ai_model:
     use_tensorrt: true
   ```

3. First run will take longer (model conversion)
4. Subsequent runs will be faster

**Note**: TensorRT setup is advanced. See [TensorRT Guide](https://docs.nvidia.com/deeplearning/tensorrt/).

## Monitoring Performance

### Built-in Performance Monitor

Enable stats display:
- Press **T** to toggle stats overlay
- View in console output

### Benchmark Tool

Run comprehensive benchmark:
```bash
python tools/benchmark.py --duration 30
```

Output includes:
- FPS for each pipeline stage
- GPU usage and memory
- Bottleneck identification
- Optimization recommendations

### Real-time Monitoring

**nvidia-smi**:
```bash
nvidia-smi -l 1
```

**Task Manager**:
- Performance tab
- GPU section
- Monitor 3D usage, VRAM, temperature

### Profiling

For detailed profiling:
```yaml
performance:
  enable_profiling: true
```

## Troubleshooting Performance

### Issue: FPS Below 60

**Check Bottleneck**:
```bash
python tools/benchmark.py
```

Look for the slowest component:

1. **Inference is bottleneck** (most common):
   - Enable FP16: `ai_model.fp16: true`
   - Lower input resolution
   - Enable TensorRT
   - Reduce character target size

2. **Tracking is bottleneck**:
   - Lower tracking confidence: `tracking.min_tracking_confidence: 0.3`
   - Reduce smoothing: `tracking.smoothing: 0.3`

3. **Capture is bottleneck**:
   - Check webcam drivers
   - Try different USB port
   - Lower capture resolution

4. **Output is bottleneck**:
   - Disable one output (Spout or NDI)
   - Check OBS isn't using too much GPU

### Issue: High GPU Temperature

If GPU temperature exceeds 85°C:

1. **Improve cooling**:
   - Clean GPU fans
   - Improve case airflow
   - Consider GPU underclock

2. **Limit GPU usage**:
   ```yaml
   performance:
     target_fps: 50  # Lower target
   ```

3. **Frame limiting**:
   Add sleep in pipeline (reduces heat but lowers FPS)

### Issue: Stuttering

Causes:
1. **Frame queue overflow**:
   - Solution: Enable async pipeline

2. **Character switching lag**:
   - Solution: Enable preloading: `character.preload_all: true`

3. **Other apps using GPU**:
   - Close browser (especially Chrome)
   - Close Discord hardware acceleration
   - Check Task Manager for GPU usage

### Issue: Memory Leaks

If memory usage increases over time:

1. **Monitor memory**:
   ```bash
   nvidia-smi -l 1
   ```

2. **Restart application periodically**

3. **Disable character preloading**:
   ```yaml
   character:
     preload_all: false
   ```

## Advanced Optimizations

### Multi-Stream CUDA

For expert users, implement custom CUDA streams for overlapping operations.

### Model Quantization

Convert model to INT8 for even faster inference (requires model retraining).

### Custom Face Detection

Replace MediaPipe with faster face detector for specific use cases.

### Network Optimization

If using remote tracking:
- Use UDP instead of TCP
- Compress landmark data
- Implement prediction/interpolation

### OBS Optimization

To maximize performance with OBS running:

1. **OBS Settings**:
   - Encoder: NVENC (GPU)
   - Preset: Quality (not Max Quality)
   - Rate Control: CBR
   - Bitrate: 6000 Kbps

2. **GPU Allocation**:
   - OBS typically uses 10-20% GPU
   - Reserve 80%+ for animator

3. **Separate GPU (if available)**:
   - Use iGPU for OBS encoding
   - RTX 3080 purely for animation

### Batch Processing

For multiple characters:
```yaml
ai_model:
  batch_size: 2  # Process 2 characters at once
```

Only useful if:
- Multiple characters visible
- GPU has spare capacity
- Reduces per-character overhead

## Performance Checklist

Before going live:

- [ ] FP16 enabled
- [ ] Async pipeline enabled
- [ ] 720p input resolution
- [ ] Character preloading enabled
- [ ] Target FPS set to 60
- [ ] Latest GPU drivers installed
- [ ] Windows High Performance mode
- [ ] OBS NVENC encoder
- [ ] GPU temperature under 80°C
- [ ] Benchmark shows 60+ FPS

## Real-World Performance Tips

### For Streaming

- **Test before going live**: Run for 10+ minutes
- **Monitor temperature**: Keep below 80°C
- **Have backup plan**: Pre-recorded video if system crashes
- **Redundancy**: Test both Spout and NDI outputs

### For Recording

- **Higher quality possible**: Can lower FPS to 30
- **Increase resolution**: 1080p input acceptable
- **Post-processing**: Apply upscaling later

### For Collaboration

- **NDI over network**: Adds 5-10ms latency
- **Lower resolution for remote**: 720p output
- **Bandwidth**: 50+ Mbps recommended

---

**Questions?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or [open an issue](https://github.com/GilbertoBitt/stream-motion-animator/issues).
