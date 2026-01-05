# Troubleshooting Guide

## Performance Issues

### Low FPS (Below 30 FPS)

**Symptoms:**
- Choppy animation
- High latency
- Dropped frames

**Solutions:**

1. **Reduce Batch Size**
   ```yaml
   animation:
     batch_size: 1  # Start with 1 and increase if possible
   ```

2. **Enable FP16 Precision**
   ```yaml
   animation:
     use_half_precision: true
     inference_precision: fp16
   ```

3. **Lower Input Resolution**
   ```yaml
   animation:
     input_resolution: [256, 256]  # Down from 512x512
   ```

4. **Reduce Capture Resolution**
   ```bash
   python main.py --image portrait.jpg --resolution 1280 720
   ```

5. **Check GPU Usage**
   ```bash
   # In another terminal
   watch -n 1 nvidia-smi
   ```
   - If GPU usage is low (<50%), there may be a CPU bottleneck
   - If GPU usage is maxed out (>95%), reduce batch size or resolution

6. **Update GPU Drivers**
   - NVIDIA: Visit nvidia.com/drivers
   - Ensure CUDA version matches PyTorch installation

7. **Close Background Applications**
   - Close other GPU-intensive apps
   - Disable browser hardware acceleration
   - Check Task Manager for GPU usage

### High Latency (>50ms per frame)

**Symptoms:**
- Noticeable delay between movement and animation
- FPS might be acceptable but feels sluggish

**Solutions:**

1. **Enable Frame Skipping**
   ```yaml
   performance:
     frame_skip_enabled: true
     frame_skip_threshold: 0.9
   ```

2. **Reduce Prefetch Buffer**
   ```yaml
   performance:
     prefetch_frames: 1  # Reduce from default
   ```

3. **Disable Smoothing**
   ```yaml
   tracking:
     smoothing_enabled: false
   ```

4. **Check Pipeline Stages**
   - Press 'R' during runtime to see stage timings
   - Identify the slowest stage and optimize it

### Intermittent Freezing

**Causes:**
- GPU memory fragmentation
- Thermal throttling
- Background processes

**Solutions:**

1. **Enable Cache Clearing**
   ```yaml
   performance:
     clear_cache_interval: 50  # Clear GPU cache every 50 frames
   ```

2. **Monitor Temperature**
   ```bash
   nvidia-smi -l 1
   ```
   - If GPU temp >85°C, improve cooling
   - Clean dust from GPU fans
   - Improve case airflow

3. **Reduce GPU Memory Usage**
   ```yaml
   performance:
     gpu_memory_fraction: 0.7  # Use only 70% of VRAM
   ```

## Installation Issues

### "CUDA not available"

**Solution 1: Verify CUDA Installation**
```bash
nvcc --version
nvidia-smi
```

**Solution 2: Reinstall PyTorch with CUDA**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Solution 3: Check CUDA Version Compatibility**
- PyTorch CUDA version must match system CUDA
- Visit: https://pytorch.org/get-started/locally/

### "ModuleNotFoundError: No module named 'mediapipe'"

**Solution:**
```bash
pip install mediapipe
```

If installation fails:
```bash
pip install --upgrade pip setuptools wheel
pip install mediapipe
```

### "Failed to load model"

**Possible Causes:**
- Model file not found
- Corrupted model file
- Incompatible model format

**Solutions:**

1. **Verify Model Path**
   ```yaml
   animation:
     model_path: models/live_portrait.pth  # Check this path exists
   ```

2. **Re-download Model**
   ```bash
   python scripts/download_models.py --model live-portrait --force
   ```

3. **Check Model File**
   ```bash
   ls -lh models/
   ```

## Camera Issues

### "Failed to open camera device"

**Solution 1: List Available Cameras**
```python
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
```

**Solution 2: Specify Camera ID**
```bash
python main.py --image portrait.jpg --camera 1
```

**Solution 3: Check Permissions (Linux)**
```bash
sudo usermod -a -G video $USER
# Log out and log back in
```

### Camera Lag/Stuttering

**Solutions:**

1. **Reduce Capture Resolution**
   ```yaml
   capture:
     width: 1280
     height: 720
   ```

2. **Use DirectShow Backend (Windows)**
   ```yaml
   capture:
     backend: dshow
   ```

3. **Use V4L2 Backend (Linux)**
   ```yaml
   capture:
     backend: v4l2
   ```

## Output Issues

### Spout Not Working (Windows)

**Error:** "Spout is not available"

**Solutions:**

1. **Install SpoutGL**
   ```bash
   pip install SpoutGL
   ```

2. **Verify Windows Version**
   - Spout requires Windows 7 or higher
   - Requires OpenGL 4.3+

3. **Check Spout Installation**
   - Download from: https://spout.zeal.co/
   - Install system-wide Spout libraries

4. **Test in OBS**
   - Install OBS-Spout2-Plugin
   - Add Spout2 Capture source
   - Select "AI_Avatar" sender

### NDI Not Working

**Error:** "NDI is not available"

**Solutions:**

1. **Install NDI SDK**
   - Download from: https://www.ndi.tv/sdk/
   - Run installer
   - Accept license agreement

2. **Install ndi-python**
   ```bash
   pip install ndi-python
   ```

3. **Set Environment Variable (if needed)**
   ```bash
   # Linux/Mac
   export NDI_RUNTIME_DIR=/usr/local/lib
   
   # Windows
   set NDI_RUNTIME_DIR=C:\Program Files\NDI\NDI 5 Runtime
   ```

4. **Restart Application**
   - NDI requires full restart after SDK installation

5. **Check Network Settings**
   - Ensure NDI is allowed through firewall
   - Check that receiver is on same network

## Face Tracking Issues

### "No face detected"

**Solutions:**

1. **Improve Lighting**
   - Use bright, even lighting
   - Avoid backlighting
   - Position face at camera level

2. **Lower Detection Threshold**
   ```yaml
   tracking:
     min_detection_confidence: 0.3  # Down from 0.5
     min_tracking_confidence: 0.3
   ```

3. **Check Camera Position**
   - Face should be clearly visible
   - Look directly at camera
   - Remove glasses if they cause glare

4. **Test with Different Images**
   - Use a clearer portrait photo
   - Ensure face is well-lit in source image

### Jittery/Unstable Tracking

**Solutions:**

1. **Enable Smoothing**
   ```yaml
   tracking:
     smoothing_enabled: true
     smoothing_window: 7  # Increase for more smoothing
   ```

2. **Increase Tracking Confidence**
   ```yaml
   tracking:
     min_tracking_confidence: 0.7  # Up from 0.5
   ```

3. **Refine Landmarks**
   ```yaml
   tracking:
     refine_landmarks: true
   ```

4. **Improve Lighting Consistency**
   - Use consistent, stable lighting
   - Avoid moving shadows

## Memory Issues

### "CUDA out of memory"

**Solutions:**

1. **Reduce Batch Size**
   ```yaml
   animation:
     batch_size: 1
   ```

2. **Use FP16**
   ```yaml
   animation:
     use_half_precision: true
   ```

3. **Clear GPU Cache**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Reduce Resolution**
   ```yaml
   animation:
     input_resolution: [256, 256]
   ```

5. **Close Other GPU Applications**
   - Check nvidia-smi for other processes
   - Kill unnecessary GPU processes

### High CPU Memory Usage

**Solutions:**

1. **Reduce Frame Buffer**
   ```yaml
   performance:
     prefetch_frames: 1
   ```

2. **Disable Metrics**
   ```yaml
   metrics:
     enabled: false
   ```

3. **Monitor Memory**
   ```bash
   python main.py --image portrait.jpg &
   top -p $(pgrep -f main.py)
   ```

## Error Messages

### "RuntimeError: Expected all tensors to be on the same device"

**Cause:** Model and input tensors on different devices

**Solution:**
Ensure GPU ID is correctly set:
```yaml
animation:
  gpu_id: 0
```

### "cv2.error: OpenCV(4.x.x) ... Camera error"

**Cause:** Camera disconnected or in use

**Solutions:**
1. Reconnect camera
2. Close other apps using camera (Zoom, Skype, etc.)
3. Restart computer

### "FileNotFoundError: Image not found"

**Solution:**
Use absolute paths:
```bash
python main.py --image /full/path/to/portrait.jpg
```

## Getting More Help

### Enable Debug Logging

```bash
python main.py --image portrait.jpg --log-level DEBUG
```

### Check Logs

```bash
cat logs/animator.log
```

### Report Issues

When reporting issues, include:
1. Full error message and stack trace
2. System info (GPU model, OS, Python version)
3. Configuration file
4. Log file
5. Steps to reproduce

### Community Support

- GitHub Issues: https://github.com/GilbertoBitt/stream-motion-animator/issues
- Discussions: https://github.com/GilbertoBitt/stream-motion-animator/discussions
