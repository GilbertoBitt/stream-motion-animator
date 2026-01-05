# Troubleshooting Guide

Solutions to common issues and problems with Stream Motion Animator.

## Table of Contents

- [Installation Issues](#installation-issues)
- [GPU and CUDA Issues](#gpu-and-cuda-issues)
- [Webcam Issues](#webcam-issues)
- [Model Issues](#model-issues)
- [Performance Issues](#performance-issues)
- [Output Issues (Spout/NDI)](#output-issues-spoutndi)
- [Character Issues](#character-issues)
- [Application Crashes](#application-crashes)
- [OBS Integration Issues](#obs-integration-issues)
- [Error Messages](#error-messages)

## Installation Issues

### "pip not found" or "python not found"

**Cause**: Python not installed or not in PATH

**Solution**:
1. Download Python from [python.org](https://www.python.org)
2. During installation, check "Add Python to PATH"
3. Restart terminal/command prompt
4. Verify: `python --version`

### "No module named 'xyz'"

**Cause**: Dependencies not installed

**Solution**:
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt

# If specific module missing
pip install module-name
```

### "Virtual environment activation failed"

**Cause**: Execution policy on Windows

**Solution**:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
venv\Scripts\activate
```

### "Permission denied" errors

**Cause**: Insufficient permissions

**Solution**:
- Run terminal as Administrator (Windows)
- Use `sudo` for system-wide installs (Linux)
- Install in user directory: `pip install --user`

## GPU and CUDA Issues

### "CUDA not available" / torch.cuda.is_available() returns False

**Cause**: PyTorch not installed with CUDA support

**Solution**:
```bash
# Uninstall PyTorch
pip uninstall torch torchvision

# Reinstall with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### "CUDA driver version insufficient"

**Cause**: Outdated NVIDIA drivers

**Solution**:
1. Check current version: `nvidia-smi`
2. Download latest drivers: [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)
3. Install and restart
4. Verify: `nvidia-smi`

### "RuntimeError: CUDA out of memory"

**Cause**: GPU VRAM exhausted

**Solutions**:

1. **Reduce character target size**:
   ```yaml
   character:
     target_size: [512, 512]  # Lower from 1024
   ```

2. **Disable character preloading**:
   ```yaml
   character:
     preload_all: false
   ```

3. **Lower resolution**:
   ```yaml
   video:
     width: 1280
     height: 720
   output:
     width: 1280
     height: 720
   ```

4. **Close other GPU applications**:
   - Close browsers with hardware acceleration
   - Close games
   - Check Task Manager → Performance → GPU

5. **Restart application** (clears memory leaks)

### GPU temperature too high (>85°C)

**Solutions**:
1. Improve case ventilation
2. Clean GPU fans
3. Lower FPS target:
   ```yaml
   performance:
     target_fps: 50
   ```
4. Enable V-Sync to limit FPS
5. Underclock GPU slightly

### Wrong GPU selected (multi-GPU systems)

**Cause**: System using integrated GPU instead of NVIDIA

**Solution**:
1. **Windows Graphics Settings**:
   - Settings → System → Display → Graphics Settings
   - Add `python.exe` from your venv
   - Set to "High Performance"

2. **NVIDIA Control Panel**:
   - Manage 3D Settings
   - Program Settings → Add python.exe
   - Select "High-performance NVIDIA processor"

3. **Force GPU in config**:
   ```python
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force first GPU
   ```

## Webcam Issues

### "Failed to open webcam"

**Solutions**:

1. **Check webcam connection**:
   - USB properly connected
   - Try different USB port
   - Test with Camera app (Windows)

2. **Close other applications**:
   - Zoom, Teams, Skype
   - Discord
   - Other webcam apps

3. **Try different camera index**:
   ```yaml
   video:
     source: 1  # Try 1, 2, 3...
   ```

4. **Update webcam drivers**:
   - Device Manager → Cameras
   - Right-click → Update driver

### Webcam lag or low FPS

**Solutions**:

1. **Lower capture resolution**:
   ```yaml
   video:
     width: 640
     height: 480
   ```

2. **Reduce capture FPS**:
   ```yaml
   video:
     fps: 30
   ```

3. **USB bandwidth**:
   - Use USB 3.0 port (blue port)
   - Don't share USB hub with other devices

4. **Webcam settings**:
   - Disable auto-exposure
   - Disable auto-focus
   - Reduce quality in webcam software

### Image upside down or mirrored

**Solution**:
Add flip in code or use OpenCV:
```python
frame = cv2.flip(frame, 1)  # Horizontal flip
frame = cv2.flip(frame, 0)  # Vertical flip
```

## Model Issues

### "Model not found"

**Cause**: Model weights not downloaded

**Solution**:
```bash
python tools/download_models.py
```

Verify files exist:
```bash
dir models\liveportrait  # Windows
ls models/liveportrait   # Linux/Mac
```

### Model download fails

**Solutions**:

1. **Check internet connection**

2. **Manual download**:
   - Visit [Live Portrait Releases](https://github.com/KwaiVGI/LivePortrait/releases)
   - Download model files
   - Extract to `models/liveportrait/`

3. **Use VPN** (if region blocked)

4. **Check disk space**:
   - Models need ~1GB free space
   - Verify: `df -h` (Linux/Mac) or File Explorer (Windows)

### "Model loading failed"

**Solutions**:

1. **Re-download model**:
   ```bash
   python tools/download_models.py --force
   ```

2. **Check file integrity**:
   - Corrupted download
   - Incomplete extraction

3. **Verify PyTorch version**:
   ```bash
   pip list | grep torch
   ```

## Performance Issues

### Low FPS (<30 FPS)

**Diagnosis**:
```bash
python tools/benchmark.py
```

**Solutions by bottleneck**:

1. **Inference bottleneck** (most common):
   ```yaml
   ai_model:
     fp16: true  # Enable FP16
   video:
     width: 1280  # Lower resolution
     height: 720
   ```

2. **Tracking bottleneck**:
   ```yaml
   tracking:
     min_tracking_confidence: 0.3  # Lower
     smoothing: 0.3
   ```

3. **System bottleneck**:
   - Close background apps
   - Windows High Performance mode
   - Update GPU drivers

### Stuttering or frame drops

**Solutions**:

1. **Enable async pipeline**:
   ```yaml
   performance:
     async_pipeline: true
   ```

2. **Increase queue sizes** (in code)

3. **Reduce frame skip threshold**:
   ```yaml
   performance:
     frame_skip_threshold: 30
   ```

### High latency (>50ms)

**Solutions**:

1. **Reduce smoothing**:
   ```yaml
   tracking:
     smoothing: 0.2  # Lower = more responsive
   ```

2. **Disable unnecessary features**

3. **Optimize pipeline** (reduce queue sizes)

## Output Issues (Spout/NDI)

### Spout not working in OBS

**Solutions**:

1. **Install Spout2 plugin for OBS**:
   - Download: [OBS Spout2 Plugin](https://github.com/Off-World-Live/obs-spout2-plugin/releases)
   - Extract to OBS plugins folder
   - Restart OBS

2. **Check Spout sender name**:
   ```yaml
   output:
     spout_name: "StreamMotionAnimator"
   ```
   Must match name in OBS source

3. **Python SpoutGL installed**:
   ```bash
   pip install SpoutGL
   ```

4. **Windows only**: Spout only works on Windows

5. **OBS version**: Use OBS 28.0+

### NDI not appearing in OBS

**Solutions**:

1. **Install NDI plugin for OBS**:
   - Download: [OBS NDI Plugin](https://github.com/obs-ndi/obs-ndi/releases)
   - Run installer
   - Restart OBS

2. **Install NDI Runtime**:
   - Download: [NDI Tools](https://ndi.tv/tools/)
   - Install NDI Runtime
   - Restart computer

3. **Check NDI sender name**:
   ```yaml
   output:
     ndi_name: "Stream Motion Animator"
   ```

4. **Firewall**: Allow NDI through firewall

5. **Network**: NDI requires proper network configuration

### No output / black screen

**Solutions**:

1. **Enable output**:
   ```yaml
   output:
     spout_enabled: true
     ndi_enabled: true
   ```

2. **Check application is running**:
   - Preview window shows animation
   - No error messages in console

3. **Toggle output** (press S for Spout, N for NDI)

4. **Restart OBS** after connecting source

### Transparent background not working

**Solutions**:

1. **Use Spout** (better transparency support)

2. **In OBS**:
   - Right-click source → Filters
   - Add "Color Key" or "Chroma Key"
   - Select background color

3. **Use RGBA character images**:
   ```yaml
   character:
     auto_crop: true  # Helps with transparency
   ```

## Character Issues

### "No character images found"

**Cause**: No images in characters directory

**Solution**:
1. Add images to `assets/characters/`
2. Supported formats: PNG, JPG, JPEG
3. Verify path in config:
   ```yaml
   character:
     images_path: "assets/characters/"
   ```

### "No face detected"

**Cause**: Face not visible or not frontal

**Solutions**:
1. Use frontal-facing images
2. Ensure face is clearly visible
3. Test image:
   ```bash
   python tools/test_character.py assets/characters/image.png
   ```
4. Disable auto-crop if it's failing:
   ```yaml
   character:
     auto_crop: false
   ```

### Character switching is slow

**Solutions**:

1. **Enable preloading**:
   ```yaml
   character:
     preload_all: true
   ```

2. **Compress images** (optimize PNGs)

3. **Limit number of characters** (5-10 max)

4. **Use lower resolution**:
   ```yaml
   character:
     target_size: [512, 512]
   ```

### Animation looks wrong

**Solutions**:

1. **Use higher quality image**:
   - Minimum 512×512
   - Recommended 1024×1024

2. **Use neutral expression**

3. **Ensure frontal face**

4. **Test with different character**

## Application Crashes

### Crash on startup

**Check**:
1. All dependencies installed
2. Config file valid YAML
3. GPU drivers up to date

**Debug**:
```bash
python src/main.py 2>&1 | tee error.log
```

### Crash during operation

**Common causes**:
1. Out of memory → Reduce settings
2. Model error → Re-download models
3. Webcam disconnected → Check connection

**Recovery**:
- Save config before testing
- Start with minimal config
- Add features one by one

### Python crashes without error

**Solutions**:
1. Update Python: `python -m pip install --upgrade pip`
2. Reinstall dependencies
3. Check antivirus (may block)
4. Run as Administrator

## OBS Integration Issues

### OBS preview is black

**Solutions**:
1. Restart OBS
2. Remove and re-add source
3. Check Spout/NDI source settings
4. Verify application is running

### OBS recording/streaming has low FPS

**Solutions**:
1. Use NVENC encoder (GPU)
2. Lower OBS output resolution
3. Reduce bitrate
4. Check GPU usage (shouldn't exceed 90%)

### Audio/video desync

**Solutions**:
1. Add delay to audio
2. Use constant framerate
3. Match output FPS to stream FPS

## Error Messages

### "ImportError: DLL load failed"

**Cause**: Missing Visual C++ Redistributable

**Solution**:
Download and install: [VC++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

### "FileNotFoundError: config.yaml"

**Solution**:
```bash
# Create from template
cp assets/config.yaml.example assets/config.yaml
```

### "yaml.scanner.ScannerError"

**Cause**: Invalid YAML syntax in config

**Solution**:
1. Check YAML syntax (indentation, colons)
2. Use YAML validator: [YAML Lint](http://www.yamllint.com/)
3. Restore from backup

### "cv2.error: OpenCV"

**Cause**: OpenCV issue with video capture

**Solution**:
```bash
pip uninstall opencv-python
pip install opencv-python==4.8.0.76
```

## Still Having Issues?

### Get Detailed Logs

```bash
python src/main.py --verbose 2>&1 | tee debug.log
```

### System Information

Collect for bug reports:
```bash
python tools/benchmark.py > system_info.txt
```

### Report a Bug

When opening an issue, include:
- Error message (full traceback)
- System info (GPU, OS, Python version)
- Config file (remove sensitive info)
- Steps to reproduce
- Expected vs actual behavior

**GitHub Issues**: [Report Issue](https://github.com/GilbertoBitt/stream-motion-animator/issues)

---

**Can't find your issue?** Check [GitHub Discussions](https://github.com/GilbertoBitt/stream-motion-animator/discussions) or open a new issue.
