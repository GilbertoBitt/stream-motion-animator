# Python 3.13 Setup Guide

## Issue Summary

You're using **Python 3.13**, which has limited MediaPipe support:
- MediaPipe 0.10.9 (with legacy API) only supports Python ≤ 3.11
- MediaPipe 0.10.30+ (with Tasks API) supports Python 3.13

## Solution: Use MediaPipe 0.10.30+ with Tasks API

### Step 1: Install MediaPipe 0.10.30+

```bash
pip install mediapipe>=0.10.30 --user
```

### Step 2: Run Test Script

```bash
cd G:\stream-motion-animator
python test_mediapipe_setup.py
```

This will:
1. Verify MediaPipe is installed
2. Check which API is available (legacy or Tasks)
3. Download the face landmarker model automatically (if Tasks API)
4. Test MotionTracker initialization

### Step 3: Run the Application

```bash
python src/main.py
```

## What Was Changed

### 1. `requirements.txt`
- Updated MediaPipe to `>=0.10.30` for Python 3.13 compatibility

### 2. `src/motion_tracker.py`
- Added support for both legacy and Tasks API
- Auto-downloads face_landmarker.task model (8.2MB)
- Model saved to: `models/mediapipe/face_landmarker.task`

## Troubleshooting

### Error: "MediaPipe is required"

**Solution**: Install MediaPipe
```bash
pip install mediapipe>=0.10.30 --user
```

### Error: "No module named 'mediapipe.tasks'"

**Solution**: Upgrade MediaPipe
```bash
pip install --upgrade mediapipe --user
```

### Error: "Model file not found"

The model should auto-download. If it fails:

1. **Manual Download**:
   - URL: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   - Save to: `G:\stream-motion-animator\models\mediapipe\face_landmarker.task`

2. **Create directory**:
```bash
mkdir -p models/mediapipe
```

3. **Download with PowerShell**:
```powershell
$url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
$output = "models\mediapipe\face_landmarker.task"
New-Item -ItemType Directory -Force -Path "models\mediapipe"
Invoke-WebRequest -Uri $url -OutFile $output
```

### Commands Timing Out / Hanging

If pip or python commands hang:

1. **Kill hanging processes**:
```powershell
Stop-Process -Name python -Force
```

2. **Use explicit Python path**:
```powershell
# Find Python
Get-Command python | Select-Object -ExpandProperty Source

# Use full path
C:\Users\gilbe\AppData\Local\Programs\Python\Python313\python.exe -m pip install mediapipe>=0.10.30 --user
```

3. **Check if MediaPipe is already installed**:
```powershell
pip list | Select-String mediapipe
```

## Alternative: Use Python 3.11

If you continue having issues with Python 3.13, you can:

1. Install Python 3.11
2. Create a new virtual environment
3. Use MediaPipe 0.10.9 with legacy API

```bash
# Install Python 3.11 from python.org
# Then:
py -3.11 -m venv .venv311
.\.venv311\Scripts\activate
pip install mediapipe==0.10.9
```

## Expected Behavior

### With Tasks API (0.10.30+)
```
MediaPipe version: 0.10.30
Using Tasks API
Downloading MediaPipe face landmarker model...
Model downloaded to models/mediapipe/face_landmarker.task
MediaPipe Face Landmarker initialized (Tasks API)
```

### With Legacy API (0.10.9-0.10.14)
```
MediaPipe version: 0.10.9
Using MediaPipe legacy solutions API
MediaPipe Face Mesh initialized (legacy API)
```

## Performance Notes

- **Tasks API (0.10.30+)**: Slightly slower first-time setup (model download)
- **Legacy API (0.10.9)**: Faster setup, models bundled
- **Runtime performance**: Similar for both APIs

## Next Steps

1. Run `python test_mediapipe_setup.py` to verify setup
2. If successful, run `python src/main.py`
3. If you see model downloading, wait for it to complete (~8MB)
4. The model is cached for future runs

## Support

If you're still having issues:

1. Check Python version: `python --version`
2. Check MediaPipe version: `pip show mediapipe`
3. Check installed location: `pip show -f mediapipe`
4. Try reinstalling: `pip install --force-reinstall mediapipe>=0.10.30 --user`

---

**Summary**: The code is now compatible with both MediaPipe 0.10.9 (Python ≤3.11) and 0.10.30+ (Python 3.13). It will automatically detect which API is available and use the appropriate one.

