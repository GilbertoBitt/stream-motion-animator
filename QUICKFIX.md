# Quick Fix for Python 3.13

## The Problem
MediaPipe 0.10.9 doesn't support Python 3.13 (only up to 3.11)

## The Solution (3 Commands)

```bash
# 1. Install compatible MediaPipe
pip install mediapipe>=0.10.30 --user

# 2. Test it works
python test_mediapipe_setup.py

# 3. Run your app
python src/main.py
```

## What Changed

- `requirements.txt`: `mediapipe==0.10.9` → `mediapipe>=0.10.30`
- `motion_tracker.py`: Added Tasks API support
- Auto-downloads model on first run (8MB, one-time)

## If Commands Hang

```powershell
Stop-Process -Name python -Force
```

Then try again.

## Files Created

- `test_mediapipe_setup.py` - Test script
- `PYTHON313_SETUP.md` - Full guide  
- `FINAL_SOLUTION.md` - What was fixed

## That's It!

The code now works with Python 3.13 + MediaPipe 0.10.30+.

