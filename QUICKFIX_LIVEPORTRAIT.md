# ⚡ QUICK FIX - LivePortrait Not Animating

## 🔍 Problem
**Character displays but doesn't animate** = Mock model active

## ✅ Solution

### Option 1: Download Real Model (Full Animation)

```bash
# Step 1: Run download helper
python tools/download_liveportrait.py

# Step 2: Download from:
# https://github.com/KwaiVGI/LivePortrait

# Step 3: Extract .pth files to:
# models/liveportrait/

# Step 4: Install deps
pip install face-alignment imageio[ffmpeg]

# Step 5: Run
run.bat
```

### Option 2: Test with Mock (Current - Limited)

```bash
# Already working! Just run:
run.bat

# Animation is basic (head tilt, mouth scale only)
```

## 🧪 Test

```bash
python test_liveportrait.py
```

Should show if real model is installed.

## 📊 Quick Check

| Feature | Mock Model | Real Model |
|---------|-----------|------------|
| Character displays | ✅ Yes | ✅ Yes |
| Face tracking | ✅ Yes | ✅ Yes |
| **Animation** | ⚠️ Basic | ✅ Full |
| Head movement | ⚠️ Slight | ✅ Full |
| Mouth sync | ⚠️ Scale | ✅ Sync |
| Expressions | ❌ No | ✅ Yes |

## 🎯 Bottom Line

**Current:** Mock model = Character shows but barely moves

**Need:** Real LivePortrait model = Full facial animation

**Download:** https://github.com/KwaiVGI/LivePortrait

**Status:** ⚠️ Mock model active (real model not installed)

