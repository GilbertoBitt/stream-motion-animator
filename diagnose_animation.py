"""
LivePortrait Animation Diagnostic Tool

This checks why the animation isn't working like LivePortrait examples.
"""

import sys
import os
sys.path.insert(0, 'src')

import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("="*70)
print("LIVEPORTRAIT ANIMATION DIAGNOSTIC")
print("="*70)
print()

# Test 1: Check model loading
print("[1/5] Checking model files...")
model_dir = Path("models/liveportrait")
pth_files = list(model_dir.glob("*.pth"))
print(f"  Found {len(pth_files)} .pth files:")
for f in pth_files:
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"    - {f.name} ({size_mb:.1f}MB)")

if len(pth_files) < 4:
    print(f"  ✗ ERROR: Need at least 4 model files, found {len(pth_files)}")
    print("  Download from: https://huggingface.co/KwaiVGI/LivePortrait")
    sys.exit(1)

# Test 2: Load LivePortrait inference
print()
print("[2/5] Loading LivePortrait models...")
try:
    from liveportrait_loader import LivePortraitInference

    lp = LivePortraitInference(model_dir, device="cuda", fp16=True)
    success = lp.load_models()

    if success:
        print("  ✓ Models loaded successfully")
        print(f"    - Appearance extractor: {type(lp.appearance_extractor).__name__}")
        print(f"    - Motion extractor: {type(lp.motion_extractor).__name__}")
        print(f"    - Generator: {type(lp.generator).__name__}")
        print(f"    - Warping module: {type(lp.warping_module).__name__}")
    else:
        print("  ✗ Failed to load models")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check if models are actual neural networks or just state dicts
print()
print("[3/5] Checking model types...")

def check_model_type(model, name):
    if isinstance(model, dict):
        print(f"  {name}: State dict ({len(model)} keys)")
        print(f"    Sample keys: {list(model.keys())[:3]}")
        return "state_dict"
    elif hasattr(model, 'forward'):
        print(f"  {name}: Neural network model ✓")
        return "model"
    else:
        print(f"  {name}: Unknown type ({type(model)})")
        return "unknown"

app_type = check_model_type(lp.appearance_extractor, "Appearance extractor")
mot_type = check_model_type(lp.motion_extractor, "Motion extractor")
gen_type = check_model_type(lp.generator, "Generator")
warp_type = check_model_type(lp.warping_module, "Warping module")

# Test 4: Try actual inference
print()
print("[4/5] Testing inference...")

if app_type == "state_dict":
    print("  ⚠ Models are state dicts, not complete models")
    print("  ⚠ Need model architecture to use them properly")
    print()
    print("  PROBLEM IDENTIFIED:")
    print("  ================")
    print("  The .pth files contain only trained weights (state dicts),")
    print("  NOT the complete model architecture.")
    print()
    print("  This is why animation doesn't work like LivePortrait examples.")
    print("  You need the actual LivePortrait source code to:")
    print("    1. Define model architectures")
    print("    2. Load weights into models")
    print("    3. Run proper inference")
    print()
    print("  Current status: Using MOCK animation (simple transforms)")
else:
    print("  ✓ Models are complete neural networks")
    print("  Testing inference pipeline...")

    # Create test images
    source = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    driving = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    try:
        appearance = lp.extract_appearance_features(source)
        motion = lp.extract_motion(driving)
        output = lp.generate_frame(appearance, motion)
        print(f"  ✓ Inference successful: {output.shape}")
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")

# Test 5: Check what the main app is actually using
print()
print("[5/5] Checking main application...")
try:
    from models.liveportrait_model import LivePortraitModel

    model = LivePortraitModel(model_dir, device="cuda", fp16=True)
    model.load_model()

    print(f"  Model loaded: {model.is_model_loaded}")
    print(f"  Real model active: {model.is_real_model}")
    print(f"  Model type: {type(model.model).__name__}")

    if not model.is_real_model:
        print()
        print("  ⚠ WARNING: Application is using MOCK model!")
        print("  ⚠ This explains why animation doesn't work properly")
except Exception as e:
    print(f"  ✗ Error checking main app: {e}")

# Summary
print()
print("="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)
print()

if app_type == "state_dict":
    print("❌ ISSUE FOUND: Models are state dicts without architecture")
    print()
    print("Why animation doesn't work like LivePortrait examples:")
    print("  1. Downloaded files contain only weights, not model code")
    print("  2. Need LivePortrait source code for model architecture")
    print("  3. Current app uses simple transforms (mock model)")
    print("  4. Real LivePortrait needs proper architecture + weights")
    print()
    print("SOLUTIONS:")
    print()
    print("Option 1 (Recommended): Clone LivePortrait repository")
    print("  git clone https://github.com/KwaiVGI/LivePortrait")
    print("  cd LivePortrait")
    print("  pip install -r requirements.txt")
    print("  # Then integrate their inference code")
    print()
    print("Option 2: Use ONNX models (if available)")
    print("  # Check if LivePortrait provides ONNX models")
    print("  # ONNX includes architecture + weights")
    print()
    print("Option 3: Implement model architectures")
    print("  # Reverse-engineer from paper/code")
    print("  # Define PyTorch models")
    print("  # Load state dicts into models")
else:
    print("✓ Models appear to be complete")
    print("  Check application logs for other issues")

print()
print("="*70)

