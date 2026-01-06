"""
LivePortrait Model Downloader - Updated for correct HuggingFace structure

The model files are organized differently on HuggingFace.
This script uses the correct paths.
"""

import sys
import os
from pathlib import Path
import urllib.request
import ssl

print("="*70)
print("LIVEPORTRAIT MODEL DOWNLOADER - HUGGING FACE")
print("="*70)
print()

# Model files are in subdirectories on Hugging Face
# Correct structure: KwaiVGI/LivePortrait/tree/main/pretrained_weights/

HF_BASE = "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/pretrained_weights"

MODEL_FILES = {
    # LivePortrait model files in pretrained_weights folder
    "liveportrait/appearance_feature_extractor.pth": {
        "url": f"{HF_BASE}/liveportrait/appearance_feature_extractor.pth",
        "dest": "appearance_feature_extractor.pth",
        "size": "~200MB"
    },
    "liveportrait/motion_extractor.pth": {
        "url": f"{HF_BASE}/liveportrait/motion_extractor.pth",
        "dest": "motion_extractor.pth",
        "size": "~100MB"
    },
    "liveportrait/spade_generator.pth": {
        "url": f"{HF_BASE}/liveportrait/spade_generator.pth",
        "dest": "spade_generator.pth",
        "size": "~500MB"
    },
    "liveportrait/warping_module.pth": {
        "url": f"{HF_BASE}/liveportrait/warping_module.pth",
        "dest": "warping_module.pth",
        "size": "~200MB"
    },
    "liveportrait/stitching_retargeting_module.pth": {
        "url": f"{HF_BASE}/liveportrait/stitching_retargeting_module.pth",
        "dest": "stitching_retargeting_module.pth",
        "size": "~50MB"
    },
}

print("IMPORTANT: LivePortrait model requires manual download")
print()
print("The model is available at:")
print("https://huggingface.co/KwaiVGI/LivePortrait")
print()
print("Please follow these steps:")
print()
print("="*70)
print("MANUAL DOWNLOAD INSTRUCTIONS")
print("="*70)
print()
print("1. Visit: https://huggingface.co/KwaiVGI/LivePortrait")
print()
print("2. Navigate to: Files and versions tab")
print()
print("3. Go to: pretrained_weights/liveportrait/")
print()
print("4. Download these files:")
print("   - appearance_feature_extractor.pth")
print("   - motion_extractor.pth")
print("   - spade_generator.pth")
print("   - warping_module.pth")
print("   - stitching_retargeting_module.pth")
print()
print("5. Place them in:")
model_dir = Path("models/liveportrait")
model_dir.mkdir(parents=True, exist_ok=True)
print(f"   {model_dir.absolute()}")
print()
print("="*70)
print()

print("Alternative: Use git-lfs to clone the repository")
print()
print("  git lfs install")
print("  git clone https://huggingface.co/KwaiVGI/LivePortrait")
print("  cd LivePortrait/pretrained_weights/liveportrait")
print(f"  copy *.pth {model_dir.absolute()}")
print()
print("="*70)
print()

# Check if user wants to open browser
print("Would you like to:")
print("  1. Open Hugging Face page in browser")
print("  2. See detailed file list")
print("  3. Check current installation status")
print("  4. Exit")
print()
choice = input("Enter choice (1-4): ").strip()

if choice == '1':
    import webbrowser
    webbrowser.open("https://huggingface.co/KwaiVGI/LivePortrait/tree/main/pretrained_weights/liveportrait")
    print()
    print("✓ Opened browser to model files")
    print()
    print("Download the .pth files and place them in:")
    print(f"  {model_dir.absolute()}")

elif choice == '2':
    print()
    print("="*70)
    print("REQUIRED FILES")
    print("="*70)
    print()
    print("Download from:")
    print("https://huggingface.co/KwaiVGI/LivePortrait/tree/main/pretrained_weights/liveportrait")
    print()
    for key, info in MODEL_FILES.items():
        print(f"File: {info['dest']}")
        print(f"  Size: {info['size']}")
        print(f"  URL: {info['url']}")
        print()

elif choice == '3':
    print()
    print("="*70)
    print("INSTALLATION STATUS")
    print("="*70)
    print()

    pth_files = list(model_dir.glob("*.pth"))
    if pth_files:
        print(f"✓ Found {len(pth_files)} model file(s):")
        for f in pth_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  ✓ {f.name} ({size_mb:.1f}MB)")
        print()

        required = ["appearance_feature_extractor.pth", "motion_extractor.pth",
                   "spade_generator.pth", "warping_module.pth"]
        missing = [r for r in required if not (model_dir / r).exists()]

        if missing:
            print("⚠ Still missing:")
            for m in missing:
                print(f"  - {m}")
        else:
            print("✅ All required files present!")
            print()
            print("Next steps:")
            print("  1. Run: python test_liveportrait.py")
            print("  2. Run: run.bat")
    else:
        print("❌ No model files found")
        print()
        print("Please download from:")
        print("https://huggingface.co/KwaiVGI/LivePortrait/tree/main/pretrained_weights/liveportrait")
        print()
        print(f"And place in: {model_dir.absolute()}")

print()
print("="*70)
print()

