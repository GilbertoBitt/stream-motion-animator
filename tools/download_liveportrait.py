"""
LivePortrait Model Downloader

This script helps you download the real LivePortrait model files.

NOTE: LivePortrait is a large model (~2GB). Make sure you have:
- Good internet connection
- At least 5GB free disk space
- CUDA-capable GPU (NVIDIA)
"""

import sys
import os
from pathlib import Path
import urllib.request
import zipfile

print("="*70)
print("LIVEPORTRAIT MODEL DOWNLOADER")
print("="*70)
print()

# Check model directory
model_dir = Path("models/liveportrait")
if not model_dir.exists():
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created model directory: {model_dir}")

# Check if model already exists
existing_files = list(model_dir.glob("*.pth"))
if len(existing_files) > 0:
    print(f"⚠ Found {len(existing_files)} existing model file(s)")
    print("  Do you want to re-download? (y/n)")
    response = input("> ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)

print()
print("="*70)
print("IMPORTANT INFORMATION")
print("="*70)
print()
print("LivePortrait model is NOT freely available for automatic download.")
print()
print("You need to:")
print()
print("1. Visit the official repository:")
print("   https://github.com/KwaiVGI/LivePortrait")
print()
print("2. Check the releases page:")
print("   https://github.com/KwaiVGI/LivePortrait/releases")
print()
print("3. Or use Hugging Face:")
print("   https://huggingface.co/KwaiVGI/LivePortrait")
print()
print("4. Download the model files manually")
print()
print("5. Extract to:")
print(f"   {model_dir.absolute()}")
print()
print("="*70)
print()
print("Required files:")
print("  - appearance_feature_extractor.pth")
print("  - motion_extractor.pth")
print("  - spade_generator.pth")
print("  - warping_module.pth")
print()
print("Alternative:")
print("  - Download: first-order-model (easier to set up)")
print("  - Or use: enhanced mock model (basic animation)")
print()
print("="*70)
print()

print("Would you like to:")
print("  1. Open LivePortrait GitHub in browser")
print("  2. Open Hugging Face page in browser")
print("  3. Use enhanced mock model (no download needed)")
print("  4. Cancel")
print()
choice = input("Enter choice (1-4): ").strip()

if choice == '1':
    import webbrowser
    webbrowser.open("https://github.com/KwaiVGI/LivePortrait")
    print("✓ Opened GitHub in browser")
    print()
    print("After downloading:")
    print(f"  Extract files to: {model_dir.absolute()}")
    print("  Then run: run.bat")

elif choice == '2':
    import webbrowser
    webbrowser.open("https://huggingface.co/KwaiVGI/LivePortrait/tree/main")
    print("✓ Opened Hugging Face in browser")
    print()
    print("After downloading:")
    print(f"  Extract files to: {model_dir.absolute()}")
    print("  Then run: run.bat")

elif choice == '3':
    print()
    print("Enhanced mock model will provide basic animation:")
    print("  - Head rotation tracking")
    print("  - Mouth opening")
    print("  - Eye blinking (if detected)")
    print()
    print("To enable:")
    print("  1. The mock model is already active")
    print("  2. Just run: run.bat")
    print()
    print("Note: This is NOT as good as real LivePortrait,")
    print("      but works without downloading anything.")

else:
    print("Cancelled.")

print()

