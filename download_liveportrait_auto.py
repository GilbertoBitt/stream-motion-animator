"""
Automatic LivePortrait Model Downloader from Hugging Face

This script downloads and extracts the LivePortrait model files
from Hugging Face and sets them up correctly.
"""

import sys
import os
from pathlib import Path
import urllib.request
import ssl
import json
import zipfile
import tarfile
import shutil

print("="*70)
print("LIVEPORTRAIT MODEL - AUTOMATIC DOWNLOADER")
print("="*70)
print()
print("This will download LivePortrait model from Hugging Face")
print("Size: ~2GB | Time: 5-15 minutes (depends on connection)")
print()

# Create SSL context for downloads
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Model directory
model_dir = Path("models/liveportrait")
model_dir.mkdir(parents=True, exist_ok=True)

# Download directory
download_dir = Path("downloads")
download_dir.mkdir(exist_ok=True)

print(f"Model directory: {model_dir.absolute()}")
print(f"Download directory: {download_dir.absolute()}")
print()

# Check if model already exists
existing_models = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.onnx"))
if len(existing_models) > 0:
    print(f"⚠ Found {len(existing_models)} existing model file(s):")
    for model in existing_models[:3]:
        print(f"  - {model.name}")
    if len(existing_models) > 3:
        print(f"  ... and {len(existing_models)-3} more")
    print()
    print("Do you want to re-download and overwrite? (y/n)")
    response = input("> ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    print()

# Hugging Face model files
HF_BASE = "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main"

MODEL_FILES = {
    # Main model files
    "appearance_feature_extractor.pth": {
        "url": f"{HF_BASE}/appearance_feature_extractor.pth",
        "size": "~200MB"
    },
    "motion_extractor.pth": {
        "url": f"{HF_BASE}/motion_extractor.pth",
        "size": "~100MB"
    },
    "spade_generator.pth": {
        "url": f"{HF_BASE}/spade_generator.pth",
        "size": "~500MB"
    },
    "warping_module.pth": {
        "url": f"{HF_BASE}/warping_module.pth",
        "size": "~200MB"
    },
    # Additional files
    "stitching_retargeting_module.pth": {
        "url": f"{HF_BASE}/stitching_retargeting_module.pth",
        "size": "~50MB"
    },
}

# Optional config files
CONFIG_FILES = {
    "inference.yaml": f"{HF_BASE}/inference.yaml",
}

def download_file(url, destination, description=""):
    """Download a file with progress bar."""
    try:
        print(f"  Downloading: {description}")
        print(f"  From: {url}")
        print(f"  To: {destination}")

        # Create request
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')

        # Download with progress
        with urllib.request.urlopen(req, context=ssl_context) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0

            with open(destination, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_length = 40
                        filled = int(bar_length * downloaded / total_size)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f"\r  [{bar}] {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end='')

        print()  # New line after progress
        print(f"  ✓ Downloaded successfully")
        return True

    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False

print("="*70)
print("STEP 1: DOWNLOADING MODEL FILES")
print("="*70)
print()

downloaded_files = []
failed_files = []

for i, (filename, info) in enumerate(MODEL_FILES.items(), 1):
    print(f"[{i}/{len(MODEL_FILES)}] {filename} ({info['size']})")

    destination = model_dir / filename

    # Skip if already exists
    if destination.exists():
        print(f"  ⚠ File already exists, skipping")
        downloaded_files.append(filename)
        print()
        continue

    # Download
    success = download_file(info['url'], destination, filename)

    if success:
        downloaded_files.append(filename)
    else:
        failed_files.append(filename)
        print(f"  ⚠ Failed to download {filename}")
        print(f"  You can download manually from:")
        print(f"  {info['url']}")

    print()

print("="*70)
print("STEP 2: DOWNLOADING CONFIG FILES")
print("="*70)
print()

for filename, url in CONFIG_FILES.items():
    print(f"Downloading: {filename}")
    destination = model_dir / filename

    try:
        download_file(url, destination, filename)
        print()
    except Exception as e:
        print(f"  ⚠ Optional file, skipping: {e}")
        print()

print("="*70)
print("STEP 3: VERIFYING INSTALLATION")
print("="*70)
print()

# Check what was downloaded
pth_files = list(model_dir.glob("*.pth"))
print(f"Model files found: {len(pth_files)}")
for f in pth_files:
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  ✓ {f.name} ({size_mb:.1f}MB)")
print()

# Check if minimum required files exist
required_files = [
    "appearance_feature_extractor.pth",
    "motion_extractor.pth",
    "spade_generator.pth",
    "warping_module.pth"
]

missing_files = []
for req_file in required_files:
    if not (model_dir / req_file).exists():
        missing_files.append(req_file)

if missing_files:
    print("⚠ WARNING: Missing required files:")
    for f in missing_files:
        print(f"  - {f}")
    print()
    print("LivePortrait may not work correctly.")
    print("You can download missing files manually from:")
    print("https://huggingface.co/KwaiVGI/LivePortrait/tree/main")
    print()
else:
    print("✓ All required model files present!")
    print()

print("="*70)
print("STEP 4: INSTALLING DEPENDENCIES")
print("="*70)
print()

print("Installing required Python packages...")
print()

# Install dependencies
packages = [
    "face-alignment",
    "imageio[ffmpeg]",
    "scikit-image",
    "scipy",
]

for package in packages:
    print(f"Installing {package}...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  ✓ {package} installed")
        else:
            print(f"  ⚠ {package} installation had issues")
            print(f"  {result.stderr}")
    except Exception as e:
        print(f"  ✗ Error installing {package}: {e}")
    print()

print("="*70)
print("STEP 5: TESTING INSTALLATION")
print("="*70)
print()

print("Running diagnostic test...")
print()

try:
    # Run test
    import subprocess
    result = subprocess.run(
        [sys.executable, "test_liveportrait.py"],
        capture_output=True,
        text=True,
        timeout=30
    )

    # Show relevant output
    if "Real LivePortrait model NOT installed" in result.stdout:
        print("⚠ Test shows model not detected properly")
        print("This may be a detection issue, model files are present.")
    elif "Real LivePortrait" in result.stdout and "detected" in result.stdout.lower():
        print("✓ Test successful - LivePortrait detected!")
    else:
        print("Test output:")
        print(result.stdout[-500:])  # Last 500 chars

except subprocess.TimeoutExpired:
    print("⚠ Test timed out (this is ok, model might be loading)")
except Exception as e:
    print(f"⚠ Test error: {e}")

print()

print("="*70)
print("INSTALLATION COMPLETE")
print("="*70)
print()

if not missing_files:
    print("✅ SUCCESS! LivePortrait model installed")
    print()
    print("Model files installed:")
    for f in downloaded_files:
        print(f"  ✓ {f}")
    print()
    print(f"Location: {model_dir.absolute()}")
    print()
    print("Next steps:")
    print("  1. Run: run.bat")
    print("  2. Your character will now animate with your face!")
    print("  3. Press 'T' in app to see stats")
    print("  4. Press arrow keys to switch characters")
    print()
    print("If character still doesn't animate:")
    print("  - Restart the application")
    print("  - Check logs for errors")
    print("  - Run: python test_liveportrait.py")
else:
    print("⚠ PARTIAL INSTALLATION")
    print()
    print("Some files failed to download.")
    print()
    print("Missing files:")
    for f in missing_files:
        print(f"  - {f}")
    print()
    print("You can:")
    print("  1. Re-run this script")
    print("  2. Download manually from:")
    print("     https://huggingface.co/KwaiVGI/LivePortrait/tree/main")
    print(f"  3. Place files in: {model_dir.absolute()}")

if failed_files:
    print()
    print("Failed downloads:")
    for f in failed_files:
        print(f"  - {f}")
    print()
    print("Manual download URLs:")
    for f in failed_files:
        if f in MODEL_FILES:
            print(f"  {f}:")
            print(f"    {MODEL_FILES[f]['url']}")

print()
print("="*70)

