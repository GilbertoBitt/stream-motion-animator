# 📥 COMPLETE GUIDE - Download LivePortrait from Hugging Face

## 🎯 Quick Summary

**LivePortrait model files need to be downloaded manually from Hugging Face.**

The automatic downloader found that the files require authentication or are organized differently. Here's how to get them:

---

## ✅ METHOD 1: Direct Browser Download (Easiest)

### Step 1: Visit Hugging Face

Open this URL in your browser:
```
https://huggingface.co/KwaiVGI/LivePortrait/tree/main/pretrained_weights/liveportrait
```

Or run:
```bash
python download_liveportrait_manual.py
# Choose option 1 to open browser
```

### Step 2: Download These Files

Click on each file and download:

1. **appearance_feature_extractor.pth** (~200MB)
2. **motion_extractor.pth** (~100MB)  
3. **spade_generator.pth** (~500MB)
4. **warping_module.pth** (~200MB)
5. **stitching_retargeting_module.pth** (~50MB)

**Total Size:** ~1GB

### Step 3: Place Files

Move/copy downloaded files to:
```
G:\stream-motion-animator\models\liveportrait\
```

### Step 4: Verify

Check the directory contains:
```
models/liveportrait/
├── appearance_feature_extractor.pth
├── motion_extractor.pth
├── spade_generator.pth
├── warping_module.pth
├── stitching_retargeting_module.pth
└── README.txt
```

### Step 5: Test

```bash
python test_liveportrait.py
```

Should show:
```
✓ Real LivePortrait model detected!
```

### Step 6: Run

```bash
run.bat
```

**Done!** Character will now animate! 🎉

---

## ✅ METHOD 2: Using Git LFS (For Advanced Users)

If you have git and git-lfs installed:

### Step 1: Install Git LFS

```bash
git lfs install
```

### Step 2: Clone Repository

```bash
cd G:\
git clone https://huggingface.co/KwaiVGI/LivePortrait
```

**Note:** This downloads ~2GB, may take 10-30 minutes

### Step 3: Copy Model Files

```bash
cd LivePortrait\pretrained_weights\liveportrait
copy *.pth G:\stream-motion-animator\models\liveportrait\
```

### Step 4: Test

```bash
cd G:\stream-motion-animator
python test_liveportrait.py
```

### Step 5: Run

```bash
run.bat
```

---

## ✅ METHOD 3: Using Hugging Face CLI

If you have Hugging Face CLI installed:

### Step 1: Install CLI

```bash
pip install huggingface_hub
```

### Step 2: Download Files

```bash
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='KwaiVGI/LivePortrait', filename='pretrained_weights/liveportrait/appearance_feature_extractor.pth', local_dir='.')"
```

Repeat for each file, or use this script:

```python
from huggingface_hub import hf_hub_download

files = [
    'pretrained_weights/liveportrait/appearance_feature_extractor.pth',
    'pretrained_weights/liveportrait/motion_extractor.pth',
    'pretrained_weights/liveportrait/spade_generator.pth',
    'pretrained_weights/liveportrait/warping_module.pth',
    'pretrained_weights/liveportrait/stitching_retargeting_module.pth',
]

for file in files:
    print(f"Downloading {file}...")
    hf_hub_download(
        repo_id='KwaiVGI/LivePortrait',
        filename=file,
        local_dir='models',
        local_dir_use_symlinks=False
    )
```

---

## 📊 File Details

| File | Size | Purpose |
|------|------|---------|
| appearance_feature_extractor.pth | ~200MB | Extracts visual features from character |
| motion_extractor.pth | ~100MB | Extracts motion from webcam |
| spade_generator.pth | ~500MB | Generates animated frames |
| warping_module.pth | ~200MB | Warps character to match motion |
| stitching_retargeting_module.pth | ~50MB | Stitches and retargets animation |

**Total:** ~1GB (1050MB)

---

## 🔍 Verification Checklist

After downloading, verify your installation:

### Run Check Script

```bash
python download_liveportrait_manual.py
# Choose option 3 to check status
```

### Expected Output

```
✓ Found 5 model file(s):
  ✓ appearance_feature_extractor.pth (200.0MB)
  ✓ motion_extractor.pth (100.0MB)
  ✓ spade_generator.pth (500.0MB)
  ✓ warping_module.pth (200.0MB)
  ✓ stitching_retargeting_module.pth (50.0MB)

✅ All required files present!
```

### Run Diagnostic

```bash
python test_liveportrait.py
```

Should show:
```
[6/6] Checking LivePortrait model status...
  Model path: models\liveportrait
  Exists: True
  Files in model directory: 5
    - appearance_feature_extractor.pth
    - motion_extractor.pth
    - spade_generator.pth
    - warping_module.pth
    - stitching_retargeting_module.pth
  
  ✓ Real LivePortrait model detected!
```

---

## 🐛 Troubleshooting

### Issue: "404 Not Found" when downloading

**Cause:** Files require direct browser download or authentication

**Solution:** Use METHOD 1 (browser download) - it's the most reliable

### Issue: Downloads are slow

**Solutions:**
1. Use browser download (can pause/resume)
2. Use download manager (IDM, FDM)
3. Try different time of day
4. Use METHOD 2 (git lfs) if you have it

### Issue: "Not enough disk space"

**Check:**
```bash
# Need ~5GB free space total
# - 1GB for downloads
# - 1GB for model files
# - 3GB for cache and temporary files
```

**Solution:** Free up space or use external drive

### Issue: Files won't copy to models/liveportrait

**Check permissions:**
```bash
# Make sure directory exists
mkdir models\liveportrait

# Try moving files instead of copying
move *.pth models\liveportrait\
```

### Issue: Model still not detected after download

**Steps:**
1. Verify files are in correct location:
   ```bash
   dir models\liveportrait\*.pth
   ```
   
2. Check file sizes (should be 50-500MB each)

3. Re-run diagnostic:
   ```bash
   python test_liveportrait.py
   ```

4. Restart application:
   ```bash
   run.bat
   ```

---

## 📝 Quick Commands

```bash
# Check what you need to download
python download_liveportrait_manual.py

# Check current installation
python download_liveportrait_manual.py
# (choose option 3)

# Test after download
python test_liveportrait.py

# Run application
run.bat
```

---

## 🎯 Expected Workflow

### Complete Installation Process

```bash
# 1. Download files
# Visit: https://huggingface.co/KwaiVGI/LivePortrait/tree/main/pretrained_weights/liveportrait
# Download all 5 .pth files

# 2. Move to correct location
# Copy files to: G:\stream-motion-animator\models\liveportrait\

# 3. Verify installation
python test_liveportrait.py

# Expected output:
# ✓ Real LivePortrait model detected!

# 4. Run application
run.bat

# 5. Test animation
# - Face appears in camera
# - Character animates with your face
# - Mouth syncs
# - Head movements track
# - Expressions transfer
```

### Total Time

- Download: 5-15 minutes (depends on internet)
- Setup: 1 minute (copy files)
- Verification: 30 seconds
- **Total: 10-20 minutes**

---

## ✨ After Installation

Once LivePortrait is installed:

### Character should:
- ✅ Follow your head movements
- ✅ Sync mouth with yours
- ✅ Blink when you blink
- ✅ Show your expressions
- ✅ Animate in real-time at 30-60 FPS

### Controls:
- **Arrow Keys**: Switch characters
- **T**: Toggle stats
- **R**: Reload characters
- **Q**: Quit

### Performance:
- First frame: 100-200ms (feature extraction)
- Subsequent frames: 15-30ms (cached)
- Expected FPS: 30-60 FPS

---

## 📚 Additional Resources

- **Official Repo:** https://github.com/KwaiVGI/LivePortrait
- **Hugging Face:** https://huggingface.co/KwaiVGI/LivePortrait
- **Paper:** https://arxiv.org/abs/2407.03168
- **Demo:** https://huggingface.co/spaces/KwaiVGI/LivePortrait

---

## 🆘 Need Help?

**Run diagnostic:**
```bash
python test_liveportrait.py
```

**Check installation:**
```bash
python download_liveportrait_manual.py
```

**Check files manually:**
```bash
dir models\liveportrait\*.pth
```

**Expected output:**
```
appearance_feature_extractor.pth
motion_extractor.pth
spade_generator.pth
warping_module.pth
stitching_retargeting_module.pth
```

---

**Status:** 📥 **Manual download required**  
**Location:** https://huggingface.co/KwaiVGI/LivePortrait  
**Size:** ~1GB  
**Time:** 10-20 minutes  
**Difficulty:** Easy (just download and copy files)

🎭 **Once downloaded, full facial animation will work!** ✨

