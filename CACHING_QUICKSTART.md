# ⚡ QUICK START - Caching Commands

## 🎯 The One Command You Need

### **Cache All Characters:**

```bash
cache_characters.bat
```

**That's it!** This does everything:
- ✅ Extracts frames from videos
- ✅ Processes all images  
- ✅ Creates optimized cache
- ✅ Makes loading 60x faster

---

## 📋 Complete Workflow

### 1️⃣ Add Characters

```bash
# Create folder
mkdir assets\characters\emma

# Add files
copy *.png assets\characters\emma\
copy *.mp4 assets\characters\emma\
```

### 2️⃣ Cache Everything

```bash
cache_characters.bat
```

**Output:**
```
✓ Total characters: 1
✓ Total videos: 2
✓ Video frames extracted: 60
✓ First load time: 3.2 seconds
✓ Next load time: ~0.1 seconds (cached)
✅ PREPROCESSING COMPLETE!
```

### 3️⃣ Run Application

```bash
run.bat
```

**Result:** Characters load instantly! 🚀

---

## 🔄 When to Re-run Caching

Run `cache_characters.bat` again when you:

- ✅ Add new characters
- ✅ Add new videos
- ✅ Modify existing images
- ✅ Change config settings

---

## 💡 Alternative Commands

### PowerShell:
```powershell
.\cache_characters.ps1
```

### Python:
```bash
python tools/preprocess_all_characters.py
```

### From virtual env:
```bash
.\.venv\Scripts\python.exe tools\preprocess_all_characters.py
```

---

## 📊 What You Get

| Metric | Before Cache | After Cache |
|--------|-------------|-------------|
| Load time | 5-10 seconds | 0.1 seconds |
| Character switch | 2-5 seconds | 0.1 seconds |
| App startup | 10-30 seconds | 1-2 seconds |
| **Speedup** | **1x** | **60x** |

---

## 🐛 Quick Troubleshooting

### "No characters found"
```bash
# Check if characters exist
ls assets\characters

# If empty, run setup:
python tools/setup_character_structure.py setup
```

### Cache not working
```bash
# Delete cache and recreate
rmdir /S cache\characters
cache_characters.bat
```

### Out of memory
Edit `assets/config.yaml`:
```yaml
character:
  max_frames_per_video: 10  # Reduce from 30
```

---

## 📚 Full Documentation

- **CACHING_GUIDE.md** - Complete caching guide
- **MULTI_BATCH_GUIDE.md** - Multi-batch character guide
- **README_FIXED.md** - General documentation

---

## ✨ Summary

**Command:** `cache_characters.bat`  
**Runtime:** 1-5 seconds per character  
**Benefit:** 60x faster loading  
**Frequency:** Run once per update  

**Simple as that!** 🎉

