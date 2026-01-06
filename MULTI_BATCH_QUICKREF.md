# 🎬 Multi-Batch Quick Reference Card

## 📁 Folder Structure

```
assets/characters/
├── character1/
│   ├── image1.png
│   ├── image2.jpg
│   └── video.mp4
└── character2/
    └── main.png
```

## ⚙️ Configuration

```yaml
character:
  enable_multi_batch: true
  enable_video_processing: true
  max_frames_per_video: 30
  video_sample_rate: 10
  use_reference_batch: true
```

## 🚀 Quick Commands

```bash
# Check structure
python tools/setup_character_structure.py check

# Migrate from flat
python tools/setup_character_structure.py migrate

# Test system
python test_multibatch.py

# Run application
run.bat
# or
python src/main.py --camera 1
```

## 📊 Supported Formats

**Images:** PNG, JPG, JPEG, BMP, WebP, TIFF  
**Videos:** MP4, AVI, MOV, MKV, WMV, FLV, WebM

## 💡 Best Practices

✅ **DO:**
- Use 2-5 images per character
- Include 1-2 videos with expressions
- Keep videos under 30 seconds
- Use clear, well-lit references
- Enable caching

❌ **DON'T:**
- Use blurry/low-quality images
- Add hundreds of references (20-100 is optimal)
- Forget to enable `enable_multi_batch`
- Disable caching

## 📈 Performance

| Setup | References | Load Time | Quality |
|-------|-----------|-----------|---------|
| Single | 1 | 100ms | Good |
| Multi | 5-10 | 100ms* | Great |
| Full | 60+ | 100ms* | Excellent |

*After first load (cached)

## 🎯 Typical Workflow

1. **Create folder:** `mkdir assets/characters/emma`
2. **Add files:** Copy PNG/JPG + MP4 files
3. **Run app:** `run.bat`
4. **Done!** System handles everything

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| No characters | Run setup tool |
| Videos not working | Check `enable_video_processing: true` |
| Slow loading | Check cache is enabled |
| High memory | Reduce `max_frames_per_video` |

## 📚 Documentation

- **MULTI_BATCH_GUIDE.md** - Complete guide
- **MULTI_BATCH_SUMMARY.md** - Technical details
- **README_FIXED.md** - General documentation

## ✨ Key Features

✅ Multiple images per character  
✅ Automatic video frame extraction  
✅ Smart caching system  
✅ Backward compatible  
✅ Easy migration tool  
✅ Better animation quality  

---

**Quick Start:** `python tools/setup_character_structure.py` then `run.bat`

