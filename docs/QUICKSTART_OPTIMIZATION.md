# Quick Start: Using the Inference Optimizer

This guide will help you get started with the new inference optimization features in just a few minutes.

## Step 1: Verify Configuration

Open `assets/config.yaml` and ensure these settings are enabled:

```yaml
performance:
  enable_optimizer: true       # ✅ Enable optimizer
  cache_dir: "cache/features"  # Cache location

ai_model:
  fp16: true                   # ✅ Use FP16 for speed
  device: "cuda"               # ✅ Use GPU
```

## Step 2: Run the Application

```bash
python src/main.py
```

### What to Expect

**On First Run:**
```
2026-01-05 10:30:15 - INFO - Initializing AI animator...
2026-01-05 10:30:16 - INFO - Model loaded successfully
2026-01-05 10:30:16 - INFO - Initializing inference optimizer...
2026-01-05 10:30:16 - INFO - Inference optimizer initialized
2026-01-05 10:30:17 - INFO - Pre-processing characters for optimized inference...
2026-01-05 10:30:17 - INFO - Pre-processing character: character1.png
2026-01-05 10:30:17 - INFO - Character character1.png pre-processed and cached
2026-01-05 10:30:18 - INFO - Pre-processing character: character2.png
...
2026-01-05 10:30:20 - INFO - Character pre-processing complete
```

**On Subsequent Runs:**
```
2026-01-05 10:35:10 - INFO - Loaded cached features for character1.png
2026-01-05 10:35:10 - INFO - Loaded cached features for character2.png
...
(Much faster startup!)
```

## Step 3: Monitor Performance

During runtime, press `T` to toggle performance stats:

```
============================================================
FPS: Capture=60.0 | Tracking=60.0 | Inference=58.5 | Output=60.0 | Total=58.2
Inference Time: 7.8ms
Character: character1.png (1/10)
Optimizer: Cache Hit Rate=92.30% | Cached=10 chars | Motion Cache=67
============================================================
```

### Understanding the Metrics

- **Inference Time**: Should be ~5-10ms (was ~15-30ms without optimizer)
- **Cache Hit Rate**: % of frames using cached motion (higher is better)
- **Cached chars**: Number of pre-processed characters in memory
- **Motion Cache**: Number of cached motion vectors

## Step 4: Verify Performance Improvement

### Before Optimizer (disable in config)
```yaml
performance:
  enable_optimizer: false
```

Run and note the inference time:
```
Inference Time: 18.5ms  # Typical without optimizer
```

### After Optimizer (enable in config)
```yaml
performance:
  enable_optimizer: true
```

Run and compare:
```
Inference Time: 7.2ms  # ~60% faster!
```

## Common Scenarios

### Scenario 1: Low Cache Hit Rate

**Problem:**
```
Optimizer: Cache Hit Rate=35.20% | ...
```

**Solution:** Your movements are very dynamic. This is normal for rapid motion.

**To Improve:**
1. Make smoother, slower head movements
2. The cache will still help with micro-movements
3. Consider increasing cache size in code

### Scenario 2: High Memory Usage

**Problem:** GPU memory warning

**Solution:**
1. Clear old cache: `rm -rf cache/features/*`
2. Reduce number of pre-loaded characters
3. Set `character.preload_all: false` in config

### Scenario 3: Slow Startup

**Problem:** First run takes a while to pre-process

**Solution:**
1. This is normal - features are cached for next time
2. Reduce number of characters
3. Pre-process in background (future feature)
4. Cache persists across runs, so subsequent starts are fast

## Advanced Usage

### Clear Cache Programmatically

Add to your code:
```python
from src.ai_animator import AIAnimator

animator = AIAnimator(enable_optimizer=True)
animator.initialize()

# Clear all cache
animator.optimizer.clear_cache()

# Clear disk cache only
animator.optimizer.clear_cache(disk_only=True)
```

### Disable for Specific Characters

```python
# Don't use optimizer for this frame
animated = animator.animate_frame(
    character_image=char_img,
    webcam_frame=webcam,
    landmarks=landmarks,
    character_id=None  # None = skip optimizer
)
```

### Get Detailed Statistics

```python
stats = animator.optimizer.get_stats()
print(f"Cached characters: {stats['cached_characters']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Motion cache size: {stats['motion_cache_size']}")
```

## Benchmarking

Run the included benchmark tool:

```bash
# With optimizer
python tools/benchmark.py --optimizer=true

# Without optimizer
python tools/benchmark.py --optimizer=false
```

Expected results on RTX 3060:
```
=== Benchmark Results ===
Optimizer: ENABLED
Average Inference Time: 7.2ms
Average FPS: 58.5
GPU Usage: 42%
Cache Hit Rate: 87%
```

## Troubleshooting

### Issue: "Optimizer not enabled" warning

**Solution:** Check config.yaml:
```yaml
performance:
  enable_optimizer: true  # Must be true
```

### Issue: Cache directory not found

**Solution:** Create manually or let app create it:
```bash
mkdir -p cache/features
```

### Issue: Characters not animating

**Solution:** 
1. Check if model is loaded: Look for "Model loaded successfully"
2. Verify character images exist in `assets/characters/`
3. Check logs for errors

### Issue: Lower performance than expected

**Solution:**
1. Enable FP16: `ai_model.fp16: true`
2. Use GPU: `ai_model.device: "cuda"`
3. Close other GPU applications
4. Update GPU drivers

## Next Steps

- Read [OPTIMIZATION.md](OPTIMIZATION.md) for detailed documentation
- See [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) for implementation details
- Check [PERFORMANCE.md](PERFORMANCE.md) for tuning tips

## Key Takeaways

✅ **Enable optimizer in config** for automatic optimization
✅ **First run pre-processes** characters (one-time cost)
✅ **Subsequent runs load from cache** (instant)
✅ **Monitor cache hit rate** for performance insights
✅ **Clear cache when characters change** (manual step)

Enjoy your 60+ FPS optimized streaming experience! 🚀

