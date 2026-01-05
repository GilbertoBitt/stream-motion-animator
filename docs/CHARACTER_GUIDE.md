# Character Image Guide

Complete guide for preparing and optimizing character images for animation.

## Table of Contents

- [Image Requirements](#image-requirements)
- [Recommended Specifications](#recommended-specifications)
- [Preparing Images](#preparing-images)
- [Face Guidelines](#face-guidelines)
- [Transparency](#transparency)
- [Batch Conversion](#batch-conversion)
- [Testing Images](#testing-images)
- [Common Issues](#common-issues)
- [Tips by Character Type](#tips-by-character-type)

## Image Requirements

### Supported Formats

✅ **Supported**:
- PNG (recommended for transparency)
- JPEG/JPG
- BMP

❌ **Not Supported**:
- GIF
- WebP
- TIFF
- SVG

### Minimum Requirements

| Property | Minimum | Recommended | Maximum |
|----------|---------|-------------|---------|
| **Resolution** | 256×256 | 1024×1024 | 2048×2048 |
| **File Size** | - | <2MB | 10MB |
| **Bit Depth** | 24-bit RGB | 32-bit RGBA | - |
| **Aspect Ratio** | 1:1 | 1:1 | 1:1 |

### Color Modes

- **RGB**: Opaque backgrounds
- **RGBA**: Transparent backgrounds (recommended)
- **Grayscale**: Not recommended (will be converted to RGB)

## Recommended Specifications

### Optimal Settings

For best results:

```
Format:          PNG
Resolution:      1024×1024 pixels
Color Mode:      RGBA (with transparency)
Bit Depth:       32-bit
Compression:     PNG default
Face Coverage:   60-80% of image
Expression:      Neutral
Lighting:        Even, frontal
Background:      Transparent or solid color
```

### Resolution Guidelines

**256×256**: ⚠️ Minimum quality, noticeable artifacts
**512×512**: ✅ Good quality, balanced performance
**1024×1024**: ✅✅ Recommended, high quality
**2048×2048**: 🔥 Maximum quality, slower loading

Choose based on your performance needs:
- **RTX 3080**: Use 1024×1024
- **RTX 3070**: Use 512×512 or 1024×1024
- **RTX 3060**: Use 512×512

## Preparing Images

### Step 1: Source Image

**Good Sources**:
- Professional portraits
- High-quality character artwork
- Anime/manga illustrations
- AI-generated characters (Stable Diffusion, Midjourney)

**Avoid**:
- Low-resolution images
- Side profiles (must be frontal)
- Multiple faces in frame
- Heavily occluded faces (hands covering, masks)

### Step 2: Crop and Frame

1. **Center the face**:
   - Face should be in the center
   - Eyes roughly at vertical center
   - Equal space on left and right

2. **Face coverage**:
   - Aim for 60-80% of image
   - Include full head (top of head to chin)
   - Include neck/shoulders for context

3. **Use auto-crop** (if enabled):
   ```yaml
   character:
     auto_crop: true
   ```
   The system will detect and crop faces automatically.

### Step 3: Resize

Use image editing software:

**Photoshop**:
1. Image → Image Size
2. Set to 1024×1024
3. Resample: Bicubic Sharper

**GIMP** (Free):
1. Image → Scale Image
2. Width: 1024, Height: 1024
3. Interpolation: Cubic

**Python Script**:
```python
from PIL import Image

img = Image.open('character.png')
img = img.resize((1024, 1024), Image.LANCZOS)
img.save('character_resized.png')
```

### Step 4: Remove Background (Optional)

For transparent backgrounds:

**remove.bg** (Online):
1. Upload image to [remove.bg](https://www.remove.bg)
2. Download result
3. Save as PNG

**Photoshop**:
1. Select → Subject
2. Select → Inverse
3. Delete
4. Save as PNG

**GIMP**:
1. Filters → Edge Detect → Edge
2. Select by Color
3. Delete background
4. Export as PNG

## Face Guidelines

### Frontal Face

✅ **Good**:
- Face directly facing camera
- Both eyes visible
- Symmetrical
- Neutral or slight smile

❌ **Bad**:
- Side profile (>30° rotation)
- Looking away
- Upward/downward angle
- Eyes closed

### Expression

**Recommended**: Neutral or slight smile
- Easier to animate
- Natural resting position
- Works for all emotions

**Avoid**:
- Extreme expressions (laughing, crying)
- Mouth wide open
- Squinting
- Tongue out

### Facial Features

**Eyes**:
- Both eyes clearly visible
- Looking straight ahead
- Not behind glasses (or remove reflections)

**Mouth**:
- Closed or slightly open
- Teeth not prominent
- Natural position

**Hair**:
- Any hairstyle works
- Avoid covering eyes
- Transparent PNG if hair overlaps face

### Lighting

**Optimal**:
- Even lighting from front
- Soft shadows
- No harsh highlights
- Consistent color temperature

**Avoid**:
- Side lighting (harsh shadows)
- Backlit (silhouette)
- Colored lights (unnatural skin tones)
- Flash reflections

## Transparency

### When to Use

✅ **Use transparency** for:
- Green screen / chroma key in OBS
- Overlaying on game footage
- Layered compositions
- Professional look

❌ **Opaque background** for:
- Simpler setup
- Smaller file sizes
- Testing/debugging

### Creating Transparent PNGs

1. **Remove background** (see Step 4 above)

2. **Check alpha channel**:
   ```python
   from PIL import Image
   img = Image.open('character.png')
   print(f"Mode: {img.mode}")  # Should be 'RGBA'
   ```

3. **Fix partial transparency**:
   - Use "Select → Color Range" to remove green screen
   - Feather edges for smooth transition
   - Clean up artifacts

### Transparent Background Tips

- Use solid color behind character for clean edges
- Avoid semi-transparent areas (can cause artifacts)
- Test in OBS before finalizing

## Batch Conversion

### Using Python Script

Create `batch_resize.py`:
```python
from PIL import Image
from pathlib import Path

input_dir = Path('input_characters')
output_dir = Path('assets/characters')
output_dir.mkdir(exist_ok=True)

target_size = (1024, 1024)

for img_path in input_dir.glob('*.png'):
    img = Image.open(img_path)
    
    # Convert to RGBA
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Resize
    img = img.resize(target_size, Image.LANCZOS)
    
    # Save
    output_path = output_dir / img_path.name
    img.save(output_path, 'PNG', optimize=True)
    print(f"Processed: {img_path.name}")
```

Run:
```bash
python batch_resize.py
```

### Using ImageMagick

Batch convert all images:
```bash
magick mogrify -resize 1024x1024 -format png -path assets/characters/ input_characters/*.jpg
```

## Testing Images

### Use Test Tool

```bash
python tools/test_character.py assets/characters/your_image.png
```

Output includes:
- Image format and size
- Face detection results
- Coverage percentage
- Recommendations

### Validation Checklist

Before using in production:

- [ ] Resolution at least 512×512
- [ ] Face detected successfully
- [ ] Face coverage 40-80%
- [ ] Both eyes visible
- [ ] Neutral expression
- [ ] Frontal view (not profile)
- [ ] Good lighting
- [ ] Proper format (PNG/JPG)
- [ ] File size under 5MB

## Common Issues

### "No face detected"

**Causes**:
- Face too small in image
- Side profile
- Low contrast
- Face partially occluded

**Solutions**:
1. Crop closer to face
2. Increase image brightness
3. Use frontal-facing image
4. Remove obstructions (hands, hair)

### "Animation looks distorted"

**Causes**:
- Low resolution source
- Extreme expression
- Non-neutral pose

**Solutions**:
1. Use higher resolution (1024×1024)
2. Use neutral expression
3. Ensure face is centered and frontal

### "Slow character switching"

**Causes**:
- Large file sizes
- Too many characters
- Not preloaded

**Solutions**:
1. Compress images (PNG optimization)
2. Limit to 5-10 characters
3. Enable preloading:
   ```yaml
   character:
     preload_all: true
   ```

### "Edges look jagged"

**Causes**:
- Poor background removal
- Low resolution
- No anti-aliasing

**Solutions**:
1. Use professional background removal tool
2. Feather selection edges
3. Higher resolution source image

## Tips by Character Type

### Realistic Photos

**Best Practices**:
- Use professional headshots
- Even studio lighting
- Neutral background
- High resolution (1080p+)

**Processing**:
1. Remove background
2. Adjust exposure if needed
3. Resize to 1024×1024
4. Sharpen slightly

### Anime/Manga

**Best Practices**:
- Clean lineart
- Consistent coloring
- Clear facial features
- Simple background

**Processing**:
1. Ensure clean lines (no artifacts)
2. Remove background (usually solid color)
3. Keep sharp edges
4. Don't over-compress

### AI-Generated Characters

**Best Practices**:
- Use high-resolution generation (1024+)
- Specify "frontal view, neutral expression"
- Clean up artifacts in editing
- Multiple generations, pick best

**Prompts**:
- ✅ "portrait, facing camera, neutral expression, centered"
- ❌ "side profile, looking away, dramatic lighting"

### VRChat/3D Models

**Best Practices**:
- Use in-game portrait mode
- Disable UI overlays
- Good lighting in scene
- High graphics settings

**Processing**:
1. Capture screenshot at high resolution
2. Crop to character
3. Remove or blur background
4. Adjust exposure if needed

### Pixel Art

**Not Recommended**: Low resolution, will be heavily upscaled

If using:
1. Use nearest-neighbor upscaling
2. Export at 1024×1024
3. Expect blocky results

## Example Workflow

### Professional Character Preparation

1. **Source**: Professional portrait or high-quality artwork

2. **Edit**:
   ```
   - Crop to 1:1 aspect ratio
   - Center face
   - Remove background (transparent PNG)
   - Adjust brightness/contrast
   ```

3. **Export**:
   ```
   Format: PNG
   Size: 1024×1024
   Mode: RGBA
   Quality: Maximum
   ```

4. **Test**:
   ```bash
   python tools/test_character.py assets/characters/character1.png
   ```

5. **Verify**: Load in application and test animation

6. **Iterate**: Adjust based on results

---

**Need Help?** Join our [Discord community](https://discord.gg/your-invite) or check [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
