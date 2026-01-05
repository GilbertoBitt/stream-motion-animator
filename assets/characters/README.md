# Character Image Guidelines

This directory contains character images used for animation. Place your character images here.

## Recommended Specifications

### Image Format
- **Supported formats**: PNG, JPG, JPEG
- **Transparency**: PNG with alpha channel recommended for best results
- **Color mode**: RGB or RGBA

### Resolution
- **Minimum**: 512x512 pixels
- **Recommended**: 1024x1024 pixels or higher
- **Maximum**: 2048x2048 pixels (larger images will be resized)

### Face Requirements
- **Face visibility**: Full frontal face clearly visible
- **Expression**: Neutral expression recommended
- **Lighting**: Even lighting, avoid harsh shadows
- **Framing**: Face should occupy 60-80% of the image
- **Eyes**: Both eyes clearly visible and open
- **Occlusions**: Avoid sunglasses, masks, or objects covering the face

## File Naming
- Use descriptive names: `character1.png`, `anime_girl.png`, `avatar_male.png`
- Avoid special characters and spaces
- Files are sorted alphabetically in the application

## Example Structure
```
assets/characters/
├── character1.png
├── character2.png
├── anime_girl.png
├── realistic_avatar.jpg
└── README.md (this file)
```

## Tips for Best Results

### For Photo Portraits
- Use high-quality photos with good resolution
- Neutral background works best
- Direct eye contact with camera
- Good lighting from the front

### For Anime/Drawn Characters
- High-resolution artwork
- Clean lines and clear facial features
- Full face visible (not side profile)
- Consistent art style across all images

### Preparing Images
1. Crop image to focus on face
2. Resize to at least 1024x1024
3. Ensure good lighting and clarity
4. Save as PNG for transparency support

## Testing Your Images
Run the test tool to verify your images are compatible:
```bash
python tools/test_character.py assets/characters/your_image.png
```

## Batch Conversion
If you need to convert multiple images:
```bash
# Resize all images to 1024x1024
python tools/batch_convert.py assets/characters/ --size 1024
```

## Troubleshooting

**Q: Face not detected**
- Ensure face is clearly visible and frontal
- Check image quality and resolution
- Verify proper lighting

**Q: Animation looks distorted**
- Use higher resolution source image
- Ensure face is properly framed
- Check for image artifacts

**Q: Character switching is slow**
- Use smaller file sizes (under 2MB per image)
- Use PNG instead of high-quality JPG
- Reduce number of characters if loading too many

For more help, see docs/TROUBLESHOOTING.md
