#!/usr/bin/env python3
"""
Character image testing tool.

Validates character images for compatibility.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from character_manager import CharacterManager
from PIL import Image
import cv2
import numpy as np


def test_character_image(image_path: str, verbose: bool = False) -> bool:
    """
    Test a character image for compatibility.
    
    Args:
        image_path: Path to character image
        verbose: Print detailed information
        
    Returns:
        True if valid
    """
    path = Path(image_path)
    
    if not path.exists():
        print(f"❌ File not found: {image_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Testing: {path.name}")
    print(f"{'='*60}")
    
    # Create temporary character manager for validation
    char_manager = CharacterManager(
        characters_path=path.parent,
        auto_crop=True,
        preload_all=False
    )
    
    # Validate image
    is_valid, message = char_manager.validate_image(path)
    
    if not is_valid:
        print(f"❌ INVALID: {message}")
        return False
    
    # Load image for detailed info
    try:
        image = Image.open(path)
        
        print(f"✅ VALID")
        print(f"\nImage Information:")
        print(f"  Format:       {image.format}")
        print(f"  Mode:         {image.mode}")
        print(f"  Size:         {image.size[0]}x{image.size[1]}")
        print(f"  File Size:    {path.stat().st_size / 1024:.1f} KB")
        
        # Check for transparency
        has_alpha = image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info)
        print(f"  Transparency: {'Yes' if has_alpha else 'No'}")
        
        # Face detection
        image_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        print(f"\nFace Detection:")
        print(f"  Faces Found:  {len(faces)}")
        
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                print(f"  Face {i+1}:      {w}x{h} at ({x}, {y})")
                
                # Calculate face coverage
                face_area = w * h
                image_area = image.size[0] * image.size[1]
                coverage = (face_area / image_area) * 100
                print(f"              Coverage: {coverage:.1f}%")
                
                if coverage < 30:
                    print(f"              ⚠️  Face is small (recommended: 40-80%)")
                elif coverage > 90:
                    print(f"              ⚠️  Face too large (recommended: 40-80%)")
                else:
                    print(f"              ✅ Good size")
        else:
            print(f"  ⚠️  No face detected - image may not work well")
        
        # Recommendations
        print(f"\nRecommendations:")
        
        if image.size[0] < 512 or image.size[1] < 512:
            print(f"  ⚠️  Low resolution - recommend at least 1024x1024")
        elif image.size[0] > 2048 or image.size[1] > 2048:
            print(f"  ℹ️  High resolution - will be resized (may slow loading)")
        else:
            print(f"  ✅ Good resolution")
        
        if not has_alpha:
            print(f"  ℹ️  No transparency - opaque background will be used")
        else:
            print(f"  ✅ Has transparency")
        
        if len(faces) == 1:
            print(f"  ✅ Single face detected")
        elif len(faces) > 1:
            print(f"  ⚠️  Multiple faces detected - only largest will be used")
        
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return False


def test_directory(directory: str) -> None:
    """Test all images in a directory."""
    path = Path(directory)
    
    if not path.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    # Find all image files
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = sorted([
        f for f in path.iterdir()
        if f.suffix.lower() in valid_extensions and f.is_file()
    ])
    
    if not image_files:
        print(f"❌ No image files found in {directory}")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    valid_count = 0
    invalid_count = 0
    
    for image_file in image_files:
        if test_character_image(str(image_file)):
            valid_count += 1
        else:
            invalid_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total:   {len(image_files)}")
    print(f"Valid:   {valid_count} ✅")
    print(f"Invalid: {invalid_count} ❌")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test character images for compatibility"
    )
    parser.add_argument(
        'path',
        help='Path to image file or directory'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    path = Path(args.path)
    
    if path.is_file():
        success = test_character_image(args.path, args.verbose)
        sys.exit(0 if success else 1)
    elif path.is_dir():
        test_directory(args.path)
    else:
        print(f"❌ Path not found: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
