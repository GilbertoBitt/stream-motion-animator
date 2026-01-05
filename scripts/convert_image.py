#!/usr/bin/env python3
"""
Image conversion utility for preparing source images
"""
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_image, save_image, setup_logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Convert and prepare images for AI animation'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input image path'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output image path (default: input_converted.jpg)'
    )
    
    parser.add_argument(
        '--size', '-s',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        help='Resize to specific dimensions'
    )
    
    parser.add_argument(
        '--square',
        action='store_true',
        help='Crop to square aspect ratio'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG quality (1-100, default: 95)'
    )
    
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize brightness and contrast'
    )
    
    return parser.parse_args()


def crop_to_square(image: np.ndarray) -> np.ndarray:
    """Crop image to square, centered"""
    h, w = image.shape[:2]
    
    if h == w:
        return image
    
    if h > w:
        # Tall image, crop height
        diff = h - w
        top = diff // 2
        return image[top:top+w, :]
    else:
        # Wide image, crop width
        diff = w - h
        left = diff // 2
        return image[:, left:left+h]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize brightness and contrast"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge([l, a, b])
    
    # Convert back to BGR
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return normalized


def main():
    """Main entry point"""
    args = parse_arguments()
    
    logger = setup_logger('convert_image', level='INFO')
    
    logger.info("Image Conversion Utility")
    logger.info("="*60)
    
    # Load image
    try:
        logger.info(f"Loading image: {args.input}")
        image = load_image(args.input)
        logger.info(f"Original size: {image.shape[1]}x{image.shape[0]}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return 1
    
    # Apply transformations
    if args.square:
        logger.info("Cropping to square...")
        image = crop_to_square(image)
        logger.info(f"Cropped size: {image.shape[1]}x{image.shape[0]}")
    
    if args.size:
        logger.info(f"Resizing to {args.size[0]}x{args.size[1]}...")
        image = cv2.resize(image, tuple(args.size), interpolation=cv2.INTER_LANCZOS4)
    
    if args.normalize:
        logger.info("Normalizing brightness and contrast...")
        image = normalize_image(image)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_converted{input_path.suffix}"
    
    # Save image
    try:
        logger.info(f"Saving to: {output_path}")
        
        # Set JPEG quality if applicable
        if str(output_path).lower().endswith(('.jpg', '.jpeg')):
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        else:
            save_image(image, output_path)
        
        logger.info("✓ Conversion complete")
        logger.info(f"Final size: {image.shape[1]}x{image.shape[0]}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
