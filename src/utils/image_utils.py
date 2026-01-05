"""
Image processing utilities
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
from pathlib import Path


def load_image(path: Union[str, Path], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load image from file with optional resizing
    
    Args:
        path: Path to image file
        target_size: Optional (width, height) tuple for resizing
        
    Returns:
        Image as numpy array in BGR format
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    try:
        # Load with PIL for better format support
        pil_image = Image.open(path)
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image = np.array(pil_image)
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if target_size:
            image = resize_image(image, target_size)
        
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save image to file
    
    Args:
        image: Image as numpy array (BGR format)
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        cv2.imwrite(str(path), image)
    except Exception as e:
        raise ValueError(f"Failed to save image to {path}: {e}")


def resize_image(
    image: np.ndarray, 
    target_size: Tuple[int, int], 
    method: str = 'lanczos'
) -> np.ndarray:
    """
    Resize image with various interpolation methods
    
    Args:
        image: Input image
        target_size: (width, height) tuple
        method: Interpolation method (nearest, bilinear, lanczos, cubic)
        
    Returns:
        Resized image
    """
    method_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'lanczos': cv2.INTER_LANCZOS4,
        'cubic': cv2.INTER_CUBIC
    }
    
    interp = method_map.get(method.lower(), cv2.INTER_LANCZOS4)
    return cv2.resize(image, target_size, interpolation=interp)


def normalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)) -> np.ndarray:
    """
    Normalize image to [-1, 1] range or custom normalization
    
    Args:
        image: Input image (0-255 range)
        mean: Mean values for normalization
        std: Standard deviation for normalization
        
    Returns:
        Normalized image
    """
    # Convert to float32 and scale to [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    # Apply normalization
    normalized = (normalized - mean) / std
    
    return normalized


def denormalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                      std: Tuple[float, float, float] = (0.5, 0.5, 0.5)) -> np.ndarray:
    """
    Denormalize image back to [0, 255] range
    
    Args:
        image: Normalized image
        mean: Mean values used in normalization
        std: Standard deviation used in normalization
        
    Returns:
        Denormalized image (0-255 range, uint8)
    """
    # Reverse normalization
    denormalized = (image * std) + mean
    
    # Scale to [0, 255] and convert to uint8
    denormalized = np.clip(denormalized * 255.0, 0, 255).astype(np.uint8)
    
    return denormalized


def crop_face_region(image: np.ndarray, landmarks: np.ndarray, 
                     scale: float = 1.5) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop face region from image based on landmarks
    
    Args:
        image: Input image
        landmarks: Face landmarks (Nx2 array)
        scale: Scale factor for crop region (1.0 = tight crop, >1.0 = include more context)
        
    Returns:
        Cropped face image and crop coordinates (x, y, w, h)
    """
    # Get bounding box from landmarks
    x_min, y_min = landmarks.min(axis=0).astype(int)
    x_max, y_max = landmarks.max(axis=0).astype(int)
    
    # Calculate center and size
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    width = x_max - x_min
    height = y_max - y_min
    
    # Scale the crop region
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)
    
    # Calculate crop coordinates
    x1 = max(0, center_x - scaled_width // 2)
    y1 = max(0, center_y - scaled_height // 2)
    x2 = min(image.shape[1], center_x + scaled_width // 2)
    y2 = min(image.shape[0], center_y + scaled_height // 2)
    
    # Crop image
    cropped = image[y1:y2, x1:x2]
    
    return cropped, (x1, y1, x2 - x1, y2 - y1)


def paste_face_region(background: np.ndarray, face: np.ndarray, 
                      coords: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Paste face region back into background image
    
    Args:
        background: Background image
        face: Face region to paste
        coords: Coordinates (x, y, w, h) where to paste
        
    Returns:
        Composite image
    """
    x, y, w, h = coords
    
    # Resize face to match target size
    face_resized = cv2.resize(face, (w, h))
    
    # Create output image
    result = background.copy()
    
    # Paste face region
    result[y:y+h, x:x+w] = face_resized
    
    return result


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, 
                   color: Tuple[int, int, int] = (0, 255, 0),
                   radius: int = 2) -> np.ndarray:
    """
    Draw landmarks on image for visualization
    
    Args:
        image: Input image
        landmarks: Landmarks to draw (Nx2 array)
        color: Color in BGR format
        radius: Point radius
        
    Returns:
        Image with landmarks drawn
    """
    result = image.copy()
    
    for x, y in landmarks:
        cv2.circle(result, (int(x), int(y)), radius, color, -1)
    
    return result


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend two images with alpha blending
    
    Args:
        img1: First image
        img2: Second image
        alpha: Blend factor (0.0 = all img1, 1.0 = all img2)
        
    Returns:
        Blended image
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    return cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)


def add_fps_overlay(image: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30),
                    color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Add FPS counter overlay to image
    
    Args:
        image: Input image
        fps: FPS value to display
        position: Text position (x, y)
        color: Text color in BGR
        
    Returns:
        Image with FPS overlay
    """
    result = image.copy()
    text = f"FPS: {fps:.1f}"
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, color, 2, cv2.LINE_AA)
    return result
