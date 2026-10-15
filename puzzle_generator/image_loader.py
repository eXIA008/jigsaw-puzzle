"""
Image Loader Module
===================
Membaca dan memvalidasi gambar untuk Puzzle Generator.
Supports RGB and Grayscale images.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict


def load_image(filepath: str) -> np.ndarray:
    """Load an image from file path.
    
    Reads the image and converts it to RGB format for consistent processing
    throughout the application. Grayscale images are converted to 3-channel RGB.
    
    Args:
        filepath: Path to the image file.
    
    Returns:
        Image as numpy array in RGB format (H×W×3, uint8).
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as an image.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not path.is_file():
        raise ValueError(f"Not a file: {filepath}")
    
    # Read with OpenCV (returns BGR)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(
            f"Cannot read image: {filepath}. "
            f"Make sure it's a valid image format (PNG, JPG, BMP, TIFF, WebP)."
        )
    
    # Convert to RGB
    if len(image.shape) == 2:
        # Grayscale → RGB (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # BGRA → RGB (drop alpha)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def get_image_info(image: np.ndarray) -> Dict:
    """Get information about an image.
    
    Args:
        image: Image as numpy array.
    
    Returns:
        Dict with keys:
            - 'width': Image width in pixels
            - 'height': Image height in pixels
            - 'channels': Number of color channels
            - 'dtype': Data type string
            - 'mode': 'RGB', 'RGBA', or 'Grayscale'
            - 'size_mb': Approximate size in megabytes
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    if channels == 1:
        mode = "Grayscale"
    elif channels == 3:
        mode = "RGB"
    elif channels == 4:
        mode = "RGBA"
    else:
        mode = f"{channels}-channel"
    
    size_bytes = image.nbytes
    size_mb = size_bytes / (1024 * 1024)
    
    return {
        'width': w,
        'height': h,
        'channels': channels,
        'dtype': str(image.dtype),
        'mode': mode,
        'size_mb': round(size_mb, 2),
    }


def validate_image(
    image: np.ndarray,
    min_size: int = 100
) -> Tuple[bool, str]:
    """Validate an image for puzzle generation.
    
    Checks that the image meets minimum requirements for being
    split into a puzzle grid.
    
    Args:
        image: Image as numpy array.
        min_size: Minimum dimension (width and height) in pixels.
    
    Returns:
        Tuple of (is_valid, message).
        If valid: (True, "OK")
        If invalid: (False, "description of the problem")
    """
    if image is None:
        return False, "Image is None"
    
    if not isinstance(image, np.ndarray):
        return False, "Image is not a numpy array"
    
    if image.size == 0:
        return False, "Image is empty (0 pixels)"
    
    h, w = image.shape[:2]
    
    if w < min_size or h < min_size:
        return False, (
            f"Image too small ({w}×{h}px). "
            f"Minimum size is {min_size}×{min_size}px."
        )
    
    if image.dtype != np.uint8:
        return False, (
            f"Unexpected dtype '{image.dtype}'. Expected uint8."
        )
    
    return True, "OK"
