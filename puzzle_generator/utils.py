"""
Utility Functions
=================
Helper functions for image conversion and display in the Puzzle Generator.
"""

import numpy as np
from PIL import Image, ImageTk
import cv2
from typing import Optional


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR/BGRA numpy array to PIL RGB/RGBA Image.
    
    Args:
        image: OpenCV image (BGR or BGRA format).
    
    Returns:
        PIL Image in RGB or RGBA mode.
    """
    if len(image.shape) == 2:
        # Grayscale
        return Image.fromarray(image, mode='L')
    elif image.shape[2] == 4:
        # BGRA → RGBA
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgba, mode='RGBA')
    else:
        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, mode='RGB')


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR numpy array.
    
    Args:
        image: PIL Image (RGB or RGBA mode).
    
    Returns:
        OpenCV image as numpy array in BGR format.
    """
    arr = np.array(image)
    if image.mode == 'RGBA':
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    elif image.mode == 'RGB':
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif image.mode == 'L':
        return arr
    else:
        # Convert to RGB first
        rgb = image.convert('RGB')
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def resize_to_fit(
    image: np.ndarray,
    max_width: int,
    max_height: int
) -> np.ndarray:
    """Resize image to fit within max dimensions while preserving aspect ratio.
    
    Uses INTER_AREA for downscaling (best quality) and INTER_LINEAR for upscaling.
    
    Args:
        image: Input image (numpy array, any number of channels).
        max_width: Maximum width in pixels.
        max_height: Maximum height in pixels.
    
    Returns:
        Resized image (numpy array).
    """
    h, w = image.shape[:2]
    
    if w <= 0 or h <= 0:
        return image
    
    # Calculate scale to fit within bounds
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    if abs(scale - 1.0) < 0.01:
        return image
    
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    return resized


def numpy_to_photoimage(image: np.ndarray) -> ImageTk.PhotoImage:
    """Convert numpy array (RGB or RGBA) to tkinter-compatible PhotoImage.
    
    Args:
        image: Numpy array in RGB or RGBA format.
    
    Returns:
        ImageTk.PhotoImage ready for tkinter display.
    """
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image, mode='L')
    elif image.shape[2] == 4:
        pil_img = Image.fromarray(image, mode='RGBA')
    else:
        pil_img = Image.fromarray(image, mode='RGB')
    
    return ImageTk.PhotoImage(pil_img)
