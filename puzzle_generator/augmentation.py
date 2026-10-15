"""
Augmentation Module
===================
Menerapkan rotasi acak dan pengacakan posisi pada potongan puzzle.
"""

import numpy as np
import cv2
import random
from typing import List, Tuple, Optional


def apply_random_rotation(
    piece: np.ndarray,
    angles: Optional[List[int]] = None
) -> Tuple[np.ndarray, int]:
    """Apply random rotation to a puzzle piece.
    
    For 90-degree increments, uses cv2.rotate() which is lossless and fast.
    For arbitrary angles, uses affine transformation with transparent background.
    
    Args:
        piece: Image array (RGB or RGBA numpy ndarray).
        angles: List of possible rotation angles in degrees.
                Defaults to [0, 90, 180, 270].
    
    Returns:
        Tuple of (rotated_piece, rotation_angle_applied).
    
    Note:
        The rotation angle is stored so the Puzzle Solver (Tahap 2)
        can determine the correct orientation.
    """
    if angles is None:
        angles = [0, 90, 180, 270]
    
    angle = random.choice(angles)
    
    if angle == 0:
        return piece.copy(), 0
    elif angle == 90:
        rotated = cv2.rotate(piece, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(piece, cv2.ROTATE_180)
    elif angle == 270:
        rotated = cv2.rotate(piece, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Arbitrary angle rotation with proper background handling
        rotated = _rotate_arbitrary(piece, angle)
    
    return rotated, angle


def _rotate_arbitrary(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by an arbitrary angle, expanding canvas to fit.
    
    Args:
        image: Input image (RGB or RGBA).
        angle: Rotation angle in degrees (clockwise).
    
    Returns:
        Rotated image with expanded canvas.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Rotation matrix (negative angle because cv2 uses CCW)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # Calculate new bounding box dimensions
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    
    # Adjust translation to center the image
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # Choose border value based on number of channels
    if len(image.shape) == 3 and image.shape[2] == 4:
        border_value = (0, 0, 0, 0)  # Transparent for RGBA
    elif len(image.shape) == 3:
        border_value = (255, 255, 255)  # White for RGB
    else:
        border_value = 255  # White for grayscale
    
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    
    return rotated


def apply_batch_rotation(
    pieces: List[np.ndarray],
    angles: Optional[List[int]] = None
) -> List[Tuple[np.ndarray, int]]:
    """Apply random rotation to a batch of puzzle pieces.
    
    Args:
        pieces: List of piece images (numpy arrays).
        angles: Possible rotation angles (default [0, 90, 180, 270]).
    
    Returns:
        List of (rotated_piece, rotation_angle) tuples.
    """
    return [apply_random_rotation(piece, angles) for piece in pieces]


def shuffle_pieces(pieces_with_metadata: List[dict]) -> List[dict]:
    """Shuffle puzzle pieces randomly.
    
    Creates a new shuffled list without modifying the original.
    Each piece dict retains its 'original_index' for tracking.
    
    Args:
        pieces_with_metadata: List of piece dicts with keys like
            'image', 'original_index', 'row', 'col', 'rotation'.
    
    Returns:
        New list with pieces in random order.
    """
    shuffled = pieces_with_metadata.copy()
    random.shuffle(shuffled)
    return shuffled
