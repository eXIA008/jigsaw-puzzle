"""
Piece Cutter Module
===================
Memotong gambar menjadi potongan puzzle grid M×M.
Supports both simple rectangular cutting and jigsaw-shaped cutting.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from puzzle_generator.jigsaw_shape import (
    EdgeConfig,
    generate_edge_config,
    extract_jigsaw_piece,
)


@dataclass
class PuzzleConfig:
    """Configuration for puzzle grid layout.
    
    Attributes:
        image_height: Original image height.
        image_width: Original image width.
        grid_size: M value (grid M×M).
        border_ratio: Border thickness as fraction of image dimensions.
        border_x: Border width in pixels (left and right).
        border_y: Border height in pixels (top and bottom).
        puzzle_w: Width of the puzzle area (center, excluding borders).
        puzzle_h: Height of the puzzle area (center, excluding borders).
        cell_w: Width of each grid cell.
        cell_h: Height of each grid cell.
        total_pieces: Total number of pieces (M×M).
    """
    image_height: int
    image_width: int
    grid_size: int
    border_ratio: float
    border_x: int
    border_y: int
    puzzle_w: int
    puzzle_h: int
    cell_w: int
    cell_h: int
    total_pieces: int


def calculate_config(
    image: np.ndarray,
    grid_size: int,
    border_ratio: float = 0.1
) -> PuzzleConfig:
    """Calculate puzzle grid configuration based on image dimensions.
    
    Args:
        image: Source image (numpy array).
        grid_size: M value for M×M grid.
        border_ratio: Border thickness as fraction of image dimensions (default 0.1 = 10%).
    
    Returns:
        PuzzleConfig with all calculated dimensions.
    
    Raises:
        ValueError: If grid_size < 2 or image is too small.
    """
    if grid_size < 2:
        raise ValueError("Grid size must be at least 2")
    
    H, W = image.shape[:2]
    
    if H < 100 or W < 100:
        raise ValueError(f"Image too small ({W}x{H}). Minimum size is 100x100 pixels.")
    
    # Calculate border dimensions
    border_x = int(W * border_ratio)
    border_y = int(H * border_ratio)
    
    # Ensure border is at least a few pixels
    border_x = max(border_x, 5)
    border_y = max(border_y, 5)
    
    # Puzzle area (center region)
    puzzle_w = W - 2 * border_x
    puzzle_h = H - 2 * border_y
    
    # Cell dimensions (integer division, may have remainder)
    cell_w = puzzle_w // grid_size
    cell_h = puzzle_h // grid_size
    
    if cell_w < 20 or cell_h < 20:
        raise ValueError(
            f"Pieces too small ({cell_w}x{cell_h}px). "
            f"Reduce grid size or use a larger image."
        )
    
    # Adjust puzzle area to be exact multiple of grid
    puzzle_w = cell_w * grid_size
    puzzle_h = cell_h * grid_size
    
    # Re-center border to account for adjusted puzzle area
    border_x = (W - puzzle_w) // 2
    border_y = (H - puzzle_h) // 2
    
    return PuzzleConfig(
        image_height=H,
        image_width=W,
        grid_size=grid_size,
        border_ratio=border_ratio,
        border_x=border_x,
        border_y=border_y,
        puzzle_w=puzzle_w,
        puzzle_h=puzzle_h,
        cell_w=cell_w,
        cell_h=cell_h,
        total_pieces=grid_size * grid_size,
    )


def cut_pieces_rectangular(
    image: np.ndarray,
    config: PuzzleConfig
) -> List[Dict]:
    """Cut image into rectangular pieces (no jigsaw shapes).
    
    Args:
        image: Source image (RGB numpy array).
        config: PuzzleConfig with grid dimensions.
    
    Returns:
        List of dicts, each containing:
            - 'image': piece image (numpy array, RGB)
            - 'row': row index (0-based)
            - 'col': column index (0-based)  
            - 'original_index': linear index (row * M + col)
            - 'rotation': 0 (no rotation applied yet)
    """
    M = config.grid_size
    pieces = []
    
    for row in range(M):
        for col in range(M):
            y0 = config.border_y + row * config.cell_h
            x0 = config.border_x + col * config.cell_w
            y1 = y0 + config.cell_h
            x1 = x0 + config.cell_w
            
            piece_img = image[y0:y1, x0:x1].copy()
            
            pieces.append({
                'image': piece_img,
                'row': row,
                'col': col,
                'original_index': row * M + col,
                'rotation': 0,
            })
    
    return pieces


def cut_pieces_jigsaw(
    image: np.ndarray,
    config: PuzzleConfig,
    edge_config: Optional[EdgeConfig] = None
) -> Tuple[List[Dict], EdgeConfig]:
    """Cut image into jigsaw-shaped pieces with tabs and blanks.
    
    Args:
        image: Source image (RGB numpy array).
        config: PuzzleConfig with grid dimensions.
        edge_config: Optional pre-generated EdgeConfig. If None, generates randomly.
    
    Returns:
        Tuple of:
            - List of piece dicts (same structure as cut_pieces_rectangular,
              but 'image' is RGBA with transparency)
            - EdgeConfig used for the cutting (for frame drawing)
    """
    M = config.grid_size
    
    if edge_config is None:
        edge_config = generate_edge_config(M)
    
    tab_size = int(0.15 * min(config.cell_w, config.cell_h))
    pieces = []
    
    for row in range(M):
        for col in range(M):
            piece_rgba = extract_jigsaw_piece(
                image=image,
                row=row,
                col=col,
                edge_config=edge_config,
                cell_w=config.cell_w,
                cell_h=config.cell_h,
                offset_x=config.border_x,
                offset_y=config.border_y,
                tab_size=tab_size,
            )
            
            pieces.append({
                'image': piece_rgba,
                'row': row,
                'col': col,
                'original_index': row * M + col,
                'rotation': 0,
            })
    
    return pieces, edge_config


def cut_pieces(
    image: np.ndarray,
    config: PuzzleConfig,
    jigsaw: bool = False,
    edge_config: Optional[EdgeConfig] = None
) -> Tuple[List[Dict], Optional[EdgeConfig]]:
    """Main function to cut image into puzzle pieces.
    
    Args:
        image: Source image (RGB numpy array).
        config: PuzzleConfig with grid dimensions.
        jigsaw: If True, create jigsaw-shaped pieces. If False, rectangular.
        edge_config: Optional EdgeConfig for jigsaw mode.
    
    Returns:
        Tuple of (pieces_list, edge_config_or_None).
    """
    if jigsaw:
        return cut_pieces_jigsaw(image, config, edge_config)
    else:
        return cut_pieces_rectangular(image, config), None
