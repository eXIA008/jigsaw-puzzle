"""
Frame Generator Module
======================
Membuat bingkai (frame) puzzle dari gambar asli.
Bingkai menampilkan tepi/border gambar asli, dengan area tengah dikosongkan
dan diberi garis grid sebagai panduan penempatan potongan puzzle.
"""

import numpy as np
import cv2
from typing import Optional, Tuple

from puzzle_generator.piece_cutter import PuzzleConfig
from puzzle_generator.jigsaw_shape import EdgeConfig, draw_jigsaw_grid


def generate_frame(
    image: np.ndarray,
    config: PuzzleConfig,
    jigsaw: bool = False,
    edge_config: Optional[EdgeConfig] = None,
    fill_color: Tuple[int, int, int] = (220, 220, 220),
    grid_color: Tuple[int, int, int] = (100, 100, 100),
    grid_thickness: int = 2,
) -> np.ndarray:
    """Generate a puzzle frame from the original image.
    
    The frame preserves the border area (edges) of the original image
    and replaces the center puzzle area with a neutral background
    overlaid with grid lines showing where pieces should be placed.
    
    Args:
        image: Source image (RGB numpy array, H×W×3).
        config: PuzzleConfig with grid and border dimensions.
        jigsaw: If True, draw jigsaw-style outlines instead of straight grid.
        edge_config: Required if jigsaw=True; the EdgeConfig used for cutting pieces.
        fill_color: RGB color to fill the center area (default: light gray).
        grid_color: RGB color for grid lines (default: dark gray).
        grid_thickness: Thickness of grid lines in pixels.
    
    Returns:
        Frame image (RGB numpy array, same size as original).
    """
    H, W = image.shape[:2]
    frame = image.copy()
    
    # Define the puzzle area (center region)
    x0 = config.border_x
    y0 = config.border_y
    x1 = x0 + config.puzzle_w
    y1 = y0 + config.puzzle_h
    
    # Fill center area with neutral color
    frame[y0:y1, x0:x1] = fill_color
    
    if jigsaw and edge_config is not None:
        # Draw jigsaw-style outlines
        tab_size = int(0.15 * min(config.cell_w, config.cell_h))
        draw_jigsaw_grid(
            canvas=frame,
            edge_config=edge_config,
            cell_w=config.cell_w,
            cell_h=config.cell_h,
            offset_x=config.border_x,
            offset_y=config.border_y,
            tab_size=tab_size,
            color=grid_color,
            thickness=grid_thickness,
        )
    else:
        # Draw simple rectangular grid lines
        _draw_rectangular_grid(
            frame, config, grid_color, grid_thickness
        )
    
    # Draw outer border rectangle around the puzzle area
    cv2.rectangle(
        frame,
        (x0, y0),
        (x1 - 1, y1 - 1),
        grid_color,
        thickness=grid_thickness + 1,
    )
    
    return frame


def _draw_rectangular_grid(
    frame: np.ndarray,
    config: PuzzleConfig,
    color: Tuple[int, int, int] = (100, 100, 100),
    thickness: int = 2,
) -> None:
    """Draw simple rectangular grid lines on the frame.
    
    Args:
        frame: Image to draw on (modified in place).
        config: PuzzleConfig with grid dimensions.
        color: Line color (RGB).
        thickness: Line thickness.
    """
    M = config.grid_size
    x0 = config.border_x
    y0 = config.border_y
    
    # Draw vertical lines
    for col in range(1, M):
        x = x0 + col * config.cell_w
        cv2.line(
            frame,
            (x, y0),
            (x, y0 + config.puzzle_h),
            color,
            thickness,
        )
    
    # Draw horizontal lines
    for row in range(1, M):
        y = y0 + row * config.cell_h
        cv2.line(
            frame,
            (x0, y),
            (x0 + config.puzzle_w, y),
            color,
            thickness,
        )
