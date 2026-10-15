"""
Jigsaw Shape Generator Module
=============================
Generates jigsaw puzzle piece outlines with tabs (pasak) and blanks (celah)
using smooth Bezier-like curves for a realistic puzzle appearance.
"""

import numpy as np
import cv2
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class EdgeConfig:
    """Stores edge directions for the jigsaw puzzle grid.
    
    Attributes:
        h_edges: 2D list [M-1][M] — horizontal edges between row i and row i+1.
                 Value +1: tab protrudes downward; -1: tab protrudes upward.
        v_edges: 2D list [M][M-1] — vertical edges between col j and col j+1.
                 Value +1: tab protrudes rightward; -1: tab protrudes leftward.
        grid_size: The M value (grid M×M).
    """
    h_edges: List[List[int]] = field(default_factory=list)
    v_edges: List[List[int]] = field(default_factory=list)
    grid_size: int = 3


def generate_edge_config(grid_size: int) -> EdgeConfig:
    """Generate random edge directions for all internal edges.
    
    Args:
        grid_size: The M value (puzzle will be M×M pieces).
    
    Returns:
        EdgeConfig with randomly assigned tab directions.
    """
    # Horizontal internal edges: (M-1) rows × M columns
    h_edges = [
        [random.choice([1, -1]) for _ in range(grid_size)]
        for _ in range(grid_size - 1)
    ]
    
    # Vertical internal edges: M rows × (M-1) columns
    v_edges = [
        [random.choice([1, -1]) for _ in range(grid_size - 1)]
        for _ in range(grid_size)
    ]
    
    return EdgeConfig(h_edges=h_edges, v_edges=v_edges, grid_size=grid_size)


def _tab_profile_points(
    start: Tuple[float, float],
    end: Tuple[float, float],
    direction: int,
    tab_size: float
) -> List[Tuple[int, int]]:
    """Generate polygon points for a jigsaw tab/blank along an edge.
    
    Creates a smooth jigsaw-style shape with a narrow neck and wider head,
    resembling a classic puzzle piece connector.
    
    Args:
        start: (x, y) start point of the edge.
        end: (x, y) end point of the edge.
        direction: +1 for tab (outward protrusion), -1 for blank (inward indentation).
        tab_size: Height of the tab in pixels.
    
    Returns:
        List of (x, y) integer points forming the tab/blank curve.
    """
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = ex - sx, ey - sy
    length = math.sqrt(dx * dx + dy * dy)
    
    if length < 1:
        return [start, end]
    
    # Unit vectors: along edge and perpendicular normal
    ux, uy = dx / length, dy / length
    # Normal direction: rotate edge vector 90° CCW, then apply direction
    nx, ny = -uy * direction, ux * direction
    
    # Jigsaw tab profile: (t_along_edge, normal_offset_as_fraction_of_tab_size)
    # Creates a shape with narrow neck and wider circular head
    profile = [
        # Straight section before tab
        (0.00, 0.00),
        (0.34, 0.00),
        # Neck entry — slight widening into neck
        (0.36, 0.00),
        (0.38, 0.10),
        (0.40, 0.25),
        # Neck to head transition — head widens beyond neck
        (0.38, 0.42),
        (0.35, 0.58),
        # Head — circular bulge
        (0.34, 0.72),
        (0.35, 0.85),
        (0.38, 0.94),
        (0.42, 1.00),
        (0.47, 1.03),
        (0.50, 1.04),  # Peak
        (0.53, 1.03),
        (0.58, 1.00),
        (0.62, 0.94),
        (0.65, 0.85),
        # Head back to neck
        (0.66, 0.72),
        (0.65, 0.58),
        (0.62, 0.42),
        # Neck exit
        (0.60, 0.25),
        (0.62, 0.10),
        (0.64, 0.00),
        # Straight section after tab
        (0.66, 0.00),
        (1.00, 0.00),
    ]
    
    points = []
    for t, d in profile:
        x = sx + t * dx + d * tab_size * nx
        y = sy + t * dy + d * tab_size * ny
        points.append((int(round(x)), int(round(y))))
    
    return points


def get_piece_outline(
    row: int,
    col: int,
    edge_config: EdgeConfig,
    cell_w: int,
    cell_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
    tab_size: Optional[int] = None
) -> List[Tuple[int, int]]:
    """Get the polygon outline for a single puzzle piece with jigsaw shapes.
    
    Constructs the outline by going clockwise: top → right → bottom → left.
    Each edge is either flat (border edges) or has a tab/blank (internal edges).
    
    Edge direction conventions:
        - For piece (row, col):
          - Top edge: uses h_edges[row-1][col]; value maps to BLANK on this piece
          - Right edge: uses v_edges[row][col]; value maps to TAB on this piece
          - Bottom edge: uses h_edges[row][col]; value maps to TAB on this piece  
          - Left edge: uses v_edges[row][col-1]; value maps to BLANK on this piece
    
    Args:
        row: Row index of the piece (0-based).
        col: Column index of the piece (0-based).
        edge_config: EdgeConfig with randomly assigned edge directions.
        cell_w: Width of each grid cell in pixels.
        cell_h: Height of each grid cell in pixels.
        offset_x: X offset for piece position (usually the border width).
        offset_y: Y offset for piece position (usually the border height).
        tab_size: Tab protrusion height in pixels. Defaults to 15% of min(cell_w, cell_h).
    
    Returns:
        List of (x, y) points forming the piece outline polygon.
    """
    M = edge_config.grid_size
    if tab_size is None:
        tab_size = int(0.15 * min(cell_w, cell_h))
    
    # Cell corners in image coordinates
    x0 = offset_x + col * cell_w
    y0 = offset_y + row * cell_h
    x1 = x0 + cell_w
    y1 = y0 + cell_h
    
    outline = []
    
    # --- Top edge (left → right) ---
    if row == 0:
        # Border edge: flat
        outline.append((x0, y0))
    else:
        # Internal edge: h_edges[row-1][col]
        # Convention: direction = h_edges value → BLANK on this piece's top
        d = edge_config.h_edges[row - 1][col]
        pts = _tab_profile_points((x0, y0), (x1, y0), d, tab_size)
        outline.extend(pts[:-1])  # Exclude last point (start of next edge)
    
    # --- Right edge (top → bottom) ---
    if col == M - 1:
        # Border edge: flat
        outline.append((x1, y0))
    else:
        # Internal edge: v_edges[row][col]
        # Convention: negate for this piece's perspective
        d = -edge_config.v_edges[row][col]
        pts = _tab_profile_points((x1, y0), (x1, y1), d, tab_size)
        outline.extend(pts[:-1])
    
    # --- Bottom edge (right → left) ---
    if row == M - 1:
        # Border edge: flat
        outline.append((x1, y1))
    else:
        # Internal edge: h_edges[row][col]
        d = -edge_config.h_edges[row][col]
        pts = _tab_profile_points((x1, y1), (x0, y1), d, tab_size)
        outline.extend(pts[:-1])
    
    # --- Left edge (bottom → top) ---
    if col == 0:
        # Border edge: flat
        outline.append((x0, y1))
    else:
        # Internal edge: v_edges[row][col-1]
        d = edge_config.v_edges[row][col - 1]
        pts = _tab_profile_points((x0, y1), (x0, y0), d, tab_size)
        outline.extend(pts[:-1])
    
    return outline


def create_piece_mask(
    outline: List[Tuple[int, int]],
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """Create a binary mask for a piece from its outline polygon.
    
    Args:
        outline: List of (x, y) points forming the piece polygon (image coords).
        bbox: Bounding box (min_x, min_y, max_x, max_y) for the mask.
    
    Returns:
        Binary mask (uint8) where 255 = inside piece, 0 = outside.
    """
    min_x, min_y, max_x, max_y = bbox
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    
    # Convert outline to local coordinates
    local_pts = np.array(
        [(p[0] - min_x, p[1] - min_y) for p in outline],
        dtype=np.int32
    )
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [local_pts], 255)
    
    return mask


def get_piece_bbox(
    outline: List[Tuple[int, int]],
    image_shape: Tuple[int, ...]
) -> Tuple[int, int, int, int]:
    """Calculate bounding box for a piece outline, clamped to image bounds.
    
    Args:
        outline: List of (x, y) points.
        image_shape: Shape of the source image (H, W, ...).
    
    Returns:
        (min_x, min_y, max_x, max_y) bounding box.
    """
    H, W = image_shape[:2]
    xs = [p[0] for p in outline]
    ys = [p[1] for p in outline]
    
    min_x = max(0, min(xs))
    min_y = max(0, min(ys))
    max_x = min(W - 1, max(xs))
    max_y = min(H - 1, max(ys))
    
    return (min_x, min_y, max_x, max_y)


def extract_jigsaw_piece(
    image: np.ndarray,
    row: int,
    col: int,
    edge_config: EdgeConfig,
    cell_w: int,
    cell_h: int,
    offset_x: int,
    offset_y: int,
    tab_size: Optional[int] = None
) -> np.ndarray:
    """Extract a single jigsaw-shaped piece from the image as an RGBA array.
    
    Args:
        image: Source image (RGB numpy array, H×W×3).
        row: Row index of the piece.
        col: Column index of the piece.
        edge_config: EdgeConfig with edge directions.
        cell_w: Width of each grid cell.
        cell_h: Height of each grid cell.
        offset_x: X offset (border width) for the puzzle area.
        offset_y: Y offset (border height) for the puzzle area.
        tab_size: Tab height in pixels (default: 15% of min(cell_w, cell_h)).
    
    Returns:
        RGBA numpy array of the piece (with transparent background).
    """
    if tab_size is None:
        tab_size = int(0.15 * min(cell_w, cell_h))
    
    # Get piece outline polygon
    outline = get_piece_outline(
        row, col, edge_config, cell_w, cell_h,
        offset_x, offset_y, tab_size
    )
    
    # Calculate bounding box
    bbox = get_piece_bbox(outline, image.shape)
    min_x, min_y, max_x, max_y = bbox
    
    # Extract the image region
    region = image[min_y:max_y + 1, min_x:max_x + 1].copy()
    
    # Create mask
    mask = create_piece_mask(outline, bbox)
    
    # Build RGBA image
    h, w = region.shape[:2]
    
    # Ensure mask matches region dimensions
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = mask[:h, :w]
    
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if len(region.shape) == 3 and region.shape[2] >= 3:
        rgba[:, :, :3] = region[:, :, :3]
    else:
        # Grayscale
        rgba[:, :, 0] = region
        rgba[:, :, 1] = region
        rgba[:, :, 2] = region
    rgba[:, :, 3] = mask
    
    return rgba


def draw_jigsaw_grid(
    canvas: np.ndarray,
    edge_config: EdgeConfig,
    cell_w: int,
    cell_h: int,
    offset_x: int,
    offset_y: int,
    tab_size: Optional[int] = None,
    color: Tuple[int, int, int] = (80, 80, 80),
    thickness: int = 1
) -> np.ndarray:
    """Draw jigsaw grid outlines on a canvas (for the frame).
    
    Args:
        canvas: Image to draw on (will be modified in place).
        edge_config: EdgeConfig with edge directions.
        cell_w: Width of each grid cell.
        cell_h: Height of each grid cell.
        offset_x: X offset for the puzzle area.
        offset_y: Y offset for the puzzle area.
        tab_size: Tab size in pixels.
        color: Line color (BGR or RGB depending on canvas format).
        thickness: Line thickness.
    
    Returns:
        The canvas with jigsaw outlines drawn.
    """
    M = edge_config.grid_size
    if tab_size is None:
        tab_size = int(0.15 * min(cell_w, cell_h))
    
    for row in range(M):
        for col in range(M):
            outline = get_piece_outline(
                row, col, edge_config, cell_w, cell_h,
                offset_x, offset_y, tab_size
            )
            pts = np.array(outline, dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
    
    return canvas
