from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from ..nucleus.neutrophil import NeutrophilNucleusParams
import cv2

@dataclass
class CellParameters:
    """Parameters for cell generation."""
    size: int = 512
    noise_amount: float = 0.2
    sigma: float = 1.0
    intensity: float = 255
    nucleus_params: Optional[NeutrophilNucleusParams] = None

    def __post_init__(self):
        # Initialize default nucleus params if none provided
        if self.nucleus_params is None:
            self.nucleus_params = NeutrophilNucleusParams(
                size=self.size,
                num_segments=3,
                segment_frequency=2.0,
                segment_amplitude=0.3,
                base_thickness=0.5,
                connection_thickness=0.15,
                compactness=0.4,
                curve_randomness=0.2
            )

@dataclass
class BoundingBox:
    """Bounding box in YOLO format."""
    x: float  # center x normalized
    y: float  # center y normalized
    width: float  # normalized
    height: float  # normalized

class Cell(ABC):
    """Abstract base class for all cell types."""

    def __init__(self, params: CellParameters):
        self.params = params

    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generate cell image.

        Returns:
            np.ndarray: Generated cell image
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get cell type name.

        Returns:
            str: Cell type name
        """
        pass

    def combine_nucleus_and_cell_body(self, nucleus_rgba: np.ndarray, cell_body_rgba: np.ndarray) -> np.ndarray:
        """Combines nucleus and cell body images with proper alpha blending.

        Args:
            nucleus_rgba: RGBA image of the nucleus
            cell_body_rgba: RGBA image of the cell body

        Returns:
            np.ndarray: Combined RGBA image
        """
        # Ensure images are same size
        if nucleus_rgba.shape != cell_body_rgba.shape:
            raise ValueError("Nucleus and cell body must be same size")

        # Create output image
        result = np.zeros_like(cell_body_rgba)

        # Alpha blending formula
        # C = α_f * C_f + (1 - α_f) * α_b * C_b
        # where f = foreground (nucleus), b = background (cell body)

        alpha_nucleus = nucleus_rgba[..., 3:4]
        alpha_cell = cell_body_rgba[..., 3:4]

        # Blend colors
        for i in range(3):
            result[..., i] = (nucleus_rgba[..., i] * alpha_nucleus[..., 0] +
                             cell_body_rgba[..., i] * alpha_cell[..., 0] * (1 - alpha_nucleus[..., 0]))

        # Combine alpha channels
        result[..., 3] = alpha_nucleus[..., 0] + alpha_cell[..., 0] * (1 - alpha_nucleus[..., 0])

        return result

    def fit_nucleus_to_cell_body(self, nucleus_rgba: np.ndarray, cell_body_rgba: np.ndarray,
                                padding: float = 0.0) -> np.ndarray:  # Reduced padding from 0.1 to 0.05
        """Fits nucleus within cell body boundaries with padding."""
        # Get cell body mask from alpha channel
        cell_mask = cell_body_rgba[..., 3] > 0
        nucleus_mask = nucleus_rgba[..., 3] > 0

        # Find cell body boundaries
        cell_rows = np.any(cell_mask, axis=1)
        cell_cols = np.any(cell_mask, axis=0)
        cell_top, cell_bottom = np.where(cell_rows)[0][[0, -1]]
        cell_left, cell_right = np.where(cell_cols)[0][[0, -1]]

        # Calculate available space in cell (with padding)
        cell_height = cell_bottom - cell_top
        cell_width = cell_right - cell_left

        # Increase available space by using a larger fraction of the cell
        size_factor = 0.8  # This controls how much of the cell the nucleus can occupy
        available_height = int(cell_height * size_factor)
        available_width = int(cell_width * size_factor)

        # Resize nucleus to fit available space
        # Use the smaller dimension to maintain aspect ratio
        aspect_ratio = nucleus_rgba.shape[1] / nucleus_rgba.shape[0]
        if available_height * aspect_ratio <= available_width:
            # Height is the limiting factor
            new_height = available_height
            new_width = int(available_height * aspect_ratio)
        else:
            # Width is the limiting factor
            new_width = available_width
            new_height = int(available_width / aspect_ratio)

        # Resize nucleus
        nucleus_rgba = cv2.resize(nucleus_rgba, (new_width, new_height))

        # Create output image same size as cell body
        result = np.zeros_like(cell_body_rgba)

        # Calculate centered position
        y_start = cell_top + (cell_height - new_height) // 2
        x_start = cell_left + (cell_width - new_width) // 2

        # Ensure coordinates are integers
        y_start = int(y_start)
        x_start = int(x_start)

        # Place nucleus
        result[y_start:y_start + new_height,
               x_start:x_start + new_width] = nucleus_rgba

        return result
