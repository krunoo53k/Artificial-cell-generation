from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class CellParameters:
    """Parameters for cell generation."""
    size: int = 512
    noise_amount: float = 0.2
    sigma: float = 1.0
    intensity: float = 255

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
