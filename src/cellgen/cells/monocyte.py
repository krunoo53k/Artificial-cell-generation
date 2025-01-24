from .base import Cell, CellParameters, BoundingBox
from .cell_body import CellBody, CellBodyParams
from ..utils.colormap import CellColorMaps
import numpy as np
from typing import Tuple

class Monocyte(Cell):
    """Monocyte cell generator."""

    def __init__(self, params: CellParameters):
        super().__init__(params)
        self.cell_body = CellBody(CellBodyParams(
            size=params.size,
            noise_amount=params.noise_amount,
            sigma=params.sigma
        ))

    def _monocyte_coloring(self, image: np.ndarray) -> np.ndarray:
        """Specific coloring strategy for monocytes."""
        # Create RGBA array
        rgba = np.zeros((*image.shape, 4))

        # Apply colormap to RGB channels
        rgb = CellColorMaps.MONOCYTE(image)[:, :, :3]
        rgba[..., :3] = rgb

        # Use original image as alpha channel
        rgba[..., 3] = image

        # Set fully transparent pixels to black
        rgba[image < 0.01] = [0, 0, 0, 0]

        return rgba

    def generate(self) -> np.ndarray:
        """Generate monocyte cell image."""
        # Generate cell body with monocyte-specific coloring
        cell_image = self.cell_body.generate(color_strategy=self._monocyte_coloring)
        return cell_image

    def get_name(self) -> str:
        return "monocyte"

    def get_bounding_box(self, image: np.ndarray) -> BoundingBox:
        """Get bounding box using cell body's calculation."""
        return self.cell_body.get_bounding_box(image)
