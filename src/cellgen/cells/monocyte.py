from .base import Cell, CellParameters, BoundingBox
from .cell_body import CellBody, CellBodyParams
from ..utils.colormap import CellColorMaps
import numpy as np
from typing import Tuple
import cv2

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

        # Calculate sizes and positions
        cell_size = self.params.size
        nucleus_size = int(cell_size * np.random.uniform(0.3, 0.9))

        # Create smaller nucleus
        nucleus = cv2.resize(cell_image, dsize=(nucleus_size, nucleus_size), interpolation=cv2.INTER_CUBIC)

        # Calculate center position
        start_x = (cell_size - nucleus_size) // 2
        start_y = (cell_size - nucleus_size) // 2

        # Create a copy of cell_image with reduced opacity
        cell_image *= 0.5

        # Blend images using alpha compositing
        # This function should probably be split into utils during future refactoring
        for i in range(4):  # For each channel (RGBA)
            cell_image[start_y:start_y+nucleus_size,
                      start_x:start_x+nucleus_size,
                      i] = cell_image[start_y:start_y+nucleus_size,
                                    start_x:start_x+nucleus_size,
                                    i] * (1 - nucleus[..., 3]) + nucleus[..., i] * nucleus[..., 3]

        return cell_image

    def get_name(self) -> str:
        return "monocyte"

    def get_bounding_box(self, image: np.ndarray) -> BoundingBox:
        """Get bounding box using cell body's calculation."""
        return self.cell_body.get_bounding_box(image)
