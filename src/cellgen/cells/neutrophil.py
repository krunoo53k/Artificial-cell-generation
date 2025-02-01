from .base import Cell, CellParameters, BoundingBox
from .cell_body import CellBody, CellBodyParams
from ..utils.colormap import CellSolidColors
import numpy as np
from typing import Tuple

class Neutrophil(Cell):
    """Neutrophil cell generator."""

    def __init__(self, params: CellParameters):
        super().__init__(params)
        self.cell_body = CellBody(CellBodyParams(
            size=params.size,
            noise_amount=params.noise_amount,
            sigma=params.sigma
        ))

    def _neutrophil_coloring(self, image: np.ndarray) -> np.ndarray:
        """Specific coloring strategy for neutrophils."""

        # Create RGBA array
        rgba = np.zeros((*image.shape, 4))

        # Set solid color for all non-zero pixels
        cytoplasm_color = CellSolidColors.NEUTROPHIL_CYTOPLASM
        mask = image > 0
        rgba[mask, 0] = cytoplasm_color[0]
        rgba[mask, 1] = cytoplasm_color[1]
        rgba[mask, 2] = cytoplasm_color[2]
        rgba[..., 3] = image

        return rgba

    def generate(self) -> np.ndarray:
        """Generate neutrophil cell image."""
        # Generate cell body with neutrophil-specific coloring
        cell_image = self.cell_body.generate(color_strategy=self._neutrophil_coloring)
        return cell_image

    def get_name(self) -> str:
        return "neutrophil"

    def get_bounding_box(self, image: np.ndarray) -> BoundingBox:
        """Get bounding box using cell body's calculation."""
        return self.cell_body.get_bounding_box(image)
