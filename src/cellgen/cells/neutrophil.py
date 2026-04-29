from cellgen.nucleus.neutrophil import NeutrophilNucleus
from .base import Cell, CellParameters, BoundingBox
from .cell_body import CellBody, CellBodyParams
from ..utils.colormap import CellSolidColors
import numpy as np
from typing import Tuple
from ..nucleus.neutrophil import NeutrophilNucleusParams

class Neutrophil(Cell):
    """Neutrophil cell generator."""

    def __init__(self, params: CellParameters):
        super().__init__(params)
        self.cell_body = CellBody(CellBodyParams(
            size=params.size,
            noise_amount=params.noise_amount,
            sigma=params.sigma
        ))

        self.nucleus = NeutrophilNucleus(params.nucleus_params)

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
        """Generate cell image."""
        # Generate cell body
        cell_body_rgba = self.cell_body.generate(color_strategy=self._neutrophil_coloring)

        # Generate nucleus
        nucleus_mask = self.nucleus.generate()
        nucleus_rgba = self.nucleus.color_nucleus(nucleus_mask)

        # Fit nucleus within cell body
        fitted_nucleus = self.fit_nucleus_to_cell_body(nucleus_rgba, cell_body_rgba)

        # Combine layers
        combined = self.combine_nucleus_and_cell_body(fitted_nucleus, cell_body_rgba)

        return combined

    def get_name(self) -> str:
        return "neutrophil"

    def get_bounding_box(self, image: np.ndarray) -> BoundingBox:
        """Get bounding box using cell body's calculation."""
        return self.cell_body.get_bounding_box(image)
