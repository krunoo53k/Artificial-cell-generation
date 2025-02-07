from typing import Tuple, List, Optional
import numpy as np
from ..background import Background, BackgroundParams
from ..cells.neutrophil import Neutrophil
from ..cells.monocyte import Monocyte
from ..cells.base import CellParameters

class CellPlacement:
    """Handles placement of cells on backgrounds."""

    @staticmethod
    def generate_cell_on_background(
        cell_type: str = "neutrophil",
        background_params: Optional[BackgroundParams] = None,
        cell_params: Optional[CellParameters] = None
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Generate a white blood cell on background and return YOLO format bounding box."""
        # Use provided params or defaults
        bg_params = background_params or BackgroundParams(
            width=512,
            height=512,
            cell_size=128,
            min_cells=10,
            max_cells=20
        )
        background = Background(bg_params).generate()

        # Generate appropriate cell type
        cell_params = cell_params or CellParameters(size=256)
        if cell_type == "neutrophil":
            cell = Neutrophil(cell_params).generate()
        elif cell_type == "monocyte":
            cell = Monocyte(cell_params).generate()
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")

        # Get cell dimensions
        cell_height, cell_width = cell.shape[:2]

        # Calculate random position for cell
        padding = 20
        x = np.random.randint(padding, background.shape[1] - cell_width - padding)
        y = np.random.randint(padding, background.shape[0] - cell_height - padding)

        # Create mask for non-zero pixels
        mask = cell[..., 3] > 0.1

        # Place cell on background using alpha compositing
        for c in range(3):
            background[y:y+cell_height, x:x+cell_width, c][mask] = \
                cell[..., c][mask] * cell[..., 3][mask] + \
                background[y:y+cell_height, x:x+cell_width, c][mask] * (1 - cell[..., 3][mask])

        # Calculate initial bounding box
        x_center = (x + cell_width/2) / background.shape[1]
        y_center = (y + cell_height/2) / background.shape[0]
        width = cell_width / background.shape[1]
        height = cell_height / background.shape[0]

        # Crop to 360x363 from center
        start_y = (background.shape[0] - 360) // 2
        start_x = (background.shape[1] - 363) // 2
        cropped = background[start_y:start_y+360, start_x:start_x+363]

        # Adjust bounding box for cropped image
        x_center = (x_center * background.shape[1] - start_x) / 363
        y_center = (y_center * background.shape[0] - start_y) / 360
        width = width * background.shape[1] / 363
        height = height * background.shape[0] / 360

        return cropped, (x_center, y_center, width, height)
