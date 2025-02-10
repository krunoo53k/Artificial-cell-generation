from typing import Tuple, List, Optional
import numpy as np
from ..background import Background, BackgroundParams
from ..cells.neutrophil import Neutrophil
from ..cells.monocyte import Monocyte
from ..cells.base import CellParameters
from cv2 import resize

class CellPlacement:
    """Handles placement of cells on backgrounds."""

    @staticmethod
    def generate_cell_on_background(
        cell_type: str = "neutrophil",
        background_params: Optional[BackgroundParams] = None,
        cell_params: Optional[CellParameters] = None
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Generate a white blood cell on background and return YOLO format bounding box."""
        # Define final crop dimensions
        CROP_HEIGHT = 360
        CROP_WIDTH = 363

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

        # Calculate crop region
        start_y = (background.shape[0] - CROP_HEIGHT) // 2
        start_x = (background.shape[1] - CROP_WIDTH) // 2
        end_y = start_y + CROP_HEIGHT
        end_x = start_x + CROP_WIDTH

        # Resize a higher resolution cell image using cv2
        cell = resize(cell, (128, 128))

        # Get cell dimensions
        cell_height, cell_width = cell.shape[:2]

        # Calculate random position within crop region
        padding = 10  # Minimum distance from edges
        x = np.random.randint(
            start_x + padding,
            end_x - cell_width - padding
        )
        y = np.random.randint(
            start_y + padding,
            end_y - cell_height - padding
        )

        # Create mask for non-zero pixels
        mask = cell[..., 3] > 0.1

        # Place cell on background using alpha compositing
        for c in range(3):
            background[y:y+cell_height, x:x+cell_width, c][mask] = \
                cell[..., c][mask] * cell[..., 3][mask] + \
                background[y:y+cell_height, x:x+cell_width, c][mask] * (1 - cell[..., 3][mask])

        # Crop image
        cropped = background[start_y:end_y, start_x:end_x]

        # Calculate bounding box relative to cropped image
        x_center = (x - start_x + cell_width/2) / CROP_WIDTH
        y_center = (y - start_y + cell_height/2) / CROP_HEIGHT
        width = cell_width / CROP_WIDTH
        height = cell_height / CROP_HEIGHT

        return cropped, (x_center, y_center, width, height)
