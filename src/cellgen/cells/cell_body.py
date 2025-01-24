from dataclasses import dataclass
import numpy as np
from typing import Optional
from ..utils.blob import BlobGenerator
import cv2

@dataclass
class CellBodyParams:
    size: int = 512
    noise_amount: float = 0.2
    sigma: float = 1.0
    points: int = 7
    scale: float = 0.8

class CellBody:
    def __init__(self, params: Optional[CellBodyParams] = None):
        self.params = params or CellBodyParams()

    def generate(self) -> np.ndarray:
        """Generate a cell body mask."""
        # Generate smooth blob boundary
        x_vals, y_vals = BlobGenerator.generate_smooth_blob(
            n_points=self.params.points,
            scale=self.params.scale
        )

        # Create image mask
        img = np.zeros((self.params.size, self.params.size))

        # Combine x and y into points array
        points = np.array(list(zip(x_vals, y_vals)))

        # Normalize points to [0,1] range
        points += 2  # Shift to positive range
        points /= 4  # Normalize (since we added 2, divide by 4)

        # Scale to image size
        points *= self.params.size

        # Format for OpenCV
        points = points.astype(np.int32)
        points = points.reshape((-1, 1, 2))

        # Fill polygon
        cv2.fillPoly(img, [points], color=1)

        return img
