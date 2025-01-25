from dataclasses import dataclass
import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2
from ..utils.blob import BlobGenerator
from ..utils.colormap import CellColorMaps

@dataclass
class ErythrocyteParams:
    """Parameters for red blood cell generation."""
    size: int = 128
    points: int = 7
    scale: float = 0.8
    center_opacity: float = 0.6  # Controls how transparent the center becomes

class Erythrocyte:
    """Generator for individual red blood cells."""

    def __init__(self, params: ErythrocyteParams = None):
        self.params = params or ErythrocyteParams()

    def generate(self) -> np.ndarray:
        """Generate a single red blood cell image.

        Returns:
            np.ndarray: RGBA image with alpha channel based on distance transform
        """
        # Generate base blob mask
        x_vals, y_vals = BlobGenerator.generate_smooth_blob(
            n_points=self.params.points,
            scale=self.params.scale
        )

        # Create mask image
        mask = np.zeros((self.params.size, self.params.size))
        points = np.array(list(zip(x_vals, y_vals)))
        points += 2  # Shift to positive range
        points /= 4  # Normalize
        points *= self.params.size
        points = points.astype(np.int32)
        points = points.reshape((-1, 1, 2))

        # Fill blob
        cv2.fillPoly(mask, [points], color=1)

        # Apply distance transform for alpha channel
        dist = distance_transform_edt(mask)
        dist = dist / dist.max()  # Normalize to [0,1]

        # Create donut-like alpha pattern
        # This creates a more pronounced center depression
        center_x = self.params.size // 2
        center_y = self.params.size // 2
        Y, X = np.ogrid[:self.params.size, :self.params.size]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(2) * (self.params.size/2)
        normalized_dist = dist_from_center / max_dist

        # Create ring pattern
        ring = np.sin(normalized_dist * np.pi) * mask

        # Add some random variation
        random_variation = np.random.uniform(0.8, 1.2)
        ring = np.clip(ring * random_variation, 0, 1)

        # Combine with original distance transform for natural edges
        alpha = ring * (1.0 - (dist * self.params.center_opacity))
        alpha = mask * alpha  # Apply mask to alpha

        # Add some random opacity variation
        overall_opacity = np.random.uniform(0.7, 1.0)
        alpha *= overall_opacity

        # Create RGBA image and apply cell coloring
        rgba = np.zeros((self.params.size, self.params.size, 4))
        colored = CellColorMaps.RED_BLOOD_CELL_BODY(mask)
        rgba[..., :3] = colored[..., :3]  # Apply colors
        rgba[..., 3] = alpha  # Set alpha channel

        return rgba
