from dataclasses import dataclass
import numpy as np
from typing import Optional, Callable
from ..utils.blob import BlobGenerator, BlobParams
import cv2
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise

@dataclass
class CellBodyParams:
    size: int = 512
    noise_amount: float = 0.2
    sigma: float = 1.0
    points: int = 7
    scale: float = 0.8

@dataclass
class BoundingBox:
    """Bounding box in YOLO format."""
    x: float  # center x normalized
    y: float  # center y normalized
    width: float  # normalized
    height: float  # normalized

class CellBody:
    def __init__(self, params: Optional[CellBodyParams] = None):
        self.params = params or CellBodyParams()

    def generate(self, color_strategy: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
        """Generate a cell body image with alpha channel.

        Args:
            color_strategy: Optional function that takes grayscale image and returns RGBA image

        Returns:
            np.ndarray: RGBA image with shape (height, width, 4)
        """
        # Generate base mask
        mask = self._generate_mask()

        # Add noise and blur
        image = self._process_mask(mask)

        # Create RGBA image
        rgba = np.zeros((self.params.size, self.params.size, 4))

        # Apply coloring if strategy provided
        if color_strategy is not None:
            rgba = color_strategy(image)
        else:
            # Default grayscale with alpha
            rgba[..., :3] = np.stack((image,) * 3, axis=-1)
            rgba[..., 3] = image

        return rgba

    def _generate_mask(self) -> np.ndarray:
        """Generate basic cell body mask."""
        # Generate smooth blob boundary
        x_vals, y_vals = BlobGenerator.generate_smooth_blob(
            BlobParams(
                n_points=self.params.points,
                scale=self.params.scale,
                interpolation_points=100
            )
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

    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Add noise and blur to mask."""
        # Add noise
        img = random_noise(mask, 'pepper', amount=self.params.noise_amount)

        # Apply Gaussian blur
        img = gaussian_filter(img, sigma=self.params.sigma)

        return img

    def get_bounding_box(self, image: np.ndarray) -> BoundingBox:
        """Calculate bounding box for the cell.

        Args:
            image: RGBA cell image

        Returns:
            BoundingBox: Bounding box in YOLO format
        """
        # Use alpha channel for bounding box calculation
        alpha = image[..., 3]

        # Find non-zero pixels
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Calculate dimensions
        height = (ymax - ymin) / image.shape[0]
        width = (xmax - xmin) / image.shape[1]

        # Calculate center points
        x_center = (xmin + xmax) / (2 * image.shape[1])
        y_center = (ymin + ymax) / (2 * image.shape[0])

        return BoundingBox(
            x=x_center,
            y=y_center,
            width=width,
            height=height
        )
