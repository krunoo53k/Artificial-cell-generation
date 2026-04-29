from dataclasses import dataclass
import numpy as np
from .erythrocyte import Erythrocyte, ErythrocyteParams
from ..utils.colormap import CellColorMaps
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise

@dataclass
class BackgroundParams:
    """Parameters for background generation."""
    width: int = 512
    height: int = 512
    cell_size: int = 128
    min_cells: int = 10
    max_cells: int = 20
    overlap: float = 0.3
    noise_amount: float = 0.005  # Amount of salt & pepper noise
    blur_sigma: float = 2.0     # Gaussian blur sigma

class Background:
    """Generator for cell background."""

    def __init__(self, params: BackgroundParams = None):
        self.params = params or BackgroundParams()
        self.erythrocyte = Erythrocyte(ErythrocyteParams(size=self.params.cell_size))

    def generate(self) -> np.ndarray:
        """Generate background with multiple red blood cells.

        Returns:
            np.ndarray: RGBA background image
        """
        # Create base RGB image with solid background color
        base_color = np.array([239/255, 211/255, 187/255])

        # Create noisy background
        background = np.full((self.params.height, self.params.width, 3), base_color)

        # Add noise to each channel separately
        for i in range(3):
            channel = background[..., i]
            noisy = random_noise(channel, mode='s&p', amount=self.params.noise_amount)
            background[..., i] = gaussian_filter(noisy, sigma=self.params.blur_sigma)

        # Convert to RGBA
        image = np.zeros((self.params.height, self.params.width, 4))
        image[..., :3] = background
        image[..., 3] = 1.0  # Full alpha for background

        # Generate random number of cells
        n_cells = np.random.randint(self.params.min_cells, self.params.max_cells + 1)

        # Place cells
        for _ in range(n_cells):
            cell_rgba = self.erythrocyte.generate()
            x = np.random.randint(0, self.params.width - self.params.cell_size)
            y = np.random.randint(0, self.params.height - self.params.cell_size)

            # Get cell region in background
            region = image[y:y+self.params.cell_size, x:x+self.params.cell_size]

            # Alpha blend the new cell with existing content
            alpha_new = cell_rgba[..., 3:4]
            alpha_existing = region[..., 3:4]

            # Composite alpha
            alpha_out = alpha_new + alpha_existing * (1 - alpha_new)

            # Blend colors
            for i in range(3):
                region[..., i] = (cell_rgba[..., i] * alpha_new[..., 0] +
                                region[..., i] * alpha_existing[..., 0] * (1 - alpha_new[..., 0])) / alpha_out[..., 0]

            # Update alpha
            region[..., 3] = alpha_out[..., 0]

        return image
