import pytest
import numpy as np
import matplotlib.pyplot as plt
from cellgen.cells.neutrophil import Neutrophil
from cellgen.cells.base import CellParameters
from cellgen.utils.colormap import CellSolidColors

def test_neutrophil_complete_generation():
    """Test complete neutrophil generation including nucleus and cell body."""
    params = CellParameters(
        size=512,
        noise_amount=0.2,
        sigma=1.0
    )

    neutrophil = Neutrophil(params)
    cell_image = neutrophil.generate()

    # Create visualization with multiple views
    plt.figure(figsize=(20, 5))

    # 1. Complete cell
    plt.subplot(141)
    plt.imshow(cell_image[..., :3])  # RGB channels
    plt.title("Complete Neutrophil")
    plt.axis('off')

    # 2. Alpha channel
    plt.subplot(142)
    plt.imshow(cell_image[..., 3], cmap='gray')
    plt.title("Alpha Channel")
    plt.axis('off')

    # 3. On checkerboard background to show transparency
    plt.subplot(143)
    checker = np.zeros((512, 512, 3))
    checker[::20, ::20] = 1
    checker[10::20, 10::20] = 1
    plt.imshow(checker)
    plt.imshow(cell_image[..., :3], alpha=cell_image[..., 3])
    plt.title("Transparency Check")
    plt.axis('off')

    # 4. Color distribution
    plt.subplot(144)
    mask = cell_image[..., 3] > 0.1
    colors = cell_image[mask, :3]
    for i, color in enumerate(['red', 'green', 'blue']):
        plt.hist(colors[:, i], bins=50, alpha=0.5, label=color)
    plt.legend()
    plt.title("Color Distribution")

    plt.tight_layout()
    plt.show()

    # Basic assertions
    assert cell_image.shape == (512, 512, 4)  # RGBA image
    assert np.max(cell_image) <= 1.0  # Normalized values
    assert np.min(cell_image) >= 0.0

    # Check for presence of both nucleus and cytoplasm
    # Nucleus typically has more blue component
    blue_regions = cell_image[..., 2] > 0.5
    assert np.any(blue_regions)  # Should have some nucleus regions

    # Cytoplasm color check
    cytoplasm_color = CellSolidColors.NEUTROPHIL_CYTOPLASM
    cytoplasm_regions = np.logical_and(
        cell_image[..., 3] > 0.1,  # Non-transparent
        cell_image[..., 2] < 0.5   # Not nucleus (less blue)
    )
    assert np.any(cytoplasm_regions)  # Should have some cytoplasm regions

def test_neutrophil_variations():
    """Test neutrophil generation with different parameters."""
    params_list = [
        CellParameters(size=512, noise_amount=0.1, sigma=0.5),
        CellParameters(size=512, noise_amount=0.3, sigma=1.0),
        CellParameters(size=512, noise_amount=0.5, sigma=1.5)
    ]

    fig, axes = plt.subplots(1, len(params_list), figsize=(15, 5))

    for ax, params in zip(axes, params_list):
        neutrophil = Neutrophil(params)
        cell_image = neutrophil.generate()

        ax.imshow(cell_image)
        ax.set_title(f'noise={params.noise_amount}\nsigma={params.sigma}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Testing complete neutrophil generation...")
    test_neutrophil_complete_generation()

    print("Testing parameter variations...")
    test_neutrophil_variations()
