import pytest
import numpy as np
import matplotlib.pyplot as plt
from cellgen.cells.neutrophil import Neutrophil
from cellgen.cells.base import CellParameters
from cellgen.utils.colormap import CellSolidColors

def test_neutrophil_cytoplasm_coloring():
    """Test that neutrophil cytoplasm has correct color values and appearance."""
    params = CellParameters(
        size=512,
        noise_amount=0.2,
        sigma=1.0,
        intensity=255
    )

    neutrophil = Neutrophil(params)
    cell_image = neutrophil.generate()

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Original RGB
    plt.subplot(131)
    plt.imshow(cell_image[..., :3])
    plt.title("RGB Channels")

    # Alpha channel
    plt.subplot(132)
    plt.imshow(cell_image[..., 3], cmap='gray')
    plt.title("Alpha Channel")

    # Color distribution
    plt.subplot(133)
    # Only look at non-zero alpha regions
    mask = cell_image[..., 3] > 0.1
    colors = cell_image[mask, :3]

    # Plot color distributions
    for i, color in enumerate(['red', 'green', 'blue']):
        plt.hist(colors[:, i], bins=50, alpha=0.5, label=color)
    plt.legend()
    plt.title("Color Distribution")

    plt.tight_layout()
    plt.show()

    # Test color values
    non_zero_mask = cell_image[..., 3] > 0.1
    cytoplasm = cell_image[non_zero_mask]

    # Expected neutrophil cytoplasm color range checks
    colors = CellSolidColors.NEUTROPHIL_CYTOPLASM
    assert np.mean(cytoplasm[:, 0]) > colors[0] * 0.1  # Red component
    assert np.mean(cytoplasm[:, 1]) > colors[1] * 0.1  # Green component
    assert np.mean(cytoplasm[:, 2]) > colors[2] * 0.1  # Blue component

def test_neutrophil_parameter_variations():
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

        # Display RGBA channels
        ax.imshow(cell_image)
        ax.set_title(f'noise={params.noise_amount}\nsigma={params.sigma}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def test_neutrophil_transparency():
    """Test neutrophil transparency and alpha blending."""
    params = CellParameters(size=512)
    neutrophil = Neutrophil(params)
    cell_image = neutrophil.generate()

    # Create figure with checkerboard background
    plt.figure(figsize=(10, 5))

    # Create checkerboard pattern
    checker = np.zeros((512, 512, 3))
    checker[::20, ::20] = 1
    checker[10::20, 10::20] = 1

    # Show with transparency on checkerboard
    plt.subplot(121)
    plt.imshow(checker)
    plt.imshow(cell_image, alpha=cell_image[..., 3])
    plt.title("With Transparency")

    # Show alpha channel
    plt.subplot(122)
    alpha_display = plt.imshow(cell_image, cmap='gray')
    plt.colorbar(alpha_display)
    plt.title("Alpha Channel")

    plt.tight_layout()
    plt.show()

    # Test alpha channel properties
    alpha = cell_image[..., 3]
    assert np.max(alpha) <= 1.0
    assert np.min(alpha) >= 0.0
    assert np.mean(alpha[alpha > 0]) > 0.3  # Average opacity of visible parts

if __name__ == "__main__":
    print("Testing neutrophil cytoplasm coloring...")
    test_neutrophil_cytoplasm_coloring()

    print("Testing parameter variations...")
    test_neutrophil_parameter_variations()

    print("Testing transparency...")
    test_neutrophil_transparency()
