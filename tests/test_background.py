import pytest
import numpy as np
from cellgen.background.erythrocyte import Erythrocyte, ErythrocyteParams
from cellgen.background.background import Background, BackgroundParams
import matplotlib.pyplot as plt

def test_erythrocyte_generation():
    """Test single erythrocyte generation with visual output."""
    params = ErythrocyteParams(
        size=128,
        points=7,
        scale=0.8,
        center_opacity=0.6
    )

    erythrocyte = Erythrocyte(params)
    cell = erythrocyte.generate()

    # Create figure with checkerboard background to show transparency
    fig = plt.figure(figsize=(15, 5))

    # 1. Show on checkerboard background to visualize transparency
    ax1 = plt.subplot(1, 3, 1)
    # Create checkerboard pattern
    checker = np.zeros((128, 128))
    checker[::20, ::20] = 1
    checker[10::20, 10::20] = 1

    # Show checkerboard
    ax1.imshow(checker, cmap='gray', extent=[0, 128, 0, 128])
    # Overlay cell with transparency
    ax1.imshow(cell[..., :3], alpha=cell[..., 3])
    ax1.set_title('With Transparency\n(on checkerboard)')

    # 2. Show alpha channel separately
    ax2 = plt.subplot(1, 3, 2)
    alpha_display = ax2.imshow(cell[..., 3], cmap='gray')
    plt.colorbar(alpha_display, ax=ax2)
    ax2.set_title('Alpha Channel')

    # 3. Show cross-section
    ax3 = plt.subplot(1, 3, 3)
    center_line = cell[cell.shape[0]//2, :, 3]
    ax3.plot(center_line, 'b-', label='Alpha value')
    ax3.set_title('Alpha Channel Cross-section')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Opacity')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close()

    # Print some debug info
    print(f"Alpha channel range: {np.min(cell[..., 3]):.3f} to {np.max(cell[..., 3]):.3f}")
    print(f"Shape: {cell.shape}")
    print(f"Number of non-zero alpha values: {np.sum(cell[..., 3] > 0)}")

def test_erythrocyte_parameters():
    """Test erythrocyte generation with different parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Create checkerboard pattern for background
    checker = np.zeros((128, 128))
    checker[::20, ::20] = 1
    checker[10::20, 10::20] = 1

    params_list = [
        ErythrocyteParams(center_opacity=0.3, points=5),
        ErythrocyteParams(center_opacity=0.6, points=5),
        ErythrocyteParams(center_opacity=0.8, points=7),
        ErythrocyteParams(center_opacity=0.9, points=9)
    ]

    for ax, params in zip(axes.flat, params_list):
        erythrocyte = Erythrocyte(params)
        cell = erythrocyte.generate()

        # Show checkerboard
        ax.imshow(checker, cmap='gray', extent=[0, 128, 0, 128])
        # Overlay cell with transparency
        ax.imshow(cell[..., :3], alpha=cell[..., 3])
        ax.set_title(f'opacity={params.center_opacity}\npoints={params.points}')

    plt.tight_layout()
    plt.show()
    plt.close()

def test_background_generation():
    """Test full background generation."""
    params = BackgroundParams(
        width=512,
        height=512,
        cell_size=128,
        min_cells=10,
        max_cells=20,
        overlap=0.3
    )

    background = Background(params)
    image = background.generate()  # Now returns RGBA

    # Visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    # Use alpha channel when plotting
    plt.imshow(image[..., :3], alpha=image[..., 3])
    plt.title('Background with Cells')

    plt.subplot(1, 2, 2)
    plt.imshow(image[..., 3], cmap='gray')
    plt.title('Alpha Channel')

    plt.tight_layout()
    plt.show()

def test_background_parameters():
    """Test background generation with different parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    params_list = [
        BackgroundParams(min_cells=5, max_cells=10),
        BackgroundParams(min_cells=15, max_cells=25),
        BackgroundParams(cell_size=100, overlap=0.2),
        BackgroundParams(cell_size=150, overlap=0.4)
    ]

    for ax, params in zip(axes.flat, params_list):
        background = Background(params)
        image = background.generate()

        ax.imshow(image)
        ax.set_title(f'cells={params.min_cells}-{params.max_cells}\n'
                    f'size={params.cell_size}, overlap={params.overlap}')

    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("Testing single erythrocyte generation...")
    test_erythrocyte_generation()

    print("Testing erythrocyte parameters...")
    test_erythrocyte_parameters()

    print("Testing full background generation...")
    test_background_generation()

    print("Testing background parameters...")
    test_background_parameters()
