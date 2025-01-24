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
        intensity=255,
        distance_weight=0.3
    )

    erythrocyte = Erythrocyte(params)
    cell = erythrocyte.generate()

    # Shape and value tests
    assert cell.shape == (128, 128)
    assert np.max(cell) <= 255
    assert np.min(cell) >= 0
    assert not np.all(cell == 0)

    # Visualization
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cell, cmap='gray')
    plt.title('Single Erythrocyte')
    plt.colorbar()

    # Show distance transform effect
    plt.subplot(1, 2, 2)
    plt.hist(cell[cell > 0].flatten(), bins=50)
    plt.title('Intensity Distribution')

    plt.tight_layout()
    plt.show()
    plt.close()

def test_erythrocyte_parameters():
    """Test erythrocyte generation with different parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    params_list = [
        ErythrocyteParams(distance_weight=0.1, points=5),
        ErythrocyteParams(distance_weight=0.5, points=5),
        ErythrocyteParams(distance_weight=0.3, points=7),
        ErythrocyteParams(distance_weight=0.3, points=9)
    ]

    for ax, params in zip(axes.flat, params_list):
        erythrocyte = Erythrocyte(params)
        cell = erythrocyte.generate()

        im = ax.imshow(cell, cmap='gray')
        ax.set_title(f'weight={params.distance_weight}\npoints={params.points}')
        plt.colorbar(im, ax=ax)

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
    image = background.generate()

    # Basic assertions
    assert image.shape == (512, 512, 3)
    assert not np.all(image == 0)
    assert np.max(image) <= 1.0
    assert np.min(image) >= 0.0

    # Visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Full Background')

    plt.subplot(1, 2, 2)
    for channel, color in zip(range(3), ['red', 'green', 'blue']):
        plt.hist(image[:,:,channel].flatten(), bins=50,
                alpha=0.5, color=color, label=color)
    plt.title('Color Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()

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
