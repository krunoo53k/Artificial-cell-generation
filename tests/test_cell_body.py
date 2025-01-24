import pytest
import numpy as np
from cellgen.cells.cell_body import CellBody, CellBodyParams
import matplotlib.pyplot as plt

def test_cell_body_generation():
    """Test basic cell body generation."""
    params = CellBodyParams(size=256)  # smaller size for testing
    cell_body = CellBody(params)
    mask = cell_body.generate()

    # Basic shape tests
    assert mask.shape == (256, 256)
    assert mask.dtype == np.float64 or mask.dtype == np.float32
    assert np.max(mask) == 1
    assert np.min(mask) == 0

    # Visualization for debug
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    plt.title('Generated Cell Body')
    plt.colorbar()
    plt.show()
    plt.close()

def test_cell_body_params():
    """Test different parameter configurations."""
    params = CellBodyParams(
        size=256,
        points=10,  # more points
        scale=0.6   # smaller cell
    )
    cell_body = CellBody(params)
    mask = cell_body.generate()

    # Visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    plt.title('Cell Body with Modified Parameters')
    plt.colorbar()
    plt.show()
    plt.close()
