import pytest
import numpy as np
import matplotlib.pyplot as plt
from cellgen.nucleus.neutrophil import NeutrophilNucleus, NeutrophilNucleusParams

def test_neutrophil_nucleus_coloring():
    """Test the coloring of neutrophil nucleus with explicit transparency check."""
    params = NeutrophilNucleusParams(size=256, base_thickness=0.5)
    nucleus = NeutrophilNucleus(params)
    base_mask = nucleus.generate()
    colored_nucleus = nucleus.color_nucleus(base_mask)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Original mask
    axes[0, 0].imshow(base_mask, cmap='gray')
    axes[0, 0].set_title('Original Mask')

    # 2. Colored result on white background
    axes[0, 1].set_facecolor('white')
    rgba_display = colored_nucleus.copy()
    axes[0, 1].imshow(rgba_display)  # Use the full RGBA data
    axes[0, 1].set_title('Colored Nucleus (RGBA)')

    # 3. On checkerboard background
    checker = np.zeros((256, 256, 3))
    checker[::20, ::20] = 1
    checker[10::20, 10::20] = 1
    axes[1, 0].imshow(checker)
    axes[1, 0].imshow(rgba_display)
    axes[1, 0].set_title('On Checkerboard')

    # 4. Alpha channel distribution
    valid_alphas = colored_nucleus[..., 3][base_mask > 0]
    axes[1, 1].hist(valid_alphas.flatten(), bins=50)
    axes[1, 1].set_title('Alpha Value Distribution')

    plt.tight_layout()
    plt.show()

def test_nucleus_coloring_parameters():
    """Test nucleus coloring with different noise parameters."""
    params = NeutrophilNucleusParams(size=256, base_thickness=0.5)
    nucleus = NeutrophilNucleus(params)
    base_mask = nucleus.generate()

    # Test different noise amounts
    noise_amounts = [0.1, 0.3, 0.5]
    sigma_values = [0.3, 0.5, 0.7]

    fig, axes = plt.subplots(len(noise_amounts), len(sigma_values),
                            figsize=(15, 15))

    for i, noise in enumerate(noise_amounts):
        for j, sigma in enumerate(sigma_values):
            colored = nucleus.color_nucleus(base_mask,
                                         noise_amount=noise,
                                         sigma=sigma)

            axes[i, j].imshow(colored)
            axes[i, j].set_title(f'Noise: {noise}\nSigma: {sigma}')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def test_color_consistency():
    """Test that nucleus coloring is consistent with biological appearance."""
    params = NeutrophilNucleusParams(size=256, base_thickness=0.5)
    nucleus = NeutrophilNucleus(params)

    # Generate multiple nuclei
    n_samples = 3
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 5))

    for i in range(n_samples):
        base_mask = nucleus.generate()
        colored = nucleus.color_nucleus(base_mask)

        axes[i].imshow(colored)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')

        # Check color ranges
        assert np.mean(colored[..., 0]) < np.mean(colored[..., 2])  # More blue than red
        assert np.mean(colored[..., 1]) < np.mean(colored[..., 2])  # More blue than green

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Testing nucleus coloring...")
    test_neutrophil_nucleus_coloring()

    print("Testing coloring parameters...")
    test_nucleus_coloring_parameters()

    print("Testing color consistency...")
    test_color_consistency()
