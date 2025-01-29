import pytest
import numpy as np
import matplotlib.pyplot as plt
from cellgen.nucleus.neutrophil import NeutrophilNucleus, NeutrophilNucleusParams

def test_neutrophil_nucleus_generation():
    """Test neutrophil nucleus generation with visual output."""
    # Test different parameter configurations
    param_sets = [
        NeutrophilNucleusParams(
            size=512,
            num_segments=3,
            segment_frequency=2.0,
            segment_amplitude=0.3,
            base_thickness=0.3
        ),
        NeutrophilNucleusParams(
            size=512,
            num_segments=2,
            segment_frequency=1.5,
            segment_amplitude=0.4,
            base_thickness=0.35
        ),
        NeutrophilNucleusParams(
            size=512,
            num_segments=4,
            segment_frequency=3.0,
            segment_amplitude=0.25,
            base_thickness=0.25
        )
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, params in zip(axes, param_sets):
        nucleus = NeutrophilNucleus(params)
        image = nucleus.generate()

        ax.imshow(image, cmap='Blues')
        ax.set_title(f'Segments: {params.num_segments}\n'
                    f'Freq: {params.segment_frequency}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_neutrophil_nucleus_generation()
