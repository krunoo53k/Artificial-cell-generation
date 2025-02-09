import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from cellgen.composition.placement import CellPlacement
from cellgen.background import BackgroundParams
from cellgen.cells.base import CellParameters

def generate_dataset(
    output_dir: str,
    num_images: int = 100,
    cell_types: List[str] = ["neutrophil", "monocyte"]
):
    """Generate a dataset of cell images with annotations."""
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Generate images
    for i in range(num_images):
        # Randomly select cell type
        cell_type = np.random.choice(cell_types)

        # Generate image and bbox
        image, bbox = CellPlacement.generate_cell_on_background(cell_type=cell_type)

        # Save image
        plt.imsave(
            images_dir / f"image_{i:04d}.png",
            image
        )

        # Save annotation
        with open(labels_dir / f"image_{i:04d}.txt", "w") as f:
            class_id = 0 if cell_type == "neutrophil" else 1
            f.write(f"{class_id} {' '.join(map(str, bbox))}")

        print(f"Generated image {i+1}/{num_images}")

if __name__ == "__main__":
    # Generate a test dataset
    generate_dataset(
        output_dir="output/dataset_v1",
        num_images=10,
        cell_types=["neutrophil", "monocyte"]
    )
