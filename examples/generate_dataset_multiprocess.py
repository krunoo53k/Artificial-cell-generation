import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from cellgen.composition.placement import CellPlacement
from cellgen.background import BackgroundParams
from cellgen.cells.base import CellParameters
import multiprocessing
from functools import partial

def generate_single_image(i, output_path, cell_types):
    """Generate a single image with its annotation"""
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"

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

    print(f"Generated image {i+1}")

def generate_dataset(
    output_dir: str,
    num_images: int = 100,
    cell_types: List[str] = ["neutrophil", "monocyte"]
):
    """Generate a dataset of cell images with annotations using multiprocessing."""
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create a partial function with fixed arguments
    generate_func = partial(generate_single_image,
                          output_path=output_path,
                          cell_types=cell_types)

    # Use multiprocessing pool
    with multiprocessing.Pool() as pool:
        pool.map(generate_func, range(num_images))

if __name__ == "__main__":
    # Generate a test dataset
    generate_dataset(
        output_dir="output/dataset_v1",
        num_images=100,
        cell_types=["neutrophil"]
    )
