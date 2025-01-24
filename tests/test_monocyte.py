from cellgen.cells.monocyte import Monocyte
from cellgen.cells.base import CellParameters
import matplotlib.pyplot as plt

def test_monocyte_generation():
    params = CellParameters(
        size=512,
        noise_amount=0.2,
        sigma=1.0,
        intensity=255
    )

    monocyte = Monocyte(params)

    # Generate cell
    cell_image = monocyte.generate()

    # Optionally get bounding box if needed
    bbox = monocyte.get_bounding_box(cell_image)

    # Display result
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cell_image[..., :3])  # RGB channels
    plt.title("RGB channels")

    plt.subplot(1, 2, 2)
    plt.imshow(cell_image[..., 3], cmap='gray')  # Alpha channel
    plt.title("Alpha channel")

    plt.suptitle(f"Monocyte (bbox: {bbox})")
    plt.show()

if __name__ == "__main__":
    test_monocyte_generation()
