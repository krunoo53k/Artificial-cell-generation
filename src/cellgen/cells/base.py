from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class CellParameters:
    """Parameters for cell generation."""
    size: int = 512
    noise_amount: float = 0.2
    sigma: float = 1.0
    intensity: float = 255

@dataclass
class BoundingBox:
    """Bounding box in YOLO format."""
    x: float  # center x normalized
    y: float  # center y normalized
    width: float  # normalized
    height: float  # normalized

class Cell(ABC):
    """Abstract base class for all cell types."""

    def __init__(self, params: CellParameters):
        self.params = params

    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generate cell image.

        Returns:
            np.ndarray: Generated cell image
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get cell type name.

        Returns:
            str: Cell type name
        """
        pass
