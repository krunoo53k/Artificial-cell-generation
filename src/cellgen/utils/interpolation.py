from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from enum import Enum
from typing import List
import cmath
from math import pi

class InterpolationMethod(Enum):
    """Supported interpolation methods."""
    SPLINE = "spline"
    FOURIER = "fourier"

class InterpolationStrategy(ABC):
    """Abstract base class for interpolation strategies."""

    @abstractmethod
    def interpolate(self, points: np.ndarray) -> np.ndarray:
        """Interpolate points to create smooth curve."""
        pass

class FourierInterpolation(InterpolationStrategy):
    """Fourier transform-based interpolation."""

    def __init__(self, n_points: int = 100):
        self.n_points = n_points

    def _dft(self, xs: List[complex]) -> List[complex]:
        """Discrete Fourier Transform."""
        return [sum(x * cmath.exp(2j * pi * i * k / len(xs))
                for i, x in enumerate(xs))
                for k in range(len(xs))]

    def interpolate(self, points: np.ndarray) -> np.ndarray:
        """Interpolate using Fourier transform."""
        # Calculate DFT
        fs = self._dft(points)

        # Get the midpoint for frequency splitting
        half = (len(points) + 1) // 2

        # Create expanded frequency domain
        fs2 = fs[:half] + [0] * (len(fs) * self.n_points) + fs[half:]

        # Calculate inverse DFT and take real part
        return np.array([x.real / len(points) for x in self._dft(fs2)[::-1]])

class SplineInterpolation(InterpolationStrategy):
    """Cubic spline-based interpolation."""

    def __init__(self, n_points: int = 100):
        self.n_points = n_points

    def interpolate(self, points: np.ndarray) -> np.ndarray:
        """Interpolate using cubic spline."""
        t = np.linspace(0, 1, len(points))
        t_new = np.linspace(0, 1, self.n_points)
        spline = CubicSpline(t, points, bc_type='periodic')
        return spline(t_new)

class InterpolationFactory:
    """Factory for creating interpolation strategies."""

    @staticmethod
    def create(method: InterpolationMethod, n_points: int = 100) -> InterpolationStrategy:
        """Create appropriate interpolation strategy.

        Args:
            method: Interpolation method to use
            n_points: Number of points for interpolation

        Returns:
            InterpolationStrategy instance
        """
        if method == InterpolationMethod.FOURIER:
            return FourierInterpolation(n_points)
        elif method == InterpolationMethod.SPLINE:
            return SplineInterpolation(n_points)
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")
