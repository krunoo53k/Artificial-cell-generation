from dataclasses import dataclass
import numpy as np
from enum import Enum
from math import atan2, pi
from typing import List, Tuple, Optional
from .interpolation import InterpolationMethod, InterpolationFactory


@dataclass
class BlobParams:
    """Parameters for blob generation."""
    n_points: int = 7
    scale: float = 0.8
    interpolation_points: int = 30
    min_radius: Optional[float] = None
    max_radius: Optional[float] = None
    interpolation_method: InterpolationMethod = InterpolationMethod.FOURIER

class BlobGenerator:
    """Main blob generator class."""

    @staticmethod
    def convex_hull(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Generate convex hull from points (Graham's scan)."""
        xleftmost, yleftmost = min(pts)
        by_theta = [(atan2(x - xleftmost, y - yleftmost), x, y) for x, y in pts]
        by_theta.sort()
        as_complex = [complex(x, y) for _, x, y in by_theta]
        chull = as_complex[:2]

        for pt in as_complex[2:]:
            while ((pt - chull[-1]).conjugate() *
                   (chull[-1] - chull[-2])).imag < 0:
                chull.pop()
            chull.append(pt)

        return [(pt.real, pt.imag) for pt in chull]

    @staticmethod
    def generate_smooth_blob(
        params: Optional[BlobParams] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate smooth blob using specified parameters.

        Args:
            params: Blob generation parameters. If None, uses defaults.

        Returns:
            Tuple of (x_coordinates, y_coordinates)
        """
        params = params or BlobParams()

        # Generate random points in polar coordinates
        angles = np.linspace(0, 2*pi, params.n_points, endpoint=False)
        min_r = params.min_radius or 0.5*params.scale
        max_r = params.max_radius or params.scale
        radii = np.random.uniform(min_r, max_r, params.n_points)

        # Convert to cartesian coordinates
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)

        # Create points for convex hull
        pts = [(x_, y_) for x_, y_ in zip(x, y)]
        hull_pts = BlobGenerator.convex_hull(pts)
        xs, ys = zip(*hull_pts)

        # Get interpolator from factory
        interpolator = InterpolationFactory.create(
            method=params.interpolation_method,
            n_points=params.interpolation_points
        )

        # Interpolate x and y coordinates
        x_smooth = interpolator.interpolate(np.array(xs))
        y_smooth = interpolator.interpolate(np.array(ys))

        return x_smooth, y_smooth
