import numpy as np
import cmath
from math import atan2, pi
from typing import List, Tuple
from scipy.special import comb

class BlobGenerator:
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
    def generate_random_points(n_points: int = 7,
                            scale: float = 0.8) -> List[Tuple[float, float]]:
       """Generate random points for blob generation.

       Args:
           n_points: Number of points to generate
           scale: Scale factor for the points

       Returns:
           List of tuples containing (x, y) coordinates
       """
       points = []
       for i in range(n_points):
           complex_point = scale * cmath.exp(2j * pi * i / n_points)
           points.append((complex_point.real, complex_point.imag))
       return points

    @staticmethod
    def bezier_curve(points: List[Tuple[float, float]],
                    n_times: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate bezier curve from control points."""
        n_points = len(points)
        x_points = np.array([p[0] for p in points])
        y_points = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, n_times)
        polynomial_array = np.array(
            [BlobGenerator.bernstein_poly(i, n_points-1, t)
             for i in range(0, n_points)])

        x_vals = np.dot(x_points, polynomial_array)
        y_vals = np.dot(y_points, polynomial_array)

        return x_vals, y_vals

    @staticmethod
    def bernstein_poly(i: int, n: int, t: np.ndarray) -> np.ndarray:
        """Bernstein polynomial."""
        return comb(n, i) * (t**(n-i)) * (1-t)**i
