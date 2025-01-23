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
        """Generate bezier curve from control points.

        Args:
            points: List of (x,y) control points
            n_times: Number of points to generate along curve

        Returns:
            Tuple of (x_values, y_values) arrays defining the curve
        """
        def single_bezier_point(t: float) -> Tuple[float, float]:
            n = len(points) - 1
            x = 0.0
            y = 0.0
            for i, point in enumerate(points):
                coef = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
                x += coef * point[0]
                y += coef * point[1]
            return x, y

        t_points = np.linspace(0.0, 1.0, n_times)
        curve_points = [single_bezier_point(t) for t in t_points]
        x_vals = np.array([p[0] for p in curve_points])
        y_vals = np.array([p[1] for p in curve_points])

        return x_vals, y_vals

    @staticmethod
    def bernstein_poly(i: int, n: int, t: np.ndarray) -> np.ndarray:
        """Bernstein polynomial."""
        return comb(n, i) * (t**(n-i)) * (1-t)**i
