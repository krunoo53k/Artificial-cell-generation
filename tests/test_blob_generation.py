import pytest
import numpy as np
from cellgen.utils.blob import BlobGenerator
from typing import List, Tuple

def test_random_points_generation():
    """Test if random points are generated correctly."""
    n_points = 7
    points = BlobGenerator.generate_random_points(n_points)

    # Check if we get the correct number of points
    assert len(points) == n_points

    # Check if each point is a tuple of two floats
    for point in points:
        assert len(point) == 2
        assert isinstance(point[0], float)
        assert isinstance(point[1], float)

    # Check if points are within expected range (-1 to 1 with scale=0.8)
    for x, y in points:
        assert -0.8 <= x <= 0.8
        assert -0.8 <= y <= 0.8

def test_convex_hull():
    """Test if convex hull is generated correctly."""
    # Create a simple square of points, explicitly as floats
    points: List[Tuple[float, float]] = [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5)
    ]
    hull = BlobGenerator.convex_hull(points)

    # Convex hull should have 4 points (corners of square)
    assert len(hull) == 4

    # Check if middle point is not in hull
    assert (0.5, 0.5) not in hull

def test_bezier_curve():
    """Test if bezier curve generation works."""
    points: List[Tuple[float, float]] = [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0)
    ]
    x_vals, y_vals = BlobGenerator.bezier_curve(points, n_times=100)

    # Add debug prints
    print(f"First x value: {x_vals[0]}, Expected: {points[0][0]}")
    print(f"First y value: {y_vals[0]}, Expected: {points[0][1]}")
    print(f"Last x value: {x_vals[-1]}, Expected: {points[-1][0]}")
    print(f"Last y value: {y_vals[-1]}, Expected: {points[-1][1]}")

    # Check output shapes
    assert len(x_vals) == 100
    assert len(y_vals) == 100

    # Check if curve starts and ends near control points
    assert np.isclose(x_vals[0], points[0][0], atol=0.2)
    assert np.isclose(y_vals[0], points[0][1], atol=0.2)
    assert np.isclose(x_vals[-1], points[-1][0], atol=0.2)
    assert np.isclose(y_vals[-1], points[-1][1], atol=0.2)
