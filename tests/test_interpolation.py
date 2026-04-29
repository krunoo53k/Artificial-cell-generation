import pytest
import numpy as np
import matplotlib.pyplot as plt
from cellgen.utils.interpolation import SplineInterpolation

def test_spline_interpolation_random_visual():
    """Test spline interpolation with random points."""
    # Generate random points in polar coordinates for a blob-like shape
    n_points = 8
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    radii = np.random.uniform(0.5, 1.0, n_points)  # Random radii between 0.5 and 1.0

    # Convert to Cartesian coordinates
    original_points = np.array([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ]).T

    # Add the first point at the end to ensure closure
    original_points = np.vstack([original_points, original_points[0]])

    # Create interpolator
    interpolator = SplineInterpolation(n_points=200)

    # Interpolate x and y coordinates separately
    x_interp = interpolator.interpolate(original_points[:, 0])
    y_interp = interpolator.interpolate(original_points[:, 1])

    # Plotting
    plt.figure(figsize=(15, 5))

    # Plot original points and interpolated curve
    plt.subplot(131)
    plt.plot(original_points[:, 0], original_points[:, 1], 'ro-', label='Original points')
    plt.plot(x_interp, y_interp, 'b-', label='Interpolated curve', linewidth=2)
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title('Spline Interpolation')

    # Plot points in parametric space
    plt.subplot(132)
    t_original = np.linspace(0, 1, len(original_points))
    t_interp = np.linspace(0, 1, len(x_interp))

    plt.plot(t_original, original_points[:, 0], 'ro-', label='Original x')
    plt.plot(t_original, original_points[:, 1], 'go-', label='Original y')
    plt.plot(t_interp, x_interp, 'b-', label='Interpolated x')
    plt.plot(t_interp, y_interp, 'g-', label='Interpolated y')
    plt.grid(True)
    plt.legend()
    plt.title('Parametric View')

    # Plot curvature visualization
    plt.subplot(133)
    dx = np.gradient(x_interp)
    dy = np.gradient(y_interp)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy)**1.5

    plt.plot(x_interp, y_interp, 'k-', linewidth=1, alpha=0.5)
    plt.scatter(x_interp, y_interp, c=curvature, cmap='viridis', s=20)
    plt.colorbar(label='Curvature')
    plt.axis('equal')
    plt.title('Curvature Analysis')

    plt.tight_layout()
    plt.show()

    # Basic assertions
    assert len(x_interp) == 200
    assert len(y_interp) == 200
    assert np.allclose(x_interp[0], x_interp[-1], atol=1e-10)  # Check if curve is closed
    assert np.allclose(y_interp[0], y_interp[-1], atol=1e-10)

if __name__ == "__main__":
    # Run multiple times to see different random shapes
    for i in range(3):
        print(f"Generating random shape {i+1}")
        test_spline_interpolation_random_visual()
