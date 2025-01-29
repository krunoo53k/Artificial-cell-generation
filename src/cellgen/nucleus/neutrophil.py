from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
import cv2

@dataclass
class NeutrophilNucleusParams:
    """Parameters for neutrophil nucleus generation."""
    size: int = 128  # Output image size
    num_segments: int = 3  # Usually 3, can be 2-4
    segment_frequency: float = 3.0  # Controls how many "beads" form
    segment_amplitude: float = 0.4  # Controls segment width variation
    base_thickness: float = 0.2  # Base thickness of the nucleus
    curve_randomness: float = 0.3  # How random the base curve is

class NeutrophilNucleus:
    def __init__(self, params: NeutrophilNucleusParams = None):
        self.params = params or NeutrophilNucleusParams()

    def generate(self) -> np.ndarray:
        # Generate base curved path
        curve_points = self._generate_base_curve()

        # Create empty image
        image = np.zeros((self.params.size, self.params.size))

        # Generate points along curve with varying thickness
        for t in np.linspace(0, 1, 200):  # Higher number = smoother result
            # Get point along curve
            point = self._get_curve_point(curve_points, t)

            # Calculate thickness at this point using sinusoidal variation
            thickness = self._calculate_thickness(t)

            # Ensure point is within image bounds
            x = np.clip(int(point[0]), 0, self.params.size-1)
            y = np.clip(int(point[1]), 0, self.params.size-1)

            # Ensure radius is positive and reasonable
            radius = max(1, int(thickness * self.params.size * 0.1))  # Reduced scaling factor

            # Draw circle at point with calculated thickness
            cv2.circle(
                image,
                (x, y),
                radius,
                1,
                -1
            )

        return image

    def _generate_base_curve(self) -> np.ndarray:
        """Generate random curved path for nucleus."""
        # Generate control points
        num_points = self.params.num_segments + 1

        # Start and end points
        points = np.zeros((num_points, 2))
        points[0] = [0.3, 0.5]  # Start left-ish
        points[-1] = [0.7, 0.5]  # End right-ish

        # Generate middle points with some randomness
        for i in range(1, num_points-1):
            points[i] = [
                0.3 + 0.4 * (i/(num_points-1)),  # More controlled x progression
                0.5 + np.random.uniform(
                    -self.params.curve_randomness,
                    self.params.curve_randomness
                )
            ]

        # Scale to image size
        points *= self.params.size

        return points

    def _get_curve_point(self, control_points: np.ndarray, t: float) -> np.ndarray:
        """Get point along spline at parameter t."""
        # Create spline
        cs = CubicSpline(
            np.linspace(0, 1, len(control_points)),
            control_points,
            bc_type='natural'
        )
        return cs(t)

    def _calculate_thickness(self, t: float) -> float:
        """Calculate thickness at point t along curve."""
        # Base thickness plus sinusoidal variation
        thickness = self.params.base_thickness + self.params.segment_amplitude * \
                   np.sin(2 * np.pi * self.params.segment_frequency * t)

        # Taper ends slightly
        edge_taper = 4 * t * (1-t)  # Parabolic tapering

        # Ensure positive thickness
        return max(0.1, thickness * edge_taper)
