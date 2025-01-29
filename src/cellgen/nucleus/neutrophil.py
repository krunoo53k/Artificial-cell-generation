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
    connection_thickness: float = 0.15  # Thickness of connections between segments
    compactness: float = 0.3  # Controls how close segments are to each other

class NeutrophilNucleus:
    def __init__(self, params: NeutrophilNucleusParams = None):
        self.params = params or NeutrophilNucleusParams()

    def generate(self) -> np.ndarray:
        # Generate base curved path
        curve_points = self._generate_base_curve()

        # Create empty image
        image = np.zeros((self.params.size, self.params.size))

        # Find segment centers
        segment_centers = np.linspace(0, 1, self.params.num_segments)
        segment_width = 1.0 / (self.params.num_segments * 2)  # Half the space between centers

        # Generate points along curve with varying thickness
        for t in np.linspace(0, 1, 200):  # Higher number = smoother result
            # Get point along curve
            point = self._get_curve_point(curve_points, t)

            # Calculate distance to nearest segment center
            distances = np.abs(t - segment_centers)
            min_distance = np.min(distances)

            # Determine if we're in a segment region or connection region
            if min_distance < segment_width:
                # We're in a segment region - make it blob-like
                thickness = self._calculate_segment_thickness(min_distance, segment_width)
            else:
                # We're in a connection region - make it thin
                thickness = self.params.connection_thickness

            # Calculate radius in pixels
            radius = int(thickness * self.params.size * 0.1)

            # Ensure point coordinates are integers and within bounds
            x = int(np.clip(point[0], radius, self.params.size - radius))
            y = int(np.clip(point[1], radius, self.params.size - radius))

            # Draw circle
            cv2.circle(image, (x, y), max(1, radius), 1, -1)

        return image

    def _generate_base_curve(self) -> np.ndarray:
        """Generate random curved path for nucleus."""
        # Generate control points
        num_points = self.params.num_segments + 1

        # Start and end points
        points = np.zeros((num_points, 2))

        # Calculate center and scale
        center = self.params.size / 2
        scale = self.params.size * 0.8

        # Start and end points closer together
        points[0] = [center - scale * self.params.compactness, center]
        points[-1] = [center + scale * self.params.compactness, center]

        # Generate middle points with controlled randomness
        for i in range(1, num_points-1):
            t = i / (num_points - 1)
            points[i] = [
                center - scale * self.params.compactness + 2 * scale * self.params.compactness * t,
                center + scale * self.params.curve_randomness * np.random.uniform(-1, 1)
            ]

        return points

    def _get_curve_point(self, control_points: np.ndarray, t: float) -> np.ndarray:
        """Get point along spline at parameter t."""
        cs = CubicSpline(
            np.linspace(0, 1, len(control_points)),
            control_points,
            bc_type='natural'
        )
        return cs(t)

    def _calculate_segment_thickness(self, distance: float, segment_width: float) -> float:
        """Calculate thickness for segment regions."""
        # Normalize distance to segment width
        normalized_dist = distance / segment_width

        # Create smooth falloff from center of segment
        thickness = self.params.base_thickness * (1 - normalized_dist**2)

        return max(self.params.connection_thickness, thickness)
