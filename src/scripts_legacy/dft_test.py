from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.fft import ifftn
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

points = rng.random((6, 2))   # 30 random points in 2-D
print(points)
plt.plot(points)
plt.show()