from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.fft import ifftn
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

points = rng.random((6, 2))   # 30 random points in 2-D

hull = ConvexHull(points)

for simplex in hull.simplices:
    print(points[simplex,0])
    #print(points)
    xs = ifftn(points[simplex,0])
    ys = ifftn(points[simplex,1])
    plt.plot(xs, ys,'k-')
plt.show()