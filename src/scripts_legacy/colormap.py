import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

colors = [ (0.62,0.54,0.51), (0.2,0.07,0.52)] # first color is black, last is red
cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)
mat = np.indices((10,10))[1]
plt.imshow(mat, cmap=cm)
plt.show()