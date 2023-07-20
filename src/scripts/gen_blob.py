import cmath
from math import atan2
from random import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def convexHull(pts):    #Graham's scan.
    xleftmost, yleftmost = min(pts)
    by_theta = [(atan2(x-xleftmost, y-yleftmost), x, y) for x, y in pts]
    by_theta.sort()
    as_complex = [complex(x, y) for _, x, y in by_theta]
    chull = as_complex[:2]
    for pt in as_complex[2:]:
        #Perp product.
        while ((pt - chull[-1]).conjugate() * (chull[-1] - chull[-2])).imag < 0:
            chull.pop()
        chull.append(pt)
    return [(pt.real, pt.imag) for pt in chull]


def dft(xs):
    pi = 3.14
    return [sum(x * cmath.exp(2j*pi*i*k/len(xs)) 
                for i, x in enumerate(xs))
            for k in range(len(xs))]

def interpolateSmoothly(xs, N):
    """For each point, add N points."""
    fs = dft(xs)
    half = (len(xs) + 1) // 2
    fs2 = fs[:half] + [0]*(len(fs)*N) + fs[half:]
    return [x.real / len(xs) for x in dft(fs2)[::-1]]

pts = [(random() + 0.8) * cmath.exp(2j*cmath.pi*i/7) for i in range(7)]
pts = convexHull([(pt.real, pt.imag ) for pt in pts])
xs, ys = [interpolateSmoothly(zs, 30) for zs in zip(*pts)]



# prazna slika 100 x 100
size = 100
img = np.zeros((size, size))

 

# prebacivanje u koordinate
points = np.array(list(zip(xs, ys)))

# normaliziranje da bude izmedju 0 i 1
points += 2
print(points.min())
points /= 2 + 2

# skaliranje na velicinu slike
points *= size

# openCV format
points = points.astype(np.int32)
points = points.reshape((-1, 1, 2))

 

cv.fillPoly(img, [points], color=255)

plt.imshow(img)
plt.show()
