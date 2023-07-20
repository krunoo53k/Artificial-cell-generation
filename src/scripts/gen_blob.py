import cmath
from math import atan2
from random import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter, distance_transform_edt

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

def noisy(noise_typ,image):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

pts = [(random() + 0.8) * cmath.exp(2j*cmath.pi*i/7) for i in range(7)]
pts = convexHull([(pt.real, pt.imag ) for pt in pts])
xs, ys = [interpolateSmoothly(zs, 30) for zs in zip(*pts)]



# prazna slika 100 x 100
size = 100
img = np.zeros((size, size,3))

 

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

 

cv.fillPoly(img, [points], color=(204/255, 160/255, 39/255))

plt.imshow(img)
plt.show()