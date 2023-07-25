import cmath
from math import atan2
from random import random
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter, distance_transform_edt
from math import pi

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

def generate_blob_image(size=100):
    pts = [(random() + 0.8) * cmath.exp(2j*cmath.pi*i/7) for i in range(7)]
    pts = convexHull([(pt.real, pt.imag) for pt in pts])
    xs, ys = [interpolateSmoothly(zs, 30) for zs in zip(*pts)]

    # prazna slika size x size
    img = np.zeros((size, size))

    # prebacivanje u koordinate
    points = np.array(list(zip(xs, ys)))

    # normaliziranje da bude izmedju 0 i 1
    points += 2
    points /= 2 + 2

    # skaliranje na velicinu slike
    points *= size

    # openCV format
    points = points.astype(np.int32)
    points = points.reshape((-1, 1, 2))

    cv.fillPoly(img, [points], color=255)
    
    return transform_blob(img)

def transform_blob(img):
    img2 = distance_transform_edt(img)
    img = img - img2 * randint(2, 4)
    img = gaussian_filter(img, sigma=2)
    return img

def generate_background_image():
    # prazna slika size x size
    max_blob_size = 140
    min_blob_size = 60
    img = np.zeros((360 + max_blob_size, 363 + min_blob_size))
    num_of_blobs = randint(6, 20)
    for i in range(0, num_of_blobs):
        size = randint(min_blob_size, max_blob_size)
        centre_x = randint(0, 360 + max_blob_size - size)
        centre_y = randint(0, 363 + min_blob_size - size)
        blob = generate_blob_image(size)

        # Binary mask to ignore zeros in the blob image
        mask = (blob > 60)
        img[centre_x:centre_x+size, centre_y:centre_y+size][mask] = blob[mask]

    return img

background_image = generate_background_image()
#background_image = distance_transform_edt(background_image)
background_image = gaussian_filter(background_image, sigma=1)
plt.imshow(background_image, cmap="Greys")
plt.show()