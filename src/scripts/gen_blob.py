import cmath
from math import atan2
from random import random
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.special import comb
from scipy.interpolate import interp1d
from skimage.transform import rescale
from matplotlib.colors import LinearSegmentedColormap
from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)
from math import pi

colors = [(0.62,0.54,0.51), (0.2,0.07,0.52)] # first color is black, last is red
cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)

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

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

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
    #plt.imshow(img, cmap="Greys")
    #plt.show()
    return img


def transform_blob(img):
    img2 = distance_transform_edt(img)
    img = img - img2 * randint(2, 4)
    img = gaussian_filter(img, sigma=2)
    return img

def generate_background_image():
    # prazna slika size x size
    max_blob_size = 140
    min_blob_size = 60
    img = np.zeros((360 + max_blob_size, 363 + max_blob_size))
    num_of_blobs = randint(10, 20)
    for i in range(0, num_of_blobs):
        size = randint(min_blob_size, max_blob_size)
        centre_x = randint(0, 360 + max_blob_size - size)
        centre_y = randint(0, 363 + min_blob_size - size)
        blob = generate_blob_image(size)
        blob = transform_blob(blob)

        # Binary mask to ignore zeros in the blob image
        mask = (blob > 30)
        img[centre_x:centre_x+size, centre_y:centre_y+size][mask] = blob[mask]

    #return img
    return img[max_blob_size::,max_blob_size::]

def add_colour(background_image, color):
    # Expand the grayscale image to have 3 channels
    color_image = np.stack((background_image,) * 3, axis=-1)

    # Multiply each pixel value of the color image with the color value
    result_image = color_image * color

    # Clip values to [0, 255]
    result_image = np.clip(result_image, 0, 255)

    return result_image

def add_background(image, color):
    mask = np.all(image >= [209, 209, 209], axis=-1)
    image[mask] = color
    return image


def invert_colors(image):
    # Invert colors by subtracting each pixel value from 255
    inverted_image = np.full(image.shape,255)
    inverted_image = np.subtract(inverted_image, image)

    return inverted_image

def generate_cell(size = 200):
    img = generate_blob_image(size)
    plt.imshow(img, cmap="Greys")
    plt.show()

def generate_nucleus(size = 256):
    img = generate_blob_image(size)
    blob = img.copy()
    np.random.seed(0)
    noise = generate_fractal_noise_2d((256, 256), (8, 8), 5)
    noise = (noise-np.min(noise))/(np.max(noise)-np.min(noise)) #normalize the noise from 0 to 1
    # Find indices of non-zero values in img
    nonzero_indices = np.nonzero(img)
    # Add noise only to non-zero values
    img[nonzero_indices] = noise[nonzero_indices]
    img = cm(img)
    # Modify img where blob has a value of 0
    img[blob == 0] = [0, 0, 0, 0]  # Setting background to zero
    return img

def generate_full_background():
    background_image = generate_background_image()
    background_image = gaussian_filter(background_image, sigma=1)

    # Define a static color (e.g., yellow)
    color = np.array([174/255,190/255,188/255])
    #color = np.array([204/255, 160/255, 39/255])  # Yellow

    # Add color to the grayscale background image
    colored_image = add_colour(background_image, color)
    print(colored_image.shape)
    colored_image = invert_colors(colored_image)
    colored_image = add_background(colored_image, np.array([100*2.2,90*2.2,80*2.2]))
    #colored_image = noisy("poisson", colored_image)
    #colored_image = add_background(colored_image, np.array([0.5,0.5,0.5]))
    # Display the result
    plt.figure
    plt.imshow(colored_image.astype(np.uint8))
    plt.show()
    return colored_image

image = generate_full_background()
np.random.seed(0)
