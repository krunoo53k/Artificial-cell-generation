import numpy as np
import matplotlib.pyplot as plt
import random
from math import pi
from utils import transform_blob

blob_size = 128


def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1, theta=0):
    x_rot = (x - mx) * np.cos(theta) - (y - my) * np.sin(theta)
    y_rot = (x - mx) * np.sin(theta) + (y - my) * np.cos(theta)
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x_rot ** 2. / (2. * sx ** 2.)) + (y_rot ** 2. / (2. * sy ** 2.))))


def generate_blob(size=128):
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    x, y = np.meshgrid(x, y)  # Get 2D variables instead of 1D

    sx = random.uniform(1, 2.5)
    sy = random.uniform(1, 2.5)
    theta = random.uniform(-2 * pi, 2 * pi)

    z = gaus2d(x, y, mx=0, my=0, sx=sx, sy=sy, theta=theta)
    z = (z - np.min(z)) / (np.max(z) - np.min(z))
    z[z <= 0.2] = 0
    z[z != 0] = 1
    return z


def color_blob(blob):
    image = transform_blob(blob)
    cell_color = np.array((0.64, 0.54, 0.6, 0))
    colored_image = np.full((128, 128, 4), cell_color)
    colored_image[:, :, 3] = image
    return colored_image


if __name__ == "__main__":
    blob_img = generate_blob()
    img = color_blob(blob_img)
    plt.imshow(img)
    plt.show()
