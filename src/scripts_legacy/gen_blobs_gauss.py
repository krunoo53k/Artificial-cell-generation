import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import random
from math import pi

blob_size = 128

def generate_blobs(num_blobs=9, blob_size=blob_size, x_std=10, y_std=10):
    blobs = []
    for i in range(num_blobs):
        x_center = np.random.randint(0, blob_size)
        y_center = np.random.randint(0, blob_size)
        x, y = np.meshgrid(np.arange(blob_size), np.arange(blob_size))
        blob = np.exp(-((x - x_center)**2 / (2 * x_std**2) + (y - y_center)**2 / (2 * y_std**2)))
        blobs.append(blob)
    return blobs

def generate_blob(blob_size=blob_size, x_std=30, y_std=30):
    x_std = random.uniform(20.0,30.0)
    y_std = random.uniform(20.0, 30.0)
    x_center = blob_size/2
    y_center = blob_size/2
    x, y = np.meshgrid(np.arange(blob_size), np.arange(blob_size))
    blob = np.exp(-((x - x_center)**2 / (2 * x_std**2) + (y - y_center)**2 / (2 * y_std**2)))
    return blob

"""
blobs = generate_blobs()

image = np.zeros((360,363))


for i, blob in enumerate(blobs):
    x_start = np.random.randint(0, 360 - blob_size)
    y_start = np.random.randint(0, 363 - blob_size)
    image[x_start:x_start + blob_size, y_start:y_start + blob_size] = blob
    plt.imshow(blob, cmap='gray')
    plt.title(f'Blob {i}')
    plt.show()

blurred = gaussian_filter(image, sigma = 7)
plt.imshow(blurred, cmap='gray')
plt.title('Final image')
plt.show()
"""



# Define normalized 2D Gaussian with rotation
def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1, theta=0):
    sx=random.uniform(1,2.5)
    sy = random.uniform(1,2.5)
    theta = random.uniform(-2*pi,2*pi)
    x_rot = (x - mx) * np.cos(theta) - (y - my) * np.sin(theta)
    y_rot = (x - mx) * np.sin(theta) + (y - my) * np.cos(theta)
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x_rot**2. / (2. * sx**2.)) + (y_rot**2. / (2. * sy**2.))))

x = np.linspace(-5, 5, 128)
y = np.linspace(-5, 5, 128)
x, y = np.meshgrid(x, y)  # Get 2D variables instead of 1D


z = gaus2d(x, y, mx=0, my=0, sx=1, sy=1, theta=np.pi/4)  # Rotated by 45 degrees (pi/4 radians)
z = (z-np.min(z))/(np.max(z)-np.min(z))
z[z<=0.2]=0
z[z!=0]=1
plt.imshow(z)
plt.show()


