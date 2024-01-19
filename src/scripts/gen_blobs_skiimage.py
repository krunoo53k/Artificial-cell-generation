import numpy as np
from skimage.draw import random_shapes
import matplotlib.pyplot as plt

def generate_unconventional_blobs(num_blobs=3, shape='circle', size=100, max_size=100):
    blobs = []
    for i in range(num_blobs):
        image, points = random_shapes((360, 363), shape=shape, max_size=max_size, min_size=size, min_shapes=1, max_shapes=8)
        blobs.append(image.astype(float))
    return blobs

"""
blobs = generate_unconventional_blobs(num_blobs=3, shape='circle', size=30, max_size=60)

image = np.zeros((360, 363))
for blob in blobs:
    x_start = np.random.randint(0, image.shape[0]-blob.shape[0])
    y_start = np.random.randint(0, image.shape[1]-blob.shape[1])
    x_end = x_start + blob.shape[0]
    y_end = y_start + blob.shape[1]
    image[x_start:x_end, y_start:y_end] = blob
"""

image, points = random_shapes((128, 128), shape='circle', max_size=128, min_size=72, min_shapes=1, max_shapes=1)
plt.imshow(image)
plt.show()