import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

blob_size = 128

def generate_blobs(num_blobs=9, blob_size=blob_size, x_std=10, y_std=20):
    blobs = []
    for i in range(num_blobs):
        x_center = np.random.randint(0, blob_size)
        y_center = np.random.randint(0, blob_size)
        x, y = np.meshgrid(np.arange(blob_size), np.arange(blob_size))
        blob = np.exp(-((x - x_center)**2 / (2 * x_std**2) + (y - y_center)**2 / (2 * y_std**2)))
        blobs.append(blob)
    return blobs

blobs = generate_blobs()

image = np.zeros((360,363))

for i, blob in enumerate(blobs):
    x_start = np.random.randint(0, 360 - blob_size)
    y_start = np.random.randint(0, 363 - blob_size)
    image[x_start:x_start + blob_size, y_start:y_start + blob_size] = blob
    #plt.imshow(blob, cmap='gray')
    #plt.title(f'Blob {i}')
    #plt.show()

blurred = gaussian_filter(image, sigma = 7)
plt.imshow(blurred, cmap='gray')
plt.title('Final image')
plt.show()
