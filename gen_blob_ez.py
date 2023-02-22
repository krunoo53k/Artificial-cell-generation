import numpy as np
import matplotlib.pyplot as plt

def generate_blob(shape=(46, 46), intensity=255):
    x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1.0, 0.0
    g = intensity * np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

def generate_blobs(n_blobs=10, image_shape=(360, 363)):
    image = np.zeros(image_shape, dtype=np.uint8)
    for i in range(n_blobs):
        shape = np.random.randint(46, 20, size=2)
        intensity = np.random.randint(100, 255)
        x_start = np.random.randint(0, image_shape[0] - shape[0])
        x_end = x_start + shape[0]
        y_start = np.random.randint(0, image_shape[1] - shape[1])
        y_end = y_start + shape[1]
        blob = generate_blob(shape, intensity)
        image[x_start:x_end, y_start:y_end] = blob
    return image

blobs = generate_blobs()
plt.imshow(blobs, cmap='gray')
plt.show()
