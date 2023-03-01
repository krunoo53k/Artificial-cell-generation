from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt


image = data.binary_blobs(length=360, blob_size_fraction=0.05, volume_fraction=0.4) 
floati = image.astype(float)

#floati = np.multiply(floati, 10)
dtf = distance_transform_edt(floati)
dtf = np.multiply(dtf, 0.1)
hehe = np.subtract(floati, dtf)
print(floati)
print(dtf)
plt.imshow(floati, cmap='binary')
plt.title('Final image')
plt.show()
plt.imshow(dtf, cmap='binary')
plt.show()
#hehe = gaussian_filter(hehe, sigma=0.1)
plt.imshow(hehe, cmap='binary')
plt.show()