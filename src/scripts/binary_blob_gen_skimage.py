from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt


image = data.binary_blobs(length=360, blob_size_fraction=0.05, volume_fraction=0.4) 
floati = image.astype(float)

#floati = np.multiply(floati, 10)
dtf = distance_transform_edt(floati)
dtf = np.multiply(dtf, 0.1)
final_image = np.subtract(floati, dtf)
print(floati)
print(dtf)
plt.imshow(floati, cmap='binary')
plt.title('base image')
plt.imsave("output\SNE\skimage_test\\base.png",floati, cmap='binary')
plt.show()
plt.imshow(dtf, cmap='binary')
plt.title('distance transform')
plt.imsave("output\SNE\skimage_test\\dtf.png",dtf, cmap='binary')
plt.show()
#final_image = gaussian_filter(final_image, sigma=0.1)
#final_image[final_image<=0.0] = np.NaN
plt.imshow(final_image, cmap='binary')
plt.title('Final image')
plt.imsave("output\SNE\skimage_test\\final.png",final_image, cmap='binary')
plt.show()