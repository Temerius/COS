from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from GausBlur import GausBlur


image_path = 'stas.jpg'
image = Image.open(image_path)
image_np = np.array(image)


gaussian_blur = GausBlur(kernel_size=15, sigma=15)
blurred_image = gaussian_blur.apply(image_np)

filtered_image = Image.fromarray(blurred_image)
filtered_image.save("GaussenBlur.jpg")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_np)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Blurred Image (Gaussian Blur)")
plt.imshow(blurred_image)
plt.axis('off')
plt.show()