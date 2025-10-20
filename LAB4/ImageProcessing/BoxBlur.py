import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import matplotlib.pyplot as plt


class BoxBlurFilter:
    def __init__(self, kernel_size=3):

        if kernel_size % 2 == 0:
            kernel_size += 1

        self.kernel_size = kernel_size
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    def apply(self, image):
        if image.ndim == 2:
            blurred_image = convolve(image, self.kernel, mode='reflect')
        elif image.ndim == 3:
            blurred_image = np.zeros_like(image)
            for channel in range(image.shape[2]):
                blurred_image[:, :, channel] = convolve(image[:, :, channel], self.kernel, mode='reflect')
        else:
            raise ValueError("Изображение должно быть 2D (ч/б) или 3D (цветное).")

        return blurred_image


image_path = 'stas.jpg'
image = Image.open(image_path)
image_np = np.array(image)

box_blur = BoxBlurFilter(kernel_size=17)
blurred_image = box_blur.apply(image_np)

filtered_image = Image.fromarray(blurred_image)
filtered_image.save("BoxBlur.jpg")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_np)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Blurred Image (Box Blur)")
plt.imshow(blurred_image)
plt.axis('off')
plt.show()