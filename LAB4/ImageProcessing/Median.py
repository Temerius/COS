import numpy as np
from scipy.ndimage import median_filter
from PIL import Image
import matplotlib.pyplot as plt


class MedianFilter:
    def __init__(self, kernel_size=3):
        if kernel_size % 2 == 0:
            raise ValueError("Размер ядра должен быть нечётным.")
        self.kernel_size = kernel_size


    def apply(self, image):

        radius = self.kernel_size // 2
        filtered_image = np.zeros_like(image)

        for c in range(3):
            padded_image = np.pad(image[:, :, c], radius, mode='reflect')
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    window = padded_image[y:y + self.kernel_size, x:x + self.kernel_size]
                    filtered_image[y, x, c] = np.median(window)
        return filtered_image


image_path = 'stas.jpg'
image = Image.open(image_path)
image_np = np.array(image)

median_filter_instance = MedianFilter(kernel_size=7)
filtered_image = median_filter_instance.apply(image_np)

filtered_image = Image.fromarray(filtered_image)
filtered_image.save("Median.jpg")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_np)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Filtered Image (Median Filter)")
plt.imshow(filtered_image)
plt.axis('off')
plt.show()
