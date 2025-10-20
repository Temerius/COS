import numpy as np
from debugpy.common.json import array
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from PIL import Image


class SobelFilter:
    def __init__(self):
        self.kernel_x = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]])

        self.kernel_y = np.array([[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]])

    def apply_2d_convolution(image, kernel):
        kernel_size = kernel.shape[0]
        radius = kernel_size // 2

        padded_image = np.pad(image, ((radius, radius), (radius, radius)), mode='reflect')
        result_image = np.zeros_like(image, dtype=np.float64)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                result_image[y, x] = np.sum(padded_image[y:y + kernel_size, x:x + kernel_size] * kernel)
        return result_image


    def apply(self, image):
        if image.ndim != 3:
            raise ValueError("Изображение должно быть цветным (3D массив).")
        gradient_magnitude = np.zeros_like(image, dtype=np.float64)
        for channel in range(3):
            grad_x = SobelFilter.apply_2d_convolution(image[:, :, channel], self.kernel_x)
            grad_y = SobelFilter.apply_2d_convolution(image[:, :, channel], self.kernel_y)
            vector = (1, 1)
            norm_vector = vector / np.linalg.norm(vector)
            dp_vector = array()
            for i in range(grad_x.shape[0]):
                for j in range(grad_x.shape[1]):
                    gradient = (grad_x[i,j], grad_y[i,j])
                    dp = np.dot(norm_vector, gradient)
                    dp = (dp + 1) * 0.5
                    dp_vector.add(dp)
                    # gradient_magnitude[:, :, channel] = np.sqrt(grad_x ** 2 + grad_y ** 2)
            gradient_magnitude[:, :, channel] = np.copy(dp_vector)


        # gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

        # gradient_magnitude = gradient_magnitude[:, :, :2]
        #dp = np.tensordot(gradient_magnitude, norm_vector, axes=([2], [0]))

        gradient_magnitude = np.clip(gradient_magnitude, 0, 1).astype(np.uint8)
        return gradient_magnitude



image_path = 'stas_part.jpg'

# image = Image.open(image_path).convert('L')  # Преобразуем изображение в ч/б
image = Image.open(image_path)
image_np = np.array(image)

sobel_filter = SobelFilter()
edge_image = sobel_filter.apply(image_np)
edge_image = Image.fromarray(edge_image)
edge_image.save("result.jpg")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Edges (Sobel Filter)")
plt.imshow(edge_image, cmap='gray')
plt.axis('off')

plt.show()


