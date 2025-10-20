import numpy as np
from scipy.ndimage import convolve


class GausBlur:
    def __init__(self, kernel_size=5, sigma=1.0):
        if kernel_size % 2 == 0:
            raise ValueError("Размер ядра должен быть нечётным.")

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.generate_gaussian_kernel(kernel_size, sigma)

    def generate_gaussian_kernel(self, size, sigma):

        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

        return kernel / np.sum(kernel)

    def apply(self, image):
        """
        Применяет фильтр размытия по Гауссу к изображению.

        :param image: Входное изображение как 2D (ч/б) или 3D массив (цветное).
        :return: Размазанное изображение.
        """
        if image.ndim == 2:
            blurred_image = convolve(image, self.kernel, mode='reflect')
        elif image.ndim == 3:
            blurred_image = np.zeros_like(image)
            for channel in range(image.shape[2]):
                blurred_image[:, :, channel] = convolve(image[:, :, channel], self.kernel, mode='reflect')
        else:
            raise ValueError("Изображение должно быть 2D (ч/б) или 3D (цветное).")

        return blurred_image
    pass

