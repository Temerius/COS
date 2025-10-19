# sketch_filter.py
import numpy as np
from base_filter import BaseFilter
from sobel_filter import SobelFilter

class SketchFilter(BaseFilter):
    """
    Превращает изображение в карандашный эскиз.
    Алгоритм:
      1. Инвертируем изображение → получаем "негатив"
      2. Слабо размываем негатив (Gaussian blur)
      3. Делаем "Color Dodge" между оригиналом и размытым негативом
    """
    def __init__(self, blur_sigma: float = 5.0, blur_kernel: int = 21):
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        super().__init__(f"Sketch Filter (σ={blur_sigma})")
        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel

    def _create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)

    def _gaussian_blur_channel(self, channel: np.ndarray) -> np.ndarray:
        kernel = self._create_gaussian_kernel(self.blur_kernel, self.blur_sigma)
        pad = self.blur_kernel // 2
        padded = np.pad(channel, pad, mode='edge')
        blurred = np.zeros_like(channel, dtype=np.float64)
        h, w = channel.shape
        for i in range(h):
            for j in range(w):
                window = padded[i:i+self.blur_kernel, j:j+self.blur_kernel]
                blurred[i, j] = np.sum(window * kernel)
        return blurred

    def _color_dodge(self, base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Color Dodge blend mode: base / (1 - blend)"""
        # Нормализуем в [0, 1]
        base_f = base.astype(np.float64) / 255.0
        blend_f = blend.astype(np.float64) / 255.0
        # Избегаем деления на 1
        result = np.divide(base_f, np.maximum(1 - blend_f, 1e-6))
        result = np.clip(result, 0, 1) * 255
        return result.astype(np.uint8)

    def apply(self, image: np.ndarray) -> np.ndarray:
        self._validate_image(image)

        # Работаем с копией в float
        img_float = image.astype(np.float64)

        if image.ndim == 3:
            # Инвертируем каждый канал
            inverted = 255 - image
            # Размываем каждый канал
            blurred = np.zeros_like(image, dtype=np.float64)
            for c in range(image.shape[2]):
                blurred[:, :, c] = self._gaussian_blur_channel(inverted[:, :, c])
            # Применяем Color Dodge
            sketch = np.zeros_like(image)
            for c in range(image.shape[2]):
                sketch[:, :, c] = self._color_dodge(image[:, :, c], blurred[:, :, c])
            return sketch
        else:
            inverted = 255 - image
            blurred = self._gaussian_blur_channel(inverted)
            return self._color_dodge(image, blurred)