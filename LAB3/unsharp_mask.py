# unsharp_mask.py
import numpy as np
from base_filter import BaseFilter

class UnsharpMaskFilter(BaseFilter):
    """
    Фильтр повышения резкости (Unsharp Masking).
    Работает по формуле: sharpened = original + amount * (original - blurred)
    """
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.5, threshold: int = 0):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size должен быть нечётным")
        if sigma <= 0:
            raise ValueError("sigma должен быть > 0")
        if amount < 0:
            raise ValueError("amount должен быть >= 0")
        if threshold < 0:
            raise ValueError("threshold должен быть >= 0")
        
        super().__init__(f"Unsharp Mask (σ={sigma}, amount={amount})")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount
        self.threshold = threshold

    def _create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Создаёт ядро Гаусса."""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)

    def _apply_to_channel(self, channel: np.ndarray) -> np.ndarray:
        """Применяет фильтр к одному каналу."""
        # Создаём размытое изображение
        kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)
        pad = self.kernel_size // 2
        padded = np.pad(channel, pad, mode='edge')
        blurred = np.zeros_like(channel, dtype=np.float64)

        h, w = channel.shape
        for i in range(h):
            for j in range(w):
                window = padded[i:i+self.kernel_size, j:j+self.kernel_size]
                blurred[i, j] = np.sum(window * kernel)

        # Вычисляем маску и усиливаем
        mask = channel.astype(np.float64) - blurred
        if self.threshold > 0:
            mask = np.where(np.abs(mask) < self.threshold, 0, mask)

        sharpened = channel.astype(np.float64) + self.amount * mask
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def apply(self, image: np.ndarray) -> np.ndarray:
        self._validate_image(image)

        if image.ndim == 3:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = self._apply_to_channel(image[:, :, c])
            return result
        else:
            return self._apply_to_channel(image)