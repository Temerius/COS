import numpy as np
from base_filter import ConvolutionFilter


class GaussianBlurFilter(ConvolutionFilter):
    
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
 
        if kernel_size % 2 == 0:
            raise ValueError("Размер ядра должен быть нечётным числом")
        
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        
        super().__init__(f"Gaussian Blur (size={kernel_size}, sigma={sigma})", kernel)
        self.sigma = sigma
        self.kernel_size_param = kernel_size
    
    @staticmethod
    def _create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        

        ax = np.arange(-size // 2 + 1, size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    @classmethod
    def create_auto_sigma(cls, kernel_size: int) -> 'GaussianBlurFilter':
        """

        Используется эвристика: sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8

        """
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        return cls(kernel_size=kernel_size, sigma=sigma)


class GaussianBlur3x3(GaussianBlurFilter):

    def __init__(self):
        super().__init__(kernel_size=3, sigma=1.0)


class GaussianBlur5x5(GaussianBlurFilter):

    def __init__(self):
        super().__init__(kernel_size=5, sigma=1.4)


class GaussianBlur7x7(GaussianBlurFilter):

    def __init__(self):
        super().__init__(kernel_size=7, sigma=2.0)



if __name__ == "__main__":
    gaussian_filter = GaussianBlurFilter(kernel_size=5, sigma=1.4)
    
    print(f"Фильтр: {gaussian_filter}")
    print(f"Ядро свёртки:\n{gaussian_filter.kernel}")
    print(f"Сумма элементов ядра: {np.sum(gaussian_filter.kernel):.6f}")
    
    print("\nВизуализация ядра (умножено на 100 для наглядности):")
    print((gaussian_filter.kernel * 100).astype(int))
    
    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    filtered_image = gaussian_filter.apply(test_image)
    
    print(f"\nВходное изображение: {test_image.shape}")
    print(f"Выходное изображение: {filtered_image.shape}")