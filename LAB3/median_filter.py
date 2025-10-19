import numpy as np
from base_filter import BaseFilter


class MedianFilter(BaseFilter):

    
    def __init__(self, kernel_size: int = 3):

        if kernel_size % 2 == 0:
            raise ValueError("Размер окна должен быть нечётным числом")
        
        super().__init__(f"Median Filter (kernel_size={kernel_size})")
        self.kernel_size = kernel_size
        self.pad_width = kernel_size // 2
    
    def _apply_median_2d(self, image: np.ndarray) -> np.ndarray:


        padded = self._pad_image(image, self.pad_width, mode='edge')
        
        h, w = image.shape
        k = self.kernel_size
        

        result = np.zeros_like(image)
        

        for i in range(h):
            for j in range(w):

                window = padded[i:i+k, j:j+k]

                result[i, j] = np.median(window)
        
        return result
    
    def apply(self, image: np.ndarray) -> np.ndarray:

        self._validate_image(image)
        
        result = self._apply_to_channels(image, self._apply_median_2d)
        
        return result.astype(image.dtype)


class OptimizedMedianFilter(MedianFilter):

    
    def _apply_median_2d(self, image: np.ndarray) -> np.ndarray:

        from scipy.ndimage import median_filter as scipy_median
        
        try:
            return scipy_median(image, size=self.kernel_size, mode='nearest')
        except ImportError:
            return super()._apply_median_2d(image)



class MedianFilter3x3(MedianFilter):
    """Медианный фильтр 3x3"""
    def __init__(self):
        super().__init__(kernel_size=3)


class MedianFilter5x5(MedianFilter):
    """Медианный фильтр 5x5"""
    def __init__(self):
        super().__init__(kernel_size=5)


class MedianFilter7x7(MedianFilter):
    """Медианный фильтр 7x7"""
    def __init__(self):
        super().__init__(kernel_size=7)



if __name__ == "__main__":

    test_image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    

    noise_ratio = 0.1
    num_salt = int(noise_ratio * test_image.size * 0.5)
    num_pepper = int(noise_ratio * test_image.size * 0.5)
    

    coords = [np.random.randint(0, i - 1, num_salt) for i in test_image.shape]
    test_image[coords[0], coords[1]] = 255
    
    coords = [np.random.randint(0, i - 1, num_pepper) for i in test_image.shape]
    test_image[coords[0], coords[1]] = 0
    
    median_filter = MedianFilter(kernel_size=5)
    filtered_image = median_filter.apply(test_image)
    
    print(f"Фильтр: {median_filter}")
    print(f"Входное изображение: {test_image.shape}, dtype: {test_image.dtype}")
    print(f"Выходное изображение: {filtered_image.shape}, dtype: {filtered_image.dtype}")
    print(f"\nМедианный фильтр эффективно удаляет шум 'соль и перец'")