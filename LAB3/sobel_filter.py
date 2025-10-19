import numpy as np
from base_filter import BaseFilter


class SobelFilter(BaseFilter):
    
    SOBEL_X = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    SOBEL_Y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)
    
    def __init__(self, direction: str = 'both', normalize: bool = True):

        valid_directions = ['x', 'y', 'both', 'magnitude']
        if direction not in valid_directions:
            raise ValueError(f"direction должен быть одним из {valid_directions}")
        
        super().__init__(f"Sobel Filter (direction={direction})")
        self.direction = direction
        self.normalize = normalize
        self.pad_width = 1
    
    def _convolve_2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        
        padded = np.pad(image, self.pad_width, mode='edge')
        
        h, w = image.shape
        kh, kw = kernel.shape
        

        result = np.zeros_like(image, dtype=np.float64)
        
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kh, j:j+kw]
                result[i, j] = np.sum(window * kernel)
        
        return result
    
    def _apply_sobel_2d(self, image: np.ndarray) -> np.ndarray:

        gx = self._convolve_2d(image, self.SOBEL_X)
        gy = self._convolve_2d(image, self.SOBEL_Y)
        
        if self.direction == 'x':
            result = np.abs(gx)
        elif self.direction == 'y':
            result = np.abs(gy)
        elif self.direction == 'both':
            result = np.abs(gx) + np.abs(gy)
        elif self.direction == 'magnitude':
            result = np.sqrt(gx**2 + gy**2)
        
        if self.normalize:
            result = self._normalize_to_uint8(result)
        
        return result
    
    def _normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:
        
        min_val = np.min(image)
        max_val = np.max(image)
        
        if max_val - min_val > 0:
            normalized = (image - min_val) / (max_val - min_val) * 255
        else:
            normalized = np.zeros_like(image)
        
        return normalized
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        self._validate_image(image)

        if image.ndim == 3:
            
            result = np.zeros_like(image, dtype=np.uint8)
            for c in range(image.shape[2]):
                channel_result = self._apply_sobel_2d(image[:, :, c])
                result[:, :, c] = channel_result
            return result
        else:
            
            result = self._apply_sobel_2d(image)
            return result
    
    def get_gradients(self, image: np.ndarray) -> tuple:
        
        self._validate_image(image)
        
       
        if image.ndim == 3:
            gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image
        
        gx = self._convolve_2d(gray, self.SOBEL_X)
        gy = self._convolve_2d(gray, self.SOBEL_Y)
        
        return gx, gy
    
    def get_gradient_direction(self, image: np.ndarray) -> np.ndarray:
        gx, gy = self.get_gradients(image)
        return np.arctan2(gy, gx)



class SobelX(SobelFilter):
    """Оператор Собеля для горизонтальных границ"""
    def __init__(self):
        super().__init__(direction='x')


class SobelY(SobelFilter):
    """Оператор Собеля для вертикальных границ"""
    def __init__(self):
        super().__init__(direction='y')


class SobelMagnitude(SobelFilter):
    """Оператор Собеля с вычислением магнитуды градиента"""
    def __init__(self):
        super().__init__(direction='magnitude')


if __name__ == "__main__":

    test_image = np.zeros((5, 5), dtype=np.uint8)
    test_image[2:4, 1:4] = 255
    test_image[3:4, 2:3] = 166
    

    sobel_both = SobelFilter(direction='both')
    sobel_x = SobelX()
    sobel_y = SobelY()
    sobel_mag = SobelMagnitude()
    
    result_both = sobel_both.apply(test_image)
    result_x = sobel_x.apply(test_image)
    result_y = sobel_y.apply(test_image)
    result_mag = sobel_mag.apply(test_image)
    
    print(f"Фильтр (both): {sobel_both}")
    print(f"Входное изображение: {test_image.shape}, dtype: {test_image.dtype}")
    print(f"Выходное изображение: {result_both.shape}, dtype: {result_both.dtype}")
    print(f"\nОператор Собеля обнаруживает границы объектов на изображении")
    print(f"Direction 'x' - вертикальные границы")
    print(f"Direction 'y' - горизонтальные границы")
    print(f"Direction 'both' - все границы (сумма)")
    print(f"Direction 'magnitude' - магнитуда градиента")