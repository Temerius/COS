import numpy as np
from base_filter import ConvolutionFilter


class BoxBlurFilter(ConvolutionFilter):

    
    def __init__(self, kernel_size: int = 3):

        if kernel_size % 2 == 0:
            raise ValueError("Размер ядра должен быть нечётным числом")
        

        kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
        kernel /= (kernel_size * kernel_size)
        
        super().__init__(f"Box Blur (kernel_size={kernel_size})", kernel)
        self.kernel_size_param = kernel_size
    
    @classmethod
    def create_with_size(cls, size: int) -> 'BoxBlurFilter':

        return cls(kernel_size=size)



class BoxBlur3x3(BoxBlurFilter):

    def __init__(self):
        super().__init__(kernel_size=3)


class BoxBlur5x5(BoxBlurFilter):

    def __init__(self):
        super().__init__(kernel_size=5)


class BoxBlur7x7(BoxBlurFilter):

    def __init__(self):
        super().__init__(kernel_size=7)



if __name__ == "__main__":

    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    

    blur_filter = BoxBlurFilter(kernel_size=5)
    filtered_image = blur_filter.apply(test_image)
    
    print(f"Фильтр: {blur_filter}")
    print(f"Входное изображение: {test_image.shape}, dtype: {test_image.dtype}")
    print(f"Выходное изображение: {filtered_image.shape}, dtype: {filtered_image.dtype}")
    print(f"Ядро свёртки:\n{blur_filter.kernel}")