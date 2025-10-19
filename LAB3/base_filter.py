from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class BaseFilter(ABC):
    
    def __init__(self, name: str):
  
        self.name = name
    
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        pass
    
    def _validate_image(self, image: np.ndarray) -> None:
        
        if not isinstance(image, np.ndarray):
            raise ValueError("Изображение должно быть numpy array")
        
        if image.ndim not in [2, 3]:
            raise ValueError("Изображение должно быть 2D (grayscale) или 3D (RGB)")
    
    def _pad_image(self, image: np.ndarray, pad_width: int, mode: str = 'edge') -> np.ndarray:
        
        if image.ndim == 2:
            return np.pad(image, pad_width, mode=mode)
        else:
            return np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode=mode)
    
    def _apply_to_channels(self, image: np.ndarray, filter_func, *args, **kwargs) -> np.ndarray:
        
        if image.ndim == 2:
            return filter_func(image, *args, **kwargs)
        else:
            channels = []
            for i in range(image.shape[2]):
                filtered_channel = filter_func(image[:, :, i], *args, **kwargs)
                channels.append(filtered_channel)
            return np.stack(channels, axis=2)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.name}"
    
    def __repr__(self) -> str:
        return self.__str__()


class ConvolutionFilter(BaseFilter):
    
    def __init__(self, name: str, kernel: np.ndarray):
        
        super().__init__(name)
        self.kernel = kernel
        self.kernel_size = kernel.shape[0]
        self.pad_width = self.kernel_size // 2
    
    def _convolve_2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        
        padded = self._pad_image(image, self.pad_width, mode='edge')
        
        h, w = image.shape
        kh, kw = kernel.shape
        
        result = np.zeros_like(image, dtype=np.float64)
        

        for i in range(h):
            for j in range(w):
                window = padded[i:i+kh, j:j+kw]
                result[i, j] = np.sum(window * kernel)
        
        return result
    
    def apply(self, image: np.ndarray) -> np.ndarray:
       
        self._validate_image(image)
        
        result = self._apply_to_channels(image, self._convolve_2d, self.kernel)
        
        result = np.clip(result, 0, 255)
        
        return result.astype(image.dtype)