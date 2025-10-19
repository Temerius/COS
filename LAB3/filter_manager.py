import numpy as np
from typing import List, Dict, Type
from base_filter import BaseFilter


class FilterManager:

    
    def __init__(self):
  
        self._filters: Dict[str, Type[BaseFilter]] = {}
        self._filter_chain: List[BaseFilter] = []
    
    def register_filter(self, name: str, filter_class: Type[BaseFilter]) -> None:

        if name in self._filters:
            print(f"Предупреждение: Фильтр '{name}' уже зарегистрирован. Перезапись.")
        
        self._filters[name] = filter_class
        print(f"Зарегистрирован фильтр: {name}")
    
    def get_available_filters(self) -> List[str]:
        """
        Получить список доступных фильтров.
        
        Returns:
            Список названий зарегистрированных фильтров
        """
        return list(self._filters.keys())
    
    def create_filter(self, name: str, **kwargs) -> BaseFilter:
        """
        Создать экземпляр фильтра по имени.
        
        Args:
            name: Имя зарегистрированного фильтра
            **kwargs: Параметры для конструктора фильтра
            
        Returns:
            Экземпляр фильтра
        """
        if name not in self._filters:
            raise ValueError(f"Фильтр '{name}' не зарегистрирован. "
                           f"Доступные: {self.get_available_filters()}")
        
        return self._filters[name](**kwargs)
    
    def add_to_chain(self, filter_instance: BaseFilter) -> 'FilterManager':
        """
        Добавить фильтр в цепочку обработки.
        
        Args:
            filter_instance: Экземпляр фильтра
            
        Returns:
            self (для цепочки вызовов)
        """
        self._filter_chain.append(filter_instance)
        return self
    
    def clear_chain(self) -> None:
        """Очистить цепочку фильтров."""
        self._filter_chain.clear()
    
    def apply_chain(self, image: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Применить цепочку фильтров к изображению.
        
        Args:
            image: Входное изображение
            verbose: Выводить информацию о каждом шаге
            
        Returns:
            Результат применения всех фильтров
        """
        result = image.copy()
        
        for i, filter_instance in enumerate(self._filter_chain, 1):
            if verbose:
                print(f"Шаг {i}: Применение {filter_instance.name}...")
            
            result = filter_instance.apply(result)
        
        if verbose:
            print(f"Обработка завершена. Применено фильтров: {len(self._filter_chain)}")
        
        return result
    
    def apply_single(self, filter_name: str, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Применить один фильтр к изображению.
        
        Args:
            filter_name: Имя фильтра
            image: Входное изображение
            **kwargs: Параметры для фильтра
            
        Returns:
            Отфильтрованное изображение
        """
        filter_instance = self.create_filter(filter_name, **kwargs)
        return filter_instance.apply(image)
    
    def compare_filters(self, image: np.ndarray, filter_configs: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Применить несколько фильтров к одному изображению для сравнения.
        
        Args:
            image: Входное изображение
            filter_configs: Список конфигураций фильтров
                           [{'name': 'box_blur', 'params': {'kernel_size': 5}}, ...]
            
        Returns:
            Словарь {название: результат}
        """
        results = {'original': image.copy()}
        
        for config in filter_configs:
            name = config.get('name')
            params = config.get('params', {})
            
            if name not in self._filters:
                print(f"Предупреждение: Фильтр '{name}' не найден, пропускаем.")
                continue
            
            filter_instance = self.create_filter(name, **params)
            result = filter_instance.apply(image)
            
            # Создаём уникальное имя для результата
            result_name = f"{name}_{params}" if params else name
            results[result_name] = result
        
        return results
    
    def get_chain_description(self) -> str:
        """
        Получить описание текущей цепочки фильтров.
        
        Returns:
            Строка с описанием цепочки
        """
        if not self._filter_chain:
            return "Цепочка фильтров пуста"
        
        description = "Цепочка фильтров:\n"
        for i, filter_instance in enumerate(self._filter_chain, 1):
            description += f"  {i}. {filter_instance.name}\n"
        
        return description


class FilterPipeline:
    """
    Класс для создания и выполнения пайплайнов обработки изображений.
    Более удобный интерфейс для создания цепочек фильтров.
    """
    
    def __init__(self, name: str = "Pipeline"):
        """
        Инициализация пайплайна.
        
        Args:
            name: Название пайплайна
        """
        self.name = name
        self.steps: List[tuple] = []
    
    def add_step(self, filter_instance: BaseFilter, description: str = "") -> 'FilterPipeline':
        """
        Добавить шаг в пайплайн.
        
        Args:
            filter_instance: Экземпляр фильтра
            description: Описание шага
            
        Returns:
            self (для цепочки вызовов)
        """
        self.steps.append((filter_instance, description))
        return self
    
    def execute(self, image: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Выполнить пайплайн.
        
        Args:
            image: Входное изображение
            verbose: Выводить информацию о каждом шаге
            
        Returns:
            Результат обработки
        """
        result = image.copy()
        
        if verbose:
            print(f"Выполнение пайплайна: {self.name}")
            print(f"Входное изображение: {image.shape}, dtype: {image.dtype}")
            print("-" * 50)
        
        for i, (filter_instance, description) in enumerate(self.steps, 1):
            if verbose:
                step_desc = description if description else filter_instance.name
                print(f"Шаг {i}/{len(self.steps)}: {step_desc}")
            
            result = filter_instance.apply(result)
            
            if verbose:
                print(f"  Результат: {result.shape}, dtype: {result.dtype}")
        
        if verbose:
            print("-" * 50)
            print(f"Пайплайн завершён. Выполнено шагов: {len(self.steps)}")
        
        return result
    
    def __str__(self) -> str:
        description = f"Pipeline: {self.name}\n"
        for i, (filter_instance, desc) in enumerate(self.steps, 1):
            step_desc = desc if desc else filter_instance.name
            description += f"  {i}. {step_desc}\n"
        return description


# Пример использования
if __name__ == "__main__":
    from box_blur import BoxBlurFilter, BoxBlur5x5
    from gaussian_blur import GaussianBlurFilter
    from median_filter import MedianFilter
    from sobel_filter import SobelFilter
    
    # Создаём менеджер фильтров
    manager = FilterManager()
    
    # Регистрируем фильтры
    manager.register_filter('box_blur', BoxBlurFilter)
    manager.register_filter('gaussian_blur', GaussianBlurFilter)
    manager.register_filter('median', MedianFilter)
    manager.register_filter('sobel', SobelFilter)
    
    print("\nДоступные фильтры:", manager.get_available_filters())
    
    # Создаём тестовое изображение
    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # Пример 1: Применение одного фильтра
    print("\n" + "="*50)
    print("Пример 1: Применение одного фильтра")
    result1 = manager.apply_single('box_blur', test_image, kernel_size=5)
    print(f"Результат: {result1.shape}")
    
    # Пример 2: Цепочка фильтров
    print("\n" + "="*50)
    print("Пример 2: Цепочка фильтров")
    manager.clear_chain()
    manager.add_to_chain(MedianFilter(kernel_size=3))
    manager.add_to_chain(GaussianBlurFilter(kernel_size=5, sigma=1.4))
    
    print(manager.get_chain_description())
    result2 = manager.apply_chain(test_image, verbose=True)
    
    # Пример 3: FilterPipeline
    print("\n" + "="*50)
    print("Пример 3: FilterPipeline")
    
    pipeline = FilterPipeline("Обработка изображения")
    pipeline.add_step(MedianFilter(3), "Удаление шума")
    pipeline.add_step(GaussianBlurFilter(5, 1.0), "Сглаживание")
    pipeline.add_step(SobelFilter('magnitude'), "Выделение границ")
    
    print(pipeline)
    result3 = pipeline.execute(test_image, verbose=True)