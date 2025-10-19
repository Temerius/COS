"""
Демонстрационный скрипт для всех реализованных фильтров.
Показывает применение фильтров к изображениям и сохраняет результаты.
"""

import numpy as np
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Предупреждение: PIL не установлен. Визуализация недоступна.")

# Импортируем все фильтры
from base_filter import BaseFilter
from box_blur import BoxBlurFilter, BoxBlur3x3, BoxBlur5x5
from gaussian_blur import GaussianBlurFilter, GaussianBlur3x3, GaussianBlur5x5
from median_filter import MedianFilter, MedianFilter3x3, MedianFilter5x5
from sobel_filter import SobelFilter, SobelX, SobelY, SobelMagnitude
from filter_manager import FilterManager, FilterPipeline


def create_test_image(size: tuple = (200, 200)) -> np.ndarray:
    """
    Создать тестовое изображение с различными элементами.
    
    Args:
        size: Размер изображения (высота, ширина)
        
    Returns:
        Тестовое изображение
    """
    h, w = size
    image = np.ones((h, w), dtype=np.uint8) * 128
    
    # Добавляем квадрат
    image[50:150, 50:150] = 255
    
    # Добавляем круг
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x)**2 + (y - center_y - 50)**2 <= 30**2
    image[mask] = 200
    
    # Добавляем линии
    image[20:25, :] = 0
    image[:, 180:185] = 0
    
    return image


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """
    Добавить шум "соль и перец" к изображению.
    
    Args:
        image: Входное изображение
        amount: Доля пикселей с шумом (0.0 - 1.0)
        
    Returns:
        Изображение с шумом
    """
    noisy = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)
    
    # Соль (белые пиксели)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[tuple(coords)] = 255
    
    # Перец (чёрные пиксели)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[tuple(coords)] = 0
    
    return noisy


def demonstrate_all_filters():
    """Демонстрация всех реализованных фильтров."""
    
    print("="*70)
    print("ДЕМОНСТРАЦИЯ ФИЛЬТРОВ СВЁРТКИ ДЛЯ ИЗОБРАЖЕНИЙ")
    print("="*70)
    
    # Создаём тестовые изображения
    print("\n1. Создание тестовых изображений...")
    clean_image = create_test_image()
    noisy_image = add_salt_pepper_noise(clean_image, amount=0.05)
    print(f"   Создано чистое изображение: {clean_image.shape}")
    print(f"   Создано изображение с шумом: {noisy_image.shape}")
    
    # ========== BOX BLUR ==========
    print("\n" + "="*70)
    print("2. КОРОБОЧНОЕ РАЗМЫТИЕ (Box Blur)")
    print("="*70)
    
    box_blur_3 = BoxBlur3x3()
    box_blur_5 = BoxBlur5x5()
    box_blur_9 = BoxBlurFilter(kernel_size=9)
    
    print(f"\n   {box_blur_3}")
    result_box3 = box_blur_3.apply(clean_image)
    print(f"   Результат: {result_box3.shape}, dtype: {result_box3.dtype}")
    
    print(f"\n   {box_blur_5}")
    result_box5 = box_blur_5.apply(clean_image)
    
    print(f"\n   {box_blur_9}")
    result_box9 = box_blur_9.apply(clean_image)
    
    print("\n   Коробочное размытие усредняет значения в квадратной области.")
    print("   Чем больше ядро, тем сильнее эффект размытия.")
    
    # ========== GAUSSIAN BLUR ==========
    print("\n" + "="*70)
    print("3. РАЗМЫТИЕ ПО ГАУССУ (Gaussian Blur)")
    print("="*70)
    
    gaussian_3 = GaussianBlur3x3()
    gaussian_5 = GaussianBlur5x5()
    gaussian_custom = GaussianBlurFilter(kernel_size=7, sigma=2.0)
    
    print(f"\n   {gaussian_3}")
    result_gauss3 = gaussian_3.apply(clean_image)
    print(f"   Ядро свёртки:\n{gaussian_3.kernel}")
    
    print(f"\n   {gaussian_5}")
    result_gauss5 = gaussian_5.apply(clean_image)
    
    print(f"\n   {gaussian_custom}")
    result_gauss7 = gaussian_custom.apply(clean_image)
    
    print("\n   Гауссово размытие использует взвешенное усреднение.")
    print("   Более естественное размытие по сравнению с Box Blur.")
    
    # ========== MEDIAN FILTER ==========
    print("\n" + "="*70)
    print("4. МЕДИАННЫЙ ФИЛЬТР (Median Filter)")
    print("="*70)
    
    median_3 = MedianFilter3x3()
    median_5 = MedianFilter5x5()
    
    print(f"\n   {median_3}")
    print("   Применяем к изображению с шумом 'соль и перец'...")
    result_median3_noisy = median_3.apply(noisy_image)
    result_median3_clean = median_3.apply(clean_image)
    
    print(f"\n   {median_5}")
    result_median5_noisy = median_5.apply(noisy_image)
    
    print("\n   Медианный фильтр эффективно удаляет импульсный шум.")
    print("   Сохраняет резкие границы лучше, чем размытие.")
    
    # ========== SOBEL OPERATOR ==========
    print("\n" + "="*70)
    print("5. ОПЕРАТОР СОБЕЛЯ (Sobel Operator)")
    print("="*70)
    
    sobel_x = SobelX()
    sobel_y = SobelY()
    sobel_mag = SobelMagnitude()
    sobel_both = SobelFilter(direction='both')
    
    print(f"\n   {sobel_x}")
    result_sobel_x = sobel_x.apply(clean_image)
    print("   Обнаруживает вертикальные границы")
    
    print(f"\n   {sobel_y}")
    result_sobel_y = sobel_y.apply(clean_image)
    print("   Обнаруживает горизонтальные границы")
    
    print(f"\n   {sobel_mag}")
    result_sobel_mag = sobel_mag.apply(clean_image)
    print("   Вычисляет магнитуду градиента")
    
    print(f"\n   Ядро Собеля X:\n{SobelFilter.SOBEL_X}")
    print(f"\n   Ядро Собеля Y:\n{SobelFilter.SOBEL_Y}")
    
    print("\n   Оператор Собеля используется для обнаружения границ.")
    print("   Вычисляет градиент изображения в разных направлениях.")
    
    # ========== FILTER PIPELINE ==========
    print("\n" + "="*70)
    print("6. ПАЙПЛАЙН ФИЛЬТРОВ (Filter Pipeline)")
    print("="*70)
    
    # Создаём пайплайн для обработки зашумлённого изображения
    pipeline = FilterPipeline("Обработка зашумлённого изображения")
    pipeline.add_step(MedianFilter(3), "Удаление импульсного шума")
    pipeline.add_step(GaussianBlurFilter(3, 1.0), "Сглаживание")
    pipeline.add_step(SobelMagnitude(), "Выделение границ")
    
    print(f"\n{pipeline}")
    result_pipeline = pipeline.execute(noisy_image, verbose=True)
    
    # ========== FILTER MANAGER ==========
    print("\n" + "="*70)
    print("7. МЕНЕДЖЕР ФИЛЬТРОВ (Filter Manager)")
    print("="*70)
    
    manager = FilterManager()
    manager.register_filter('box_blur', BoxBlurFilter)
    manager.register_filter('gaussian_blur', GaussianBlurFilter)
    manager.register_filter('median', MedianFilter)
    manager.register_filter('sobel', SobelFilter)
    
    print(f"\nДоступные фильтры: {manager.get_available_filters()}")
    
    # Сравнение разных фильтров размытия
    filter_configs = [
        {'name': 'box_blur', 'params': {'kernel_size': 5}},
        {'name': 'gaussian_blur', 'params': {'kernel_size': 5, 'sigma': 1.4}},
        {'name': 'median', 'params': {'kernel_size': 5}},
    ]
    
    print("\nСравнение фильтров размытия...")
    results = manager.compare_filters(clean_image, filter_configs)
    print(f"Создано результатов: {len(results)}")
    
    # ========== ИТОГИ ==========
    print("\n" + "="*70)
    print("ИТОГИ")
    print("="*70)
    print("\nРеализованы следующие фильтры:")
    print("  1. Box Blur - коробочное размытие")
    print("  2. Gaussian Blur - размытие по Гауссу")
    print("  3. Median Filter - медианный фильтр")
    print("  4. Sobel Operator - оператор Собеля")
    print("\nДополнительные возможности:")
    print("  - Модульная архитектура с базовым классом")
    print("  - Поддержка grayscale и RGB изображений")
    print("  - Менеджер фильтров для удобного управления")
    print("  - Пайплайны для применения цепочек фильтров")
    print("  - Легко расширяется новыми фильтрами")
    print("\n" + "="*70)


def demonstrate_edge_detection_pipeline():
    """
    Демонстрация типичного пайплайна для обнаружения границ.
    """
    print("\n" + "="*70)
    print("БОНУС: ПАЙПЛАЙН ОБНАРУЖЕНИЯ ГРАНИЦ")
    print("="*70)
    
    # Создаём изображение с шумом
    image = create_test_image((300, 300))
    noisy = add_salt_pepper_noise(image, amount=0.03)
    
    # Классический пайплайн для обнаружения границ:
    # 1. Удаление шума медианным фильтром
    # 2. Сглаживание гауссовым фильтром
    # 3. Применение оператора Собеля
    
    pipeline = FilterPipeline("Edge Detection Pipeline")
    pipeline.add_step(
        MedianFilter(kernel_size=3),
        "Шаг 1: Удаление импульсного шума"
    )
    pipeline.add_step(
        GaussianBlurFilter(kernel_size=3, sigma=1.0),
        "Шаг 2: Гауссово сглаживание"
    )
    pipeline.add_step(
        SobelMagnitude(),
        "Шаг 3: Обнаружение границ оператором Собеля"
    )
    
    print(f"\n{pipeline}")
    result = pipeline.execute(noisy, verbose=True)
    
    print("\nЭтот пайплайн является стандартным подходом для обнаружения границ:")
    print("  - Медианный фильтр сохраняет границы при удалении шума")
    print("  - Гауссово размытие уменьшает оставшийся шум")
    print("  - Оператор Собеля выделяет границы объектов")


if __name__ == "__main__":
    # Устанавливаем seed для воспроизводимости
    np.random.seed(42)
    
    # Запускаем демонстрацию
    demonstrate_all_filters()
    demonstrate_edge_detection_pipeline()
    
    print("\n✓ Демонстрация завершена!")
    print("\nДля работы с реальными изображениями используйте PIL/Pillow:")
    print("  from PIL import Image")
    print("  img = np.array(Image.open('image.jpg').convert('L'))")
    print("  result = filter.apply(img)")
    print("  Image.fromarray(result).save('result.jpg')")