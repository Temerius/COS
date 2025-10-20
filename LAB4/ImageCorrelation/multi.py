import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Загружаем изображения
img_rgb = cv.imread('pic.png')
assert img_rgb is not None, "Файл не может быть прочитан, проверьте путь к файлу"
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

template = cv.imread('coin.png', cv.IMREAD_GRAYSCALE)
template_color = cv.imread('coin.png')
assert template is not None, "Файл не может быть прочитан, проверьте путь к файлу"
w, h = template.shape[::-1]

# Выполняем поиск шаблона и создаем матрицу корреляции
res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)

# Пороговое значение для поиска совпадений
threshold = 0.8
loc = np.where(res >= threshold)

# Копируем изображение, чтобы нарисовать на нем результаты
img_result = img_rgb.copy()
for pt in zip(*loc[::-1]):
    cv.rectangle(img_result, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

# Сохраняем изображение с результатами
cv.imwrite('res.png', img_result)

# Построение итогового изображения с помощью matplotlib
plt.figure(figsize=(12, 8))

# Отображение исходного изображения
plt.subplot(2, 2, 1)
plt.imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# Отображение шаблона (монеты)
plt.subplot(2, 2, 2)
plt.imshow(cv.cvtColor(template_color, cv.COLOR_BGR2RGB))
plt.title("Template (Coin)")
plt.axis('off')

# Отображение матрицы корреляции
plt.subplot(2, 2, 3)
plt.imshow(res, cmap='hot')  # Использование цветовой карты 'hot'
plt.title("Correlation Matrix")
plt.colorbar()  # Добавляем цветовую шкалу для визуализации значений корреляции
plt.axis('off')

# Отображение итогового изображения с прямоугольниками
plt.subplot(2, 2, 4)
plt.imshow(cv.cvtColor(img_result, cv.COLOR_BGR2RGB))
plt.title("Detected Coins")
plt.axis('off')

plt.tight_layout()
plt.savefig('correlation_results_mario.png', dpi=300, bbox_inches='tight')
