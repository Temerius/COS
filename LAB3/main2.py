

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import threading
from typing import Optional, Tuple

from base_filter import BaseFilter
from box_blur import BoxBlurFilter
from gaussian_blur import GaussianBlurFilter
from median_filter import MedianFilter
from sobel_filter import SobelFilter
from unsharp_mask import UnsharpMaskFilter
from sketch_filter import SketchFilter


class ImageFilterApp:
    """Главное приложение для работы с фильтрами изображений."""
    
    def __init__(self, root):
        """
        Инициализация приложения.
        
        Args:
            root: Корневое окно tkinter
        """
        self.root = root
        self.root.title("Фильтры свёртки для изображений")
        self.root.geometry("1500x900")
        
        # Данные изображений
        self.original_image: Optional[np.ndarray] = None
        self.filtered_image: Optional[np.ndarray] = None
        self.current_filter: Optional[BaseFilter] = None
        self.processing = False
        
        # Данные для выделения области
        self.selection_mode = False
        self.selection_start: Optional[Tuple[int, int]] = None
        self.selection_end: Optional[Tuple[int, int]] = None
        self.selection_rect_id = None
        self.mask: Optional[np.ndarray] = None
        
        # Параметры масштабирования (для сохранения размера)
        self.display_scale = 1.0
        self.display_size: Optional[Tuple[int, int]] = None
        
        # Создаём интерфейс
        self._create_ui()
        
        # Загружаем тестовое изображение
        self._load_test_image()
    
    def _create_ui(self):
        """Создать пользовательский интерфейс."""
        
        # Главный контейнер
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка сетки
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # === ПАНЕЛЬ УПРАВЛЕНИЯ (СЛЕВА) ===
        control_frame = ttk.LabelFrame(main_container, text="Управление", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Добавляем скроллбар для панели управления
        canvas_control = tk.Canvas(control_frame, width=280)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas_control.yview)
        scrollable_frame = ttk.Frame(canvas_control)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_control.configure(scrollregion=canvas_control.bbox("all"))
        )
        
        canvas_control.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_control.configure(yscrollcommand=scrollbar.set)
        
        canvas_control.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Кнопки загрузки
        ttk.Button(scrollable_frame, text="📂 Загрузить изображение", 
                  command=self._load_image, width=25).pack(pady=5)
        ttk.Button(scrollable_frame, text="🎲 Тестовое изображение", 
                  command=self._load_test_image, width=25).pack(pady=5)
        ttk.Button(scrollable_frame, text="💾 Сохранить результат", 
                  command=self._save_image, width=25).pack(pady=5)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # === ВЫДЕЛЕНИЕ ОБЛАСТИ ===
        selection_frame = ttk.LabelFrame(scrollable_frame, text="Выделение области", padding="5")
        selection_frame.pack(fill='x', pady=5)
        
        self.selection_enabled_var = tk.BooleanVar(value=False)
        selection_check = ttk.Checkbutton(
            selection_frame, 
            text="Включить режим выделения",
            variable=self.selection_enabled_var,
            command=self._toggle_selection_mode
        )
        selection_check.pack(anchor='w', pady=2)
        
        ttk.Label(selection_frame, text="Применить фильтр к:", 
                 font=('Arial', 9, 'bold')).pack(anchor='w', pady=(5, 2))
        
        self.mask_mode_var = tk.StringVar(value='selected')
        ttk.Radiobutton(
            selection_frame, 
            text="Выделенной области",
            variable=self.mask_mode_var, 
            value='selected'
        ).pack(anchor='w', padx=10)
        
        ttk.Radiobutton(
            selection_frame, 
            text="Невыделенной области",
            variable=self.mask_mode_var, 
            value='unselected'
        ).pack(anchor='w', padx=10)
        
        ttk.Button(
            selection_frame, 
            text="🗑️ Очистить выделение", 
            command=self._clear_selection
        ).pack(fill='x', pady=5)
        
        self.selection_info_var = tk.StringVar(value="Выделение не создано")
        ttk.Label(
            selection_frame, 
            textvariable=self.selection_info_var,
            font=('Arial', 8),
            foreground='blue'
        ).pack(anchor='w')
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # === ФИЛЬТРЫ ===
        filters_label = ttk.Label(scrollable_frame, text="Фильтры:", font=('Arial', 11, 'bold'))
        filters_label.pack(anchor='w', pady=(10, 5))
        
        # Box Blur
        box_frame = ttk.LabelFrame(scrollable_frame, text="Box Blur", padding="5")
        box_frame.pack(fill='x', pady=5)
        
        self.box_size_var = tk.IntVar(value=5)
        ttk.Label(box_frame, text="Размер ядра:").pack(anchor='w')
        box_scale = ttk.Scale(box_frame, from_=3, to=45, variable=self.box_size_var, 
                 orient='horizontal')
        box_scale.pack(fill='x')
        box_scale.configure(command=lambda v: self._update_box_label())
        self.box_label = ttk.Label(box_frame, text="5")
        self.box_label.pack(anchor='w')
        ttk.Button(box_frame, text="Применить Box Blur", 
                  command=lambda: self._apply_filter('box')).pack(fill='x', pady=5)
        
        # Gaussian Blur
        gauss_frame = ttk.LabelFrame(scrollable_frame, text="Gaussian Blur", padding="5")
        gauss_frame.pack(fill='x', pady=5)
        
        self.gauss_size_var = tk.IntVar(value=5)
        self.gauss_sigma_var = tk.DoubleVar(value=1.4)
        
        ttk.Label(gauss_frame, text="Размер ядра:").pack(anchor='w')
        gauss_scale = ttk.Scale(gauss_frame, from_=3, to=45, variable=self.gauss_size_var,
                 orient='horizontal')
        gauss_scale.pack(fill='x')
        gauss_scale.configure(command=lambda v: self._update_gauss_label())
        self.gauss_size_label = ttk.Label(gauss_frame, text="5")
        self.gauss_size_label.pack(anchor='w')
        
        ttk.Label(gauss_frame, text="Sigma:").pack(anchor='w')
        sigma_scale = ttk.Scale(gauss_frame, from_=0.5, to=50.0, variable=self.gauss_sigma_var,
                 orient='horizontal')
        sigma_scale.pack(fill='x')
        sigma_scale.configure(command=lambda v: self._update_sigma_label())
        self.gauss_sigma_label = ttk.Label(gauss_frame, text="1.4")
        self.gauss_sigma_label.pack(anchor='w')
        
        ttk.Button(gauss_frame, text="Применить Gaussian Blur", 
                  command=lambda: self._apply_filter('gaussian')).pack(fill='x', pady=5)
        
        # Median Filter
        median_frame = ttk.LabelFrame(scrollable_frame, text="Median Filter", padding="5")
        median_frame.pack(fill='x', pady=5)
        
        self.median_size_var = tk.IntVar(value=5)
        ttk.Label(median_frame, text="Размер окна:").pack(anchor='w')
        median_scale = ttk.Scale(median_frame, from_=3, to=45, variable=self.median_size_var,
                 orient='horizontal')
        median_scale.pack(fill='x')
        median_scale.configure(command=lambda v: self._update_median_label())
        self.median_label = ttk.Label(median_frame, text="5")
        self.median_label.pack(anchor='w')
        ttk.Button(median_frame, text="Применить Median Filter", 
                  command=lambda: self._apply_filter('median')).pack(fill='x', pady=5)
        

        # Unsharp Mask
        sharp_frame = ttk.LabelFrame(scrollable_frame, text="Unsharp Mask (резкость)", padding="5")
        sharp_frame.pack(fill='x', pady=5)
        
        self.sharp_sigma_var = tk.DoubleVar(value=1.0)
        self.sharp_amount_var = tk.DoubleVar(value=1.5)
        
        ttk.Label(sharp_frame, text="Sigma (размытие):").pack(anchor='w')
        ttk.Scale(sharp_frame, from_=0.5, to=3.0, variable=self.sharp_sigma_var, orient='horizontal').pack(fill='x')
        ttk.Label(sharp_frame, textvariable=tk.StringVar(value="1.0"), 
                 text="1.0").pack(anchor='w')  
        
        ttk.Label(sharp_frame, text="Сила:").pack(anchor='w')
        ttk.Scale(sharp_frame, from_=0.5, to=3.0, variable=self.sharp_amount_var, orient='horizontal').pack(fill='x')
        
        ttk.Button(sharp_frame, text="Применить резкость", 
                  command=lambda: self._apply_filter('unsharp')).pack(fill='x', pady=5)

        # Sketch
        ttk.Button(scrollable_frame, text="🎨 Применить эскиз", 
                  command=lambda: self._apply_filter('sketch')).pack(fill='x', pady=5)
        
        # Sobel Operator
        sobel_frame = ttk.LabelFrame(scrollable_frame, text="Sobel Operator", padding="5")
        sobel_frame.pack(fill='x', pady=5)
        
        self.sobel_direction_var = tk.StringVar(value='magnitude')
        directions = [('Магнитуда', 'magnitude'), ('Оба направления', 'both'),
                     ('Горизонтальные (X)', 'x'), ('Вертикальные (Y)', 'y')]
        
        for text, value in directions:
            ttk.Radiobutton(sobel_frame, text=text, variable=self.sobel_direction_var, 
                           value=value).pack(anchor='w')
        
        ttk.Button(sobel_frame, text="Применить Sobel", 
                  command=lambda: self._apply_filter('sobel')).pack(fill='x', pady=5)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Кнопки действий
        ttk.Button(scrollable_frame, text="↩️ Сбросить к оригиналу", 
                  command=self._reset_image, width=25).pack(pady=5)
        ttk.Button(scrollable_frame, text="🗑️ Добавить шум", 
                  command=self._add_noise, width=25).pack(pady=5)
        
        # Статус
        self.status_var = tk.StringVar(value="Готов к работе")
        status_label = ttk.Label(scrollable_frame, textvariable=self.status_var, 
                                relief='sunken', anchor='w', wraplength=250)
        status_label.pack(side='bottom', fill='x', pady=(10, 0))
        
        # === ПАНЕЛЬ ИЗОБРАЖЕНИЙ (СПРАВА) ===
        images_frame = ttk.Frame(main_container)
        images_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(1, weight=1)
        
        # Заголовок
        title_label = ttk.Label(images_frame, text="Просмотр изображений", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Оригинальное изображение
        original_frame = ttk.LabelFrame(images_frame, text="Оригинал (кликните и тяните для выделения)", padding="5")
        original_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        
        self.original_canvas = tk.Canvas(original_frame, bg='#2b2b2b', highlightthickness=0)
        self.original_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Привязываем события для выделения
        self.original_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.original_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.original_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        
        # Отфильтрованное изображение
        filtered_frame = ttk.LabelFrame(images_frame, text="Результат", padding="5")
        filtered_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        filtered_frame.columnconfigure(0, weight=1)
        filtered_frame.rowconfigure(0, weight=1)
        
        self.filtered_canvas = tk.Canvas(filtered_frame, bg='#2b2b2b', highlightthickness=0)
        self.filtered_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Информация о фильтре
        self.filter_info_var = tk.StringVar(value="Фильтр не применён")
        info_label = ttk.Label(images_frame, textvariable=self.filter_info_var, 
                              font=('Arial', 10, 'italic'))
        info_label.grid(row=2, column=0, columnspan=2, pady=5)
    
    # === ОБРАБОТЧИКИ СЛАЙДЕРОВ ===
    
    def _update_box_label(self):
        """Обновить label для Box Blur и сделать значение нечётным."""
        val = int(self.box_size_var.get())
        if val % 2 == 0:
            val += 1
            self.box_size_var.set(val)
        self.box_label.config(text=str(val))
    
    def _update_gauss_label(self):
        """Обновить label для Gaussian Blur и сделать значение нечётным."""
        val = int(self.gauss_size_var.get())
        if val % 2 == 0:
            val += 1
            self.gauss_size_var.set(val)
        self.gauss_size_label.config(text=str(val))
    
    def _update_sigma_label(self):
        """Обновить label для sigma."""
        val = self.gauss_sigma_var.get()
        self.gauss_sigma_label.config(text=f"{val:.2f}")
    
    def _update_median_label(self):
        """Обновить label для Median Filter и сделать значение нечётным."""
        val = int(self.median_size_var.get())
        if val % 2 == 0:
            val += 1
            self.median_size_var.set(val)
        self.median_label.config(text=str(val))
    
    # === ВЫДЕЛЕНИЕ ОБЛАСТИ ===
    
    def _toggle_selection_mode(self):
        """Переключить режим выделения."""
        self.selection_mode = self.selection_enabled_var.get()
        if self.selection_mode:
            self.status_var.set("Режим выделения: кликните и тяните на оригинале")
        else:
            self.status_var.set("Режим выделения отключён")
            self._clear_selection()
    
    def _on_canvas_press(self, event):
        """Обработчик нажатия мыши на canvas."""
        if not self.selection_mode or self.original_image is None:
            return
        
        # Сохраняем начальную точку
        self.selection_start = (event.x, event.y)
        self.selection_end = None
        
        # Удаляем старый прямоугольник
        if self.selection_rect_id:
            self.original_canvas.delete(self.selection_rect_id)
            self.selection_rect_id = None
    
    def _on_canvas_drag(self, event):
        """Обработчик перетаскивания мыши."""
        if not self.selection_mode or self.selection_start is None:
            return
        
        # Удаляем старый прямоугольник
        if self.selection_rect_id:
            self.original_canvas.delete(self.selection_rect_id)
        
        # Рисуем новый прямоугольник
        self.selection_rect_id = self.original_canvas.create_rectangle(
            self.selection_start[0], self.selection_start[1],
            event.x, event.y,
            outline='yellow', width=2, dash=(5, 5)
        )
    
    def _on_canvas_release(self, event):
        """Обработчик отпускания мыши."""
        if not self.selection_mode or self.selection_start is None:
            return
        
        self.selection_end = (event.x, event.y)
        self._create_mask()
    
    def _create_mask(self):
        """Создать маску на основе выделенной области."""
        if self.selection_start is None or self.selection_end is None or self.original_image is None:
            return
        
        # Получаем координаты в пикселях оригинального изображения
        img_coords = self._canvas_to_image_coords(
            self.selection_start[0], self.selection_start[1],
            self.selection_end[0], self.selection_end[1]
        )
        
        if img_coords is None:
            return
        
        x1, y1, x2, y2 = img_coords
        
        # Создаём маску
        h, w = self.original_image.shape[:2]
        self.mask = np.zeros((h, w), dtype=bool)
        self.mask[y1:y2, x1:x2] = True
        
        # Обновляем информацию
        area = (y2 - y1) * (x2 - x1)
        total = h * w
        percentage = (area / total) * 100
        self.selection_info_var.set(
            f"Выделено: {x2-x1}x{y2-y1} пикселей ({percentage:.1f}%)"
        )
        self.status_var.set("Область выделена! Примените фильтр")
    
    def _canvas_to_image_coords(self, cx1, cy1, cx2, cy2):
        """
        Преобразовать координаты canvas в координаты изображения.
        
        Returns:
            Tuple (x1, y1, x2, y2) или None
        """
        if self.display_size is None or self.original_image is None:
            return None
        
        # Получаем размеры canvas и отображаемого изображения
        canvas_w = self.original_canvas.winfo_width()
        canvas_h = self.original_canvas.winfo_height()
        
        disp_w, disp_h = self.display_size
        img_h, img_w = self.original_image.shape[:2]
        
        # Вычисляем смещение (изображение центрировано)
        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2
        
        # Преобразуем координаты
        img_x1 = int((cx1 - offset_x) / disp_w * img_w)
        img_y1 = int((cy1 - offset_y) / disp_h * img_h)
        img_x2 = int((cx2 - offset_x) / disp_w * img_w)
        img_y2 = int((cy2 - offset_y) / disp_h * img_h)
        
        # Нормализуем (чтобы x1 < x2, y1 < y2)
        img_x1, img_x2 = sorted([img_x1, img_x2])
        img_y1, img_y2 = sorted([img_y1, img_y2])
        
        # Ограничиваем границами изображения
        img_x1 = max(0, min(img_x1, img_w))
        img_x2 = max(0, min(img_x2, img_w))
        img_y1 = max(0, min(img_y1, img_h))
        img_y2 = max(0, min(img_y2, img_h))
        
        return (img_x1, img_y1, img_x2, img_y2)
    
    def _clear_selection(self):
        """Очистить выделение."""
        self.selection_start = None
        self.selection_end = None
        self.mask = None
        
        if self.selection_rect_id:
            self.original_canvas.delete(self.selection_rect_id)
            self.selection_rect_id = None
        
        self.selection_info_var.set("Выделение не создано")
        self._update_displays()
    
    # === РАБОТА С ИЗОБРАЖЕНИЯМИ ===
    
    def _load_image(self):
        """Загрузить изображение из файла."""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("Все файлы", "*.*")
            ]
        )
        
        if file_path:
            try:
                img = Image.open(file_path)
                img = img.convert('RGB')
                self.original_image = np.array(img)
                self.filtered_image = None
                self.mask = None
                self.display_size = None  # Сброс размера
                self._update_displays()
                self.status_var.set(f"Загружено: {file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{e}")
    
    def _load_test_image(self):
        """Загрузить тестовое изображение."""
        size = 600
        image = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Фон - градиент
        for i in range(size):
            image[i, :] = [100 + i // 4, 150, 200 - i // 4]
        
        # Красный квадрат
        image[100:300, 100:300] = [255, 50, 50]
        
        # Зелёный круг
        center_y, center_x = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        mask = (x - center_x - 100)**2 + (y - center_y)**2 <= 80**2
        image[mask] = [50, 255, 50]
        
        # Синий круг
        mask = (x - center_x + 100)**2 + (y - center_y)**2 <= 80**2
        image[mask] = [50, 50, 255]
        
        # Жёлтый треугольник
        for i in range(150):
            start = 400 - i // 2
            end = 400 + i // 2
            image[350 + i, start:end] = [255, 255, 0]
        
        # Белые линии
        image[30:40, :] = [255, 255, 255]
        image[:, 560:570] = [255, 255, 255]
        
        # Текстура в углу
        texture = np.random.randint(100, 200, (80, 120, 3), dtype=np.uint8)
        image[500:580, 20:140] = texture
        
        self.original_image = image
        self.filtered_image = None
        self.mask = None
        self.display_size = None
        self._update_displays()
        self.status_var.set("Загружено тестовое цветное изображение")
    
    def _save_image(self):
        """Сохранить отфильтрованное изображение."""
        if self.filtered_image is None:
            messagebox.showwarning("Предупреждение", "Нет отфильтрованного изображения для сохранения")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Сохранить изображение",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("Все файлы", "*.*")
            ]
        )
        
        if file_path:
            try:
                if self.filtered_image.ndim == 3:
                    img = Image.fromarray(self.filtered_image.astype(np.uint8))
                else:
                    img = Image.fromarray(self.filtered_image.astype(np.uint8), mode='L')
                img.save(file_path)
                self.status_var.set(f"Сохранено: {file_path}")
                messagebox.showinfo("Успех", "Изображение успешно сохранено!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить изображение:\n{e}")
    
    def _add_noise(self):
        """Добавить шум 'соль и перец' к изображению."""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
        
        noisy = self.original_image.copy()
        amount = 0.05
        
        if noisy.ndim == 3:
            h, w, c = noisy.shape
            num_salt = int(amount * h * w * 0.5)
            num_pepper = int(amount * h * w * 0.5)
            
            coords_y = np.random.randint(0, h, num_salt)
            coords_x = np.random.randint(0, w, num_salt)
            noisy[coords_y, coords_x] = 255
            
            coords_y = np.random.randint(0, h, num_pepper)
            coords_x = np.random.randint(0, w, num_pepper)
            noisy[coords_y, coords_x] = 0
        else:
            num_salt = int(amount * noisy.size * 0.5)
            num_pepper = int(amount * noisy.size * 0.5)
            
            coords = [np.random.randint(0, i, num_salt) for i in noisy.shape]
            noisy[tuple(coords)] = 255
            
            coords = [np.random.randint(0, i, num_pepper) for i in noisy.shape]
            noisy[tuple(coords)] = 0
        
        self.original_image = noisy
        self.filtered_image = None
        self._update_displays()
        self.status_var.set("Добавлен шум 'соль и перец'")
    
    def _reset_image(self):
        """Сбросить изображение к оригиналу."""
        self.filtered_image = None
        self.current_filter = None
        self.filter_info_var.set("Фильтр не применён")
        self._update_displays()
        self.status_var.set("Изображение сброшено к оригиналу")
    
    # === ПРИМЕНЕНИЕ ФИЛЬТРОВ ===
    
    def _apply_filter(self, filter_type: str):
        """Применить выбранный фильтр."""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
        
        if self.processing:
            return
        
        try:
            if filter_type == 'box':
                size = self.box_size_var.get()
                self.current_filter = BoxBlurFilter(kernel_size=size)
            
            elif filter_type == 'gaussian':
                size = self.gauss_size_var.get()
                sigma = self.gauss_sigma_var.get()
                self.current_filter = GaussianBlurFilter(kernel_size=size, sigma=sigma)
            
            elif filter_type == 'median':
                size = self.median_size_var.get()
                self.current_filter = MedianFilter(kernel_size=size)
            
            elif filter_type == 'sobel':
                direction = self.sobel_direction_var.get()
                self.current_filter = SobelFilter(direction=direction)

            elif filter_type == 'unsharp':
                sigma = self.sharp_sigma_var.get()
                amount = self.sharp_amount_var.get()
                self.current_filter = UnsharpMaskFilter(kernel_size=5, sigma=sigma, amount=amount)

            elif filter_type == 'sketch':
                self.current_filter = SketchFilter(blur_sigma=5.0, blur_kernel=21)
            
            # Применяем фильтр в отдельном потоке
            self.processing = True
            self.status_var.set(f"Применение фильтра: {self.current_filter.name}...")
            self.filter_info_var.set(f"Применяется: {self.current_filter.name}")
            
            thread = threading.Thread(target=self._process_filter)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать фильтр:\n{e}")
            self.processing = False
    
    def _process_filter(self):
        """Обработать изображение фильтром (в отдельном потоке)."""
        try:
            # Применяем фильтр к изображению
            filtered = self.current_filter.apply(self.original_image)

            # Если есть маска, комбинируем результаты
            if self.mask is not None:
                if self.mask_mode_var.get() == 'selected':
                    # Применяем фильтр к выделенной области
                    result = self.original_image.copy()
                    if result.ndim == 3:
                        for c in range(result.shape[2]):
                            result[:, :, c][self.mask] = filtered[:, :, c][self.mask]
                    else:
                        result[self.mask] = filtered[self.mask]
                    self.filtered_image = result
                else:
                    # Применяем фильтр к невыделенной области
                    result = self.original_image.copy()
                    inverse_mask = ~self.mask
                    if result.ndim == 3:
                        for c in range(result.shape[2]):
                            result[:, :, c][inverse_mask] = filtered[:, :, c][inverse_mask]
                    else:
                        result[inverse_mask] = filtered[inverse_mask]
                    self.filtered_image = result
            else:
                # Применяем ко всему изображению
                self.filtered_image = filtered

            # Успешно завершено — планируем обновление UI
            self.root.after(0, self._on_filter_complete)

        except Exception as e:
            # Сохраняем сообщение об ошибке до вызова через after
            error_message = str(e)
            self.root.after(0, lambda: self._on_filter_error(error_message))

    def _on_filter_complete(self):
        """Callback при завершении фильтрации."""
        self.processing = False
        self._update_displays()
        
        mask_info = ""
        if self.mask is not None:
            mode = "выделенной" if self.mask_mode_var.get() == 'selected' else "невыделенной"
            mask_info = f" к {mode} области"
        
        self.filter_info_var.set(f"Применён: {self.current_filter.name}{mask_info}")
        self.status_var.set(f"Фильтр применён: {self.current_filter.name}")
    
    def _on_filter_error(self, error_msg: str):
        """Callback при ошибке фильтрации."""
        self.processing = False
        messagebox.showerror("Ошибка", f"Ошибка при применении фильтра:\n{error_msg}")
        self.status_var.set("Ошибка при применении фильтра")
    
    # === ОТОБРАЖЕНИЕ ===
    
    def _update_displays(self):
        """Обновить отображение изображений на canvas."""
        # Обновляем оригинал
        if self.original_image is not None:
            self._display_image(self.original_image, self.original_canvas, is_original=True)
        
        # Обновляем результат
        if self.filtered_image is not None:
            self._display_image(self.filtered_image, self.filtered_canvas, is_original=False)
        else:
            # Показываем оригинал и в правой части
            if self.original_image is not None:
                self._display_image(self.original_image, self.filtered_canvas, is_original=False)
    
    def _display_image(self, image: np.ndarray, canvas: tk.Canvas, is_original: bool = False):
        """
        Отобразить изображение на canvas с фиксированным масштабированием.
        
        Args:
            image: Изображение для отображения
            canvas: Canvas для отрисовки
            is_original: Если True, сохраняем параметры масштабирования
        """
        # Обновляем canvas для получения актуальных размеров
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Минимальные размеры
        if canvas_width < 100:
            canvas_width = 600
        if canvas_height < 100:
            canvas_height = 600
        
        # Создаём PIL изображение
        if image.ndim == 3:
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = Image.fromarray(image.astype(np.uint8), mode='L')
        
        # Вычисляем масштаб (один раз для оригинала, затем используем тот же)
        img_width, img_height = pil_image.size
        
        if is_original or self.display_size is None:
            # Вычисляем новый масштаб
            scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Сохраняем для последующего использования
            if is_original:
                self.display_scale = scale
                self.display_size = (new_width, new_height)
        else:
            # Используем сохранённые размеры
            new_width, new_height = self.display_size
        
        # Масштабируем изображение
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Конвертируем в PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Очищаем canvas
        canvas.delete("all")
        
        # Центрируем изображение
        x = canvas_width // 2
        y = canvas_height // 2
        canvas.create_image(x, y, image=photo, anchor='center')
        
        # Если это оригинал и есть выделение, перерисовываем прямоугольник
        if is_original and self.selection_start and self.selection_end and self.selection_mode:
            self.selection_rect_id = canvas.create_rectangle(
                self.selection_start[0], self.selection_start[1],
                self.selection_end[0], self.selection_end[1],
                outline='yellow', width=2, dash=(5, 5)
            )
        
        # Сохраняем ссылку на изображение
        canvas.image = photo


def main():
    """Главная функция запуска приложения."""
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()