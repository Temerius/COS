import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import math

class CustomImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Редактор с наложением фильтров v3.0")

        # ... (GUI-часть остается без изменений, поэтому я ее сверну для краткости) ...
        # Переменные для хранения изображений
        self.original_image = None
        self.processed_image = None
        self.is_image_loaded = False
        self.filter_widgets = []

        # --- Структура GUI ---
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Фрейм для изображений
        image_frame = tk.Frame(main_frame)
        image_frame.pack(pady=5, fill=tk.X, expand=True)
        self.original_label = tk.Label(image_frame, text="Оригинальное изображение")
        self.original_label.pack(side=tk.LEFT, padx=10, expand=True)
        self.processed_label = tk.Label(image_frame, text="Обработанное изображение")
        self.processed_label.pack(side=tk.LEFT, padx=10, expand=True)

        # Фрейм для кнопок управления
        file_controls_frame = tk.Frame(main_frame)
        file_controls_frame.pack(pady=10, fill=tk.X)
        self.load_button = tk.Button(file_controls_frame, text="Загрузить", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(file_controls_frame, text="Сохранить", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.reset_button = tk.Button(file_controls_frame, text="Сбросить", command=self.reset_image, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        # --- Интерфейс с сеткой 2x2 ---
        filters_container = tk.Frame(main_frame, relief=tk.RIDGE, borderwidth=2)
        filters_container.pack(pady=10, fill=tk.BOTH, expand=True)

        filters_container.grid_columnconfigure(0, weight=1)
        filters_container.grid_columnconfigure(1, weight=1)

        # --- Фильтр 1: Коробочное размытие (Box Blur) ---
        box_blur_frame = tk.LabelFrame(filters_container, text="Коробочное размытие")
        box_blur_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.box_blur_slider = tk.Scale(box_blur_frame, from_=3, to=51, orient=tk.HORIZONTAL, label="Размер ядра (нечетные)")
        self.box_blur_slider.config(command=self._make_odd_callback(self.box_blur_slider))
        self.box_blur_slider.pack(fill=tk.X, padx=5, expand=True)
        self.box_blur_button = tk.Button(box_blur_frame, text="Применить", command=self.apply_box_blur)
        self.box_blur_button.pack(pady=5)
        self.filter_widgets.extend([self.box_blur_slider, self.box_blur_button])

        # --- Фильтр 2: Размытие по Гауссу (Gaussian Blur) ---
        gaussian_blur_frame = tk.LabelFrame(filters_container, text="Размытие по Гауссу")
        gaussian_blur_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.gaussian_size_slider = tk.Scale(gaussian_blur_frame, from_=3, to=51, orient=tk.HORIZONTAL, label="Размер ядра (нечетные)")
        self.gaussian_size_slider.config(command=self._make_odd_callback(self.gaussian_size_slider))
        self.gaussian_size_slider.pack(fill=tk.X, padx=5, expand=True)
        self.gaussian_sigma_slider = tk.Scale(gaussian_blur_frame, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, label="Сигма")
        self.gaussian_sigma_slider.set(1.0)
        self.gaussian_sigma_slider.pack(fill=tk.X, padx=5, expand=True)
        self.gaussian_button = tk.Button(gaussian_blur_frame, text="Применить", command=self.apply_gaussian_blur)
        self.gaussian_button.pack(pady=5)
        self.filter_widgets.extend([self.gaussian_size_slider, self.gaussian_sigma_slider, self.gaussian_button])

        # --- Фильтр 3: Медианный фильтр ---
        median_filter_frame = tk.LabelFrame(filters_container, text="Медианный фильтр")
        median_filter_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.median_slider = tk.Scale(median_filter_frame, from_=3, to=51, orient=tk.HORIZONTAL, label="Размер (нечетные)")
        self.median_slider.config(command=self._make_odd_callback(self.median_slider))
        self.median_slider.pack(fill=tk.X, padx=5, expand=True)
        self.median_button = tk.Button(median_filter_frame, text="Применить", command=self.apply_median_filter)
        self.median_button.pack(pady=5)
        self.filter_widgets.extend([self.median_slider, self.median_button])

        # --- Фильтр 4: Оператор Собеля ---
        sobel_frame = tk.LabelFrame(filters_container, text="Оператор Собеля")
        sobel_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.sobel_button = tk.Button(sobel_frame, text="Применить", command=self.apply_sobel_operator)
        self.sobel_button.pack(pady=15, expand=True)
        self.filter_widgets.append(self.sobel_button)

        self._set_filter_widgets_state(tk.DISABLED)

    def _set_filter_widgets_state(self, state):
        for widget in self.filter_widgets:
            widget.config(state=state)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not file_path: return
        try:
            self.original_image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            self.processed_image = self.original_image.copy()
            self.is_image_loaded = True

            self.display_image(self.original_image, self.original_label)
            self.display_image(self.processed_image, self.processed_label)

            self.save_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            self._set_filter_widgets_state(tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")

    def save_image(self):
        if not self.is_image_loaded: return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not file_path: return
        try:
            image_to_save = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, image_to_save)
            messagebox.showinfo("Успех", "Изображение сохранено.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить изображение: {e}")

    def reset_image(self):
        if not self.is_image_loaded: return
        self.processed_image = self.original_image.copy()
        self.display_image(self.processed_image, self.processed_label)

    def display_image(self, cv2_image, label_widget):
        h, w = cv2_image.shape[:2]
        max_h, max_w = 400, 400
        scale = min(max_w/w, max_h/h)
        w_new, h_new = int(w*scale), int(h*scale)

        resized_img = cv2.resize(cv2_image, (w_new, h_new), interpolation=cv2.INTER_AREA)

        image = Image.fromarray(resized_img)
        photo = ImageTk.PhotoImage(image)
        label_widget.config(image=photo, text="")
        label_widget.image = photo

    def _make_odd_callback(self, slider):
        def callback(value):
            value = int(value)
            if value % 2 == 0:
                slider.set(value + 1)
        return callback

    def _convolve2d(self, image, kernel):
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        image_float = image.astype(np.float64)

        if image.ndim == 3:
            img_h, img_w, channels = image.shape
            output = np.zeros_like(image_float)
            for c in range(channels):
                padded = np.pad(image_float[:, :, c], ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
                shape = (img_h, img_w, k_h, k_w)
                strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
                windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
                output[:, :, c] = np.einsum('ij,xyij->xy', kernel, windows)
        else: # Grayscale
            img_h, img_w = image.shape
            padded = np.pad(image_float, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            shape = (img_h, img_w, k_h, k_w)
            strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
            windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
            output = np.einsum('ij,xyij->xy', kernel, windows)
        
        # Клиппинг и преобразование типов здесь могут быть не нужны, если мы обрабатываем градиенты
        return output


    def _median_filter(self, image, k_size):
        pad = k_size // 2
        output = np.zeros_like(image)
        if image.ndim == 3:
            img_h, img_w, channels = image.shape
            padded_img = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
            for c in range(channels):
                channel = padded_img[:, :, c]
                shape = (img_h, img_w, k_size, k_size)
                strides = (channel.strides[0], channel.strides[1], channel.strides[0], channel.strides[1])
                windows = np.lib.stride_tricks.as_strided(channel, shape=shape, strides=strides)
                output[:, :, c] = np.median(windows, axis=(2, 3))
        else:
            img_h, img_w = image.shape
            padded_img = np.pad(image, pad, mode='edge')
            shape = (img_h, img_w, k_size, k_size)
            strides = (padded_img.strides[0], padded_img.strides[1], padded_img.strides[0], padded_img.strides[1])
            windows = np.lib.stride_tricks.as_strided(padded_img, shape=shape, strides=strides)
            output = np.median(windows, axis=(2, 3))
        return output.astype(np.uint8)

    def _create_gaussian_kernel(self, size, sigma):
        center = size // 2
        x, y = np.mgrid[-center:center+1, -center:center+1]
        g = (1 / (2 * math.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g / g.sum()

    def apply_box_blur(self):
        if not self.is_image_loaded: return
        k_size = self.box_blur_slider.get()
        kernel = np.ones((k_size, k_size)) / (k_size * k_size)
        # Применяем свертку и сразу нормализуем результат в диапазон 0-255
        blurred_image = self._convolve2d(self.processed_image, kernel)
        self.processed_image = np.clip(blurred_image, 0, 255).astype(np.uint8)
        self.display_image(self.processed_image, self.processed_label)


    def apply_gaussian_blur(self):
        if not self.is_image_loaded: return
        k_size = self.gaussian_size_slider.get()
        sigma = self.gaussian_sigma_slider.get()
        kernel = self._create_gaussian_kernel(k_size, sigma)
        # Применяем свертку и сразу нормализуем результат
        blurred_image = self._convolve2d(self.processed_image, kernel)
        self.processed_image = np.clip(blurred_image, 0, 255).astype(np.uint8)
        self.display_image(self.processed_image, self.processed_label)


    def apply_median_filter(self):
        if not self.is_image_loaded: return
        k_size = self.median_slider.get()
        self.processed_image = self._median_filter(self.processed_image, k_size)
        self.display_image(self.processed_image, self.processed_label)
    
   
    def apply_sobel_operator(self):
        if not self.is_image_loaded: return
        
        # Ядра Собеля
        sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        # Создаем пустое изображение для результата
        h, w, _ = self.processed_image.shape
        gradient_magnitude = np.zeros((h, w, 3), dtype=np.float64)

       
        for c in range(3):
            channel = self.processed_image[:, :, c]
            grad_x = self._convolve2d(channel, sobel_x_kernel)
            grad_y = self._convolve2d(channel, sobel_y_kernel)
            
            
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
           
            max_val = np.max(magnitude)
            if max_val > 0:
                magnitude = (magnitude / max_val) * 255
            
            gradient_magnitude[:, :, c] = magnitude

        self.processed_image = gradient_magnitude.astype(np.uint8)
        self.display_image(self.processed_image, self.processed_label)


if __name__ == "__main__":
    root = tk.Tk()
    app = CustomImageFilterApp(root)
    root.mainloop()