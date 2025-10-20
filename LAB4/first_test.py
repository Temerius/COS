import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import random

class CorrelationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Корреляция изображений")
        self.root.geometry("1200x800")

        self.original_image = None
        self.original_cv = None
        self.fragment_cv = None
        self.correlation_result = None

        # Верхнее меню
        menu = tk.Menu(root)
        root.config(menu=menu)
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Загрузить изображение", command=self.load_image)

        # Кнопки действий
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.btn_cross_corr = tk.Button(btn_frame, text="Взаимная корреляция", state=tk.DISABLED, command=self.cross_correlation)
        self.btn_auto_corr = tk.Button(btn_frame, text="Автокорреляция", state=tk.DISABLED, command=self.auto_correlation)
        self.btn_cross_corr.pack(side=tk.LEFT, padx=5)
        self.btn_auto_corr.pack(side=tk.LEFT, padx=5)

        # Canvas для отображения изображений
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas_original = tk.Canvas(self.canvas_frame, bg='lightgray', width=400, height=300)
        self.canvas_fragment = tk.Canvas(self.canvas_frame, bg='lightgray', width=200, height=150)
        self.canvas_corr = tk.Canvas(self.canvas_frame, bg='lightgray', width=400, height=300)

        self.canvas_original.grid(row=0, column=0, padx=5, pady=5)
        self.canvas_fragment.grid(row=0, column=1, padx=5, pady=5)
        self.canvas_corr.grid(row=0, column=2, padx=5, pady=5)

        self.label_original = tk.Label(self.canvas_frame, text="Исходное изображение")
        self.label_fragment = tk.Label(self.canvas_frame, text="Фрагмент")
        self.label_corr = tk.Label(self.canvas_frame, text="Корреляция")

        self.label_original.grid(row=1, column=0)
        self.label_fragment.grid(row=1, column=1)
        self.label_corr.grid(row=1, column=2)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return

        self.original_cv = cv2.imread(file_path)
        if self.original_cv is None:
            messagebox.showerror("Ошибка", "Не удалось загрузить изображение.")
            return

        # Сохраняем копию для отрисовки с выделением
        self.original_image = self.original_cv.copy()

        self.display_image(self.original_cv, self.canvas_original)
        self.btn_cross_corr.config(state=tk.NORMAL)
        self.btn_auto_corr.config(state=tk.NORMAL)

    def display_image(self, img_cv, canvas):
        # Конвертируем OpenCV BGR -> RGB -> PIL -> PhotoImage
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Масштабируем под размер canvas (сохраняя пропорции)
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 400, 300

        img_pil.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        canvas.delete("all")
        canvas.image = img_tk  # Сохраняем ссылку
        canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=img_tk)

    def cross_correlation(self):
        if self.original_cv is None:
            return

        h, w = self.original_cv.shape[:2]
        # Выбираем случайный фрагмент 1/4 размера
        frag_h, frag_w = max(30, h // 4), max(30, w // 4)
        y = random.randint(0, h - frag_h)
        x = random.randint(0, w - frag_w)

        self.fragment_cv = self.original_cv[y:y+frag_h, x:x+frag_w].copy()

        # Вычисляем взаимную корреляцию
        method = cv2.TM_CCOEFF_NORMED
        result = cv2.matchTemplate(self.original_cv, self.fragment_cv, method)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # Рисуем прямоугольник на копии оригинала
        matched_img = self.original_image.copy()
        h_f, w_f = self.fragment_cv.shape[:2]
        top_left = max_loc
        bottom_right = (top_left[0] + w_f, top_left[1] + h_f)
        cv2.rectangle(matched_img, top_left, bottom_right, (0, 255, 0), 2)

        # Отображаем
        self.display_image(matched_img, self.canvas_original)
        self.display_image(self.fragment_cv, self.canvas_fragment)

        # Нормализуем корреляционную карту для отображения
        corr_norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        corr_colored = cv2.applyColorMap(corr_norm, cv2.COLORMAP_JET)
        self.display_image(corr_colored, self.canvas_corr)

    def auto_correlation(self):
        if self.original_cv is None:
            return

        img_gray = cv2.cvtColor(self.original_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Вычисляем автокорреляцию через FFT
        f = np.fft.fft2(img_gray)
        autocorr = np.fft.ifft2(f * np.conj(f)).real
        autocorr = np.fft.fftshift(autocorr)  # Центрируем пик

        # Нормализуем для отображения
        autocorr_norm = cv2.normalize(autocorr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        autocorr_colored = cv2.applyColorMap(autocorr_norm, cv2.COLORMAP_JET)

        self.display_image(self.original_cv, self.canvas_original)
        self.canvas_fragment.delete("all")
        self.canvas_fragment.create_text(100, 75, text="—", font=("Arial", 24))
        self.display_image(autocorr_colored, self.canvas_corr)


if __name__ == "__main__":
    root = tk.Tk()
    app = CorrelationApp(root)
    root.mainloop()