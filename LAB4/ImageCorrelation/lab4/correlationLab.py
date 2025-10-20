import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2 # Используем библиотеку OpenCV для быстрых вычислений
import random

# =============================================================================
# ВЫЧИСЛИТЕЛЬНЫЕ ФУНКЦИИ НА ОСНОВЕ OPENCV И NUMPY.FFT
# =============================================================================

def calculate_mutual_correlation_cv(image_np, template_np):
    """
    Вычисляет быструю и точную взаимную корреляцию с помощью OpenCV.
    - image_np: Основное изображение в формате NumPy array (BGR).
    - template_np: Шаблон в формате NumPy array (BGR).
    """
    # Конвертируем в градации серого для корреляции
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template_np, cv2.COLOR_BGR2GRAY)
    
    # Используем метод нормированной кросс-корреляции из OpenCV
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    
    # Находим точку с максимальным совпадением
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    # Нормализуем карту корреляции для красивого отображения
    cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX)
    
    return result.astype(np.uint8), max_loc

def calculate_autocorrelation_fft(image_np):
    """
    Вычисляет автокорреляцию через БПФ из библиотеки NumPy.
    - image_np: Изображение в формате NumPy array (BGR).
    """
    # Конвертируем в градации серого и тип float32
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Вычитаем среднее значение для корректности
    gray -= np.mean(gray)

    # Вычисление автокорреляции через БПФ по теореме Винера-Хинчина
    f = np.fft.fft2(gray)
    power_spectrum = np.abs(f)**2
    acf = np.fft.ifft2(power_spectrum).real
    
    # Сдвигаем пик в центр для визуализации
    acf_shifted = np.fft.fftshift(acf)

    # Нормализация результата для отображения от 0 до 255
    min_val, max_val = np.min(acf_shifted), np.max(acf_shifted)
    if max_val == min_val:
        return np.zeros_like(acf_shifted, dtype=np.uint8)
        
    acf_norm = (acf_shifted - min_val) / (max_val - min_val) * 255
    return acf_norm.astype(np.uint8)

# =============================================================================
# КЛАСС ГРАФИЧЕСКОГО ИНТЕРФЕЙСА (функционал тот же)
# =============================================================================
class CorrelationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Вычисление корреляции изображений (OpenCV + NumPy)")
        self.root.geometry("1400x800")

        self.original_image_pil = None
        self.template_image_pil = None

        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", padding=5)
        style.configure("TFrame", background="#f0f0f0")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(fill=tk.X, side=tk.TOP)

        self.load_image_btn = ttk.Button(control_frame, text="1. Загрузить изображение", command=self.load_image)
        self.load_image_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.load_template_btn = ttk.Button(control_frame, text="2а. Загрузить шаблон", command=self.load_template)
        self.load_template_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.random_template_btn = ttk.Button(control_frame, text="2б. Случайный фрагмент", command=self.create_random_template)
        self.random_template_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.run_cross_corr_btn = ttk.Button(control_frame, text="3. Найти фрагмент", command=self.run_cross_correlation)
        self.run_cross_corr_btn.pack(side=tk.LEFT, padx=15, pady=5)
        
        self.run_auto_corr_btn = ttk.Button(control_frame, text="Найти повторы", command=self.run_autocorrelation)
        self.run_auto_corr_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.images_frame = ttk.Frame(main_frame)
        self.images_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.image_panel = self.create_image_panel("Исходное изображение", self.images_frame)
        self.template_panel = self.create_image_panel("Шаблон", self.images_frame)
        self.corr_map_panel = self.create_image_panel("Карта корреляции", self.images_frame)

    def create_image_panel(self, text, parent):
        frame = ttk.LabelFrame(parent, text=text, relief=tk.RIDGE, borderwidth=2)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        label = ttk.Label(frame)
        label.pack(fill=tk.BOTH, expand=True)
        return label

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.original_image_pil = Image.open(path).convert("RGB")
        self.display_image(self.original_image_pil, self.image_panel)

    def load_template(self):
        if not self.original_image_pil:
            messagebox.showerror("Ошибка", "Сначала загрузите основное изображение.")
            return
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.template_image_pil = Image.open(path).convert("RGB")
        self.display_image(self.template_image_pil, self.template_panel)

    def create_random_template(self):
        if not self.original_image_pil:
            messagebox.showerror("Ошибка", "Сначала загрузите основное изображение.")
            return
        w, h = self.original_image_pil.size
        tw, th = random.randint(w // 8, w // 4), random.randint(h // 8, h // 4)
        x, y = random.randint(0, w - tw), random.randint(0, h - th)
        self.template_image_pil = self.original_image_pil.crop((x, y, x + tw, y + th))
        self.display_image(self.template_image_pil, self.template_panel)

    def run_cross_correlation(self):
        if not self.original_image_pil or not self.template_image_pil:
            messagebox.showerror("Ошибка", "Необходимо загрузить основное изображение и шаблон.")
            return
        
        self.root.config(cursor="watch"); self.root.update_idletasks()

        # Конвертация из PIL в формат OpenCV (NumPy array, BGR)
        image_cv = cv2.cvtColor(np.array(self.original_image_pil), cv2.COLOR_RGB2BGR)
        template_cv = cv2.cvtColor(np.array(self.template_image_pil), cv2.COLOR_RGB2BGR)
        
        # Вычисление с помощью функции-обертки над OpenCV
        correlation_map_np, max_loc = calculate_mutual_correlation_cv(image_cv, template_cv)
        
        # Отображение карты корреляции
        corr_map_pil = Image.fromarray(correlation_map_np)
        self.display_image(corr_map_pil, self.corr_map_panel)

        # Рисование прямоугольника на исходном изображении
        result_image_pil = self.original_image_pil.copy()
        draw = ImageDraw.Draw(result_image_pil)
        
        tx, ty = max_loc
        th, tw = template_cv.shape[:2]
        
        draw.rectangle([tx, ty, tx + tw, ty + th], outline="lime", width=3)
        self.display_image(result_image_pil, self.image_panel)
        
        self.root.config(cursor="")
        messagebox.showinfo("Завершено", f"Найден фрагмент. Максимум корреляции в точке: ({tx}, {ty})")

    def run_autocorrelation(self):
        if not self.original_image_pil:
            messagebox.showerror("Ошибка", "Сначала загрузите основное изображение.")
            return
        
        self.root.config(cursor="watch"); self.root.update_idletasks()

        image_cv = cv2.cvtColor(np.array(self.original_image_pil), cv2.COLOR_RGB2BGR)
        
        # Вычисление с помощью функции-обертки над NumPy FFT
        autocorr_map_np = calculate_autocorrelation_fft(image_cv)

        autocorr_map_pil = Image.fromarray(autocorr_map_np)
        self.display_image(autocorr_map_pil, self.corr_map_panel)
        
        self.template_panel.configure(image=None); self.template_panel.image = None
        self.display_image(self.original_image_pil, self.image_panel)
        
        self.root.config(cursor="")
        messagebox.showinfo("Завершено", "Автокорреляционная функция рассчитана.")

    def display_image(self, img, panel):
        panel_w, panel_h = panel.winfo_width(), panel.winfo_height()
        if panel_w < 2 or panel_h < 2:
            self.root.after(50, lambda: self.display_image(img, panel))
            return
        img_copy = img.copy()
        img_copy.thumbnail((panel_w - 10, panel_h - 10), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_copy)
        panel.config(image=photo)
        panel.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = CorrelationApp(root)
    root.mainloop()