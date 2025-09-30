import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft, ifft, fftfreq
from scipy import signal
import sounddevice as sd
import threading
import time
import sys
import warnings
from collections import deque

if sys.platform == "win32":
    import msvcrt
else:
    import termios
    import tty

warnings.filterwarnings('ignore')

class SignalGenerator:
    """Генератор сигналов из лабораторной работы №1"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
    
    def phase_array(self, frequency, phase_offset, n_array, N):
        return 2 * np.pi * frequency * n_array / N + phase_offset

    def sine_wave(self, amplitude, phi):
        return amplitude * np.sin(phi)

    def pulse_wave(self, amplitude, phi, duty_cycle):
        return np.where((np.mod(phi, 2 * np.pi) / (2 * np.pi)) <= duty_cycle, amplitude, -amplitude)

    def triangle_wave(self, amplitude, phi):
        return (2 * amplitude / np.pi) * (
            np.abs(np.mod(phi + 3 * np.pi / 2, 2 * np.pi) - np.pi) - (np.pi / 2)
        )

    def sawtooth_wave(self, amplitude, phi):
        return (amplitude / np.pi) * (np.mod(phi + np.pi, 2 * np.pi) - np.pi)

    def waveform(self, amplitude, phi, signal_type, duty_cycle=0.5):
        if signal_type == 'sine':
            return self.sine_wave(amplitude, phi)
        elif signal_type == 'pulse':
            return self.pulse_wave(amplitude, phi, duty_cycle)
        elif signal_type == 'triangle':
            return self.triangle_wave(amplitude, phi)
        elif signal_type == 'sawtooth':
            return self.sawtooth_wave(amplitude, phi)
        else:
            return np.zeros_like(phi)

class FourierAnalyzer:
    """Класс для анализа Фурье с собственными реализациями"""
    
    def __init__(self):
        pass
    
    def dft_direct(self, signal):
        """Прямое дискретное преобразование Фурье"""
        N = len(signal)
        spectrum = np.zeros(N, dtype=complex)
        
        for k in range(N):
            for n in range(N):
                spectrum[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
        
        return spectrum
    
    def idft_direct(self, spectrum):
        """Обратное дискретное преобразование Фурье"""
        N = len(spectrum)
        signal = np.zeros(N, dtype=complex)
        
        for n in range(N):
            for k in range(N):
                signal[n] += spectrum[k] * np.exp(2j * np.pi * k * n / N)
            signal[n] /= N
        
        return signal
    
    def fft_custom(self, signal):
        """Быстрое преобразование Фурье (БПФ) - алгоритм Кули-Тьюки"""
        N = len(signal)
        
        # Базовый случай
        if N <= 1:
            return signal
        
        # Проверяем, что N - степень 2
        if N & (N - 1) != 0:
            # Дополняем до ближайшей степени 2
            next_power_of_2 = 1 << (N - 1).bit_length()
            padded_signal = np.zeros(next_power_of_2, dtype=complex)
            padded_signal[:N] = signal
            result = self.fft_custom(padded_signal)
            return result[:N] if N < next_power_of_2 else result
        
        # Разделение на четные и нечетные элементы
        even = self.fft_custom(signal[0::2])
        odd = self.fft_custom(signal[1::2])
        
        # Объединение результатов
        T = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        spectrum = np.zeros(N, dtype=complex)
        
        for k in range(N // 2):
            t = T[k] * odd[k]
            spectrum[k] = even[k] + t
            spectrum[k + N // 2] = even[k] - t
        
        return spectrum
    
    def ifft_custom(self, spectrum):
        """Обратное быстрое преобразование Фурье"""
        # Используем свойство: IFFT(X) = conj(FFT(conj(X))) / N
        N = len(spectrum)
        conjugated = np.conj(spectrum)
        fft_result = self.fft_custom(conjugated)
        return np.conj(fft_result) / N

class DigitalFilters:
    """Класс для цифровой фильтрации"""
    
    def __init__(self):
        pass
    
    def design_lowpass_filter(self, cutoff_freq, sample_rate, order=5):
        """Проектирование НЧ-фильтра"""
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        return b, a
    
    def design_highpass_filter(self, cutoff_freq, sample_rate, order=5):
        """Проектирование ВЧ-фильтра"""
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        return b, a
    
    def design_bandpass_filter(self, low_freq, high_freq, sample_rate, order=5):
        """Проектирование полосового фильтра"""
        nyquist = sample_rate / 2
        low_normalized = low_freq / nyquist
        high_normalized = high_freq / nyquist
        b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
        return b, a
    
    def apply_filter(self, signal_data, b, a):
        """Применение фильтра к сигналу"""
        return signal.filtfilt(b, a, signal_data)

class KeyboardController:
    """Контроллер клавиатуры"""
    
    def __init__(self):
        if sys.platform != "win32":
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
    
    def __del__(self):
        if sys.platform != "win32":
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass
    
    def get_key(self):
        if sys.platform == "win32":
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\xe0':  
                    key = msvcrt.getch()
                    return None
                elif key == b'\x1b':  
                    return '\x1b'
                elif key == b'\x03':  
                    return '\x03'
                else:
                    return key.decode('utf-8', errors='ignore')
            return None
        else:
            # Unix-подобные системы
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None

class RealTimeFourierAnalyzer:
    """Интерактивный анализатор Фурье в реальном времени"""
    
    def __init__(self, sample_rate=44100, buffer_size=44100):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.dt = 1.0 / sample_rate
        self.t = 0.0    
        
        # Параметры сигнала
        self.carrier_frequency = 800.0
        self.carrier_amplitude = 0.8
        self.carrier_phase = 0.0
        self.carrier_duty_cycle = 0.5
        self.carrier_type = 'sine'
        
        self.mod_frequency = 5.0
        self.mod_amplitude = 0.5
        self.mod_phase = 0.0
        self.mod_duty_cycle = 0.5
        self.mod_type = 'sine'
        
        self.modulation_mode = 'AM'
        self.modulation_enabled = False
        
        # Параметры фильтра
        self.filter_enabled = False
        self.filter_type = 'lowpass'  # 'lowpass', 'highpass', 'bandpass'
        self.filter_cutoff = 1000.0
        self.filter_low = 500.0
        self.filter_high = 1500.0
        self.filter_order = 5
        
        # Объекты для работы
        self.generator = SignalGenerator(sample_rate)
        self.analyzer = FourierAnalyzer()
        self.filters = DigitalFilters()
        
        self.running = True
        
        # Буферы для данных
        self.buffer_size_vis = 1024
        self.time_buffer = deque(maxlen=self.buffer_size_vis)
        self.signal_buffer = deque(maxlen=self.buffer_size_vis)
        self.filtered_buffer = deque(maxlen=self.buffer_size_vis)
        self.spectrum_buffer = deque(maxlen=self.buffer_size_vis//2)
        
        # Аудио поток
        self.stream = sd.OutputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=sample_rate,
            blocksize=buffer_size
        )
        
        # Инициализация буферов
        for i in range(self.buffer_size_vis):
            self.time_buffer.append(i * self.dt)
            self.signal_buffer.append(0.0)
            self.filtered_buffer.append(0.0)
        
        for i in range(self.buffer_size_vis//2):
            self.spectrum_buffer.append(0.0)
    
    def audio_callback(self, outdata, frames, time, status):
        """Callback для аудио потока"""
        if status:
            print(f"Audio status: {status}")

        n_array = np.arange(frames)
        N = self.sample_rate

        # Модулирующий сигнал
        mod_phi = self.generator.phase_array(self.mod_frequency, self.mod_phase, n_array, N)
        mod_signal = self.generator.waveform(self.mod_amplitude, mod_phi, self.mod_type, self.mod_duty_cycle)

        # Несущий сигнал
        if self.modulation_enabled and self.modulation_mode == 'FM':
            freq_deviation = self.mod_frequency * 50
            instantaneous_freq = self.carrier_frequency + freq_deviation * mod_signal
            carrier_phi = np.mod(
                2 * np.pi * np.cumsum(instantaneous_freq) / N + self.carrier_phase,
                2 * np.pi
            )
        else:
            carrier_phi = self.generator.phase_array(self.carrier_frequency, self.carrier_phase, n_array, N)

        carrier_signal = self.generator.waveform(self.carrier_amplitude, carrier_phi, self.carrier_type, self.carrier_duty_cycle)

        # Модуляция
        if self.modulation_enabled and self.modulation_mode == 'AM':
            modulated_signal = carrier_signal * (1 + mod_signal)
        else:
            modulated_signal = carrier_signal
        
        # Фильтрация
        if self.filter_enabled:
            try:
                if self.filter_type == 'lowpass':
                    b, a = self.filters.design_lowpass_filter(self.filter_cutoff, self.sample_rate, self.filter_order)
                elif self.filter_type == 'highpass':
                    b, a = self.filters.design_highpass_filter(self.filter_cutoff, self.sample_rate, self.filter_order)
                elif self.filter_type == 'bandpass':
                    b, a = self.filters.design_bandpass_filter(self.filter_low, self.filter_high, self.sample_rate, self.filter_order)
                
                # Применяем фильтр только к новому сегменту
                filtered_signal = self.filters.apply_filter(modulated_signal, b, a)
            except:
                filtered_signal = modulated_signal
        else:
            filtered_signal = modulated_signal
        
        # Выводим отфильтрованный сигнал
        outdata[:, 0] = filtered_signal.astype(np.float32)
        
        # Обновляем буферы для визуализации
        for i in range(frames):
            self.time_buffer.append(self.t + i * self.dt)
            self.signal_buffer.append(modulated_signal[i])
            self.filtered_buffer.append(filtered_signal[i])
        
        # Спектральный анализ последнего буфера
        if len(self.filtered_buffer) >= self.buffer_size_vis:
            spectrum = fft(list(self.filtered_buffer))
            spectrum_magnitude = np.abs(spectrum[:len(spectrum)//2])
            
            # Обновляем спектральный буфер
            for i, mag in enumerate(spectrum_magnitude[:len(self.spectrum_buffer)]):
                if i < len(self.spectrum_buffer):
                    self.spectrum_buffer[i] = mag
        
        self.t += frames * self.dt
    
    def start(self):
        """Запуск аудио потока"""
        self.stream.start()
    
    def stop(self):
        """Остановка"""
        self.running = False
        self.stream.stop()
        self.stream.close()
    
    def get_data_for_plot(self):
        """Получить данные для построения графика"""
        return (np.array(self.time_buffer), 
                np.array(self.signal_buffer), 
                np.array(self.filtered_buffer),
                np.array(self.spectrum_buffer))
    
    def print_status(self):
        """Вывод текущего состояния"""
        mod_status = "ON" if self.modulation_enabled else "OFF"
        filter_status = "ON" if self.filter_enabled else "OFF"
        
        if self.filter_type == 'bandpass':
            filter_info = f"{self.filter_type} {self.filter_low:.0f}-{self.filter_high:.0f}Hz"
        else:
            filter_info = f"{self.filter_type} {self.filter_cutoff:.0f}Hz"
        
        print(f"\rМодуляция: {mod_status} ({self.modulation_mode}) | "
              f"Несущая: {self.carrier_type} f={self.carrier_frequency:.1f}Hz a={self.carrier_amplitude:.2f} | "
              f"Модулир.: {self.mod_type} f={self.mod_frequency:.1f}Hz a={self.mod_amplitude:.2f} | "
              f"Фильтр: {filter_status} ({filter_info})",
              end='', flush=True)

class RealTimeVisualizer:
    """Визуализатор в реальном времени как в музыкальных плеерах"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
        # Настройка matplotlib для интерактивности
        plt.ion()
        
        # Создание фигуры с тремя субплотами
        self.fig = plt.figure(figsize=(16, 10))
        
        # График временной области
        self.ax_time = plt.subplot(2, 2, 1)
        self.line_original, = self.ax_time.plot([], [], 'b-', linewidth=1, label='Оригинал', alpha=0.7)
        self.line_filtered, = self.ax_time.plot([], [], 'r-', linewidth=2, label='Фильтрованный')
        self.ax_time.set_xlim(0, 0.05)
        self.ax_time.set_ylim(-2, 2)
        self.ax_time.set_xlabel('Время (с)')
        self.ax_time.set_ylabel('Амплитуда')
        self.ax_time.set_title('Сигналы во временной области')
        self.ax_time.legend()
        self.ax_time.grid(True, alpha=0.3)
        
        # Спектрограмма (как в плеерах!)
        self.ax_spectrum = plt.subplot(2, 2, 2)
        self.spectrum_bars = None
        self.ax_spectrum.set_xlim(0, self.analyzer.sample_rate // 4)
        self.ax_spectrum.set_ylim(0, 1)
        self.ax_spectrum.set_xlabel('Частота (Гц)')
        self.ax_spectrum.set_ylabel('Амплитуда')
        self.ax_spectrum.set_title('Спектр в реальном времени 🎵')
        self.ax_spectrum.grid(True, alpha=0.3)
        
        # Частотная характеристика фильтра
        self.ax_filter = plt.subplot(2, 2, 3)
        self.filter_line, = self.ax_filter.plot([], [], 'g-', linewidth=2)
        self.ax_filter.set_xlim(0, self.analyzer.sample_rate // 4)
        self.ax_filter.set_ylim(-60, 5)
        self.ax_filter.set_xlabel('Частота (Гц)')
        self.ax_filter.set_ylabel('Амплитуда (дБ)')
        self.ax_filter.set_title('Частотная характеристика фильтра')
        self.ax_filter.grid(True, alpha=0.3)
        
        # 3D спектрограмма-водопад
        self.ax_waterfall = plt.subplot(2, 2, 4)
        self.waterfall_data = deque(maxlen=50)  # История спектров
        self.waterfall_im = None
        self.ax_waterfall.set_xlabel('Частота (Гц)')
        self.ax_waterfall.set_ylabel('Время')
        self.ax_waterfall.set_title('Спектрограмма-водопад')
        
        # Анимация
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot, interval=50, blit=False
        )
        
        # Установка заголовка окна
        try:
            self.fig.canvas.set_window_title('Real-time Fourier Analyzer 🎵')
        except:
            try:
                self.fig.canvas.manager.set_window_title('Real-time Fourier Analyzer 🎵')
            except:
                pass
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update_plot(self, frame):
        """Обновление всех графиков"""
        if not self.analyzer.running:
            return []
        
        # Получаем данные
        time_data, signal_data, filtered_data, spectrum_data = self.analyzer.get_data_for_plot()
        
        if len(time_data) > 0:
            # Ограничиваем окно отображения
            window_size = min(1000, len(time_data))
            time_window = time_data[-window_size:]
            signal_window = signal_data[-window_size:]
            filtered_window = filtered_data[-window_size:]
            
            # Нормализуем время
            if len(time_window) > 0:
                time_normalized = time_window - time_window[0]
                
                # Обновляем временные графики
                self.line_original.set_data(time_normalized, signal_window)
                self.line_filtered.set_data(time_normalized, filtered_window)
                
                if len(time_normalized) > 0:
                    self.ax_time.set_xlim(0, time_normalized[-1])
                    max_amp = max(2.0, np.max(np.abs(np.concatenate([signal_window, filtered_window]))) * 1.2)
                    self.ax_time.set_ylim(-max_amp, max_amp)
        
        # Обновляем спектр (столбики как в плеерах!)
        if len(spectrum_data) > 50:  # Достаточно данных для спектра
            # Частотная сетка
            freqs = np.fft.fftfreq(len(spectrum_data)*2, 1/self.analyzer.sample_rate)[:len(spectrum_data)]
            
            # Ограничиваем частотный диапазон для лучшего отображения
            max_freq_idx = min(len(spectrum_data), len(spectrum_data) // 4)
            freqs_display = freqs[:max_freq_idx]
            spectrum_display = spectrum_data[:max_freq_idx]
            
            # Нормализуем спектр
            if np.max(spectrum_display) > 0:
                spectrum_normalized = spectrum_display / np.max(spectrum_display)
            else:
                spectrum_normalized = spectrum_display
            
            # Создаем/обновляем столбики спектра
            self.ax_spectrum.clear()
            bars = self.ax_spectrum.bar(freqs_display, spectrum_normalized, 
                                      width=freqs_display[1]-freqs_display[0] if len(freqs_display) > 1 else 10,
                                      color='cyan', alpha=0.7, edgecolor='blue', linewidth=0.5)
            
            # Цветовая градация по высоте (как в плеерах)
            for bar, height in zip(bars, spectrum_normalized):
                if height > 0.7:
                    bar.set_color('red')
                elif height > 0.4:
                    bar.set_color('orange')
                elif height > 0.2:
                    bar.set_color('yellow')
                else:
                    bar.set_color('cyan')
            
            self.ax_spectrum.set_xlim(0, max(freqs_display) if len(freqs_display) > 0 else 1000)
            self.ax_spectrum.set_ylim(0, 1.1)
            self.ax_spectrum.set_xlabel('Частота (Гц)')
            self.ax_spectrum.set_ylabel('Амплитуда')
            self.ax_spectrum.set_title('Спектр в реальном времени 🎵')
            self.ax_spectrum.grid(True, alpha=0.3)
            
            # Добавляем текущий спектр в историю для водопада
            self.waterfall_data.append(spectrum_normalized)
        
        # Обновляем частотную характеристику фильтра
        if self.analyzer.filter_enabled:
            try:
                if self.analyzer.filter_type == 'lowpass':
                    b, a = self.analyzer.filters.design_lowpass_filter(
                        self.analyzer.filter_cutoff, self.analyzer.sample_rate, self.analyzer.filter_order)
                elif self.analyzer.filter_type == 'highpass':
                    b, a = self.analyzer.filters.design_highpass_filter(
                        self.analyzer.filter_cutoff, self.analyzer.sample_rate, self.analyzer.filter_order)
                elif self.analyzer.filter_type == 'bandpass':
                    b, a = self.analyzer.filters.design_bandpass_filter(
                        self.analyzer.filter_low, self.analyzer.filter_high, 
                        self.analyzer.sample_rate, self.analyzer.filter_order)
                
                w, h = signal.freqz(b, a, worN=8000)
                freqs_filter = w * self.analyzer.sample_rate / (2 * np.pi)
                magnitude_db = 20 * np.log10(np.abs(h))
                
                self.filter_line.set_data(freqs_filter, magnitude_db)
                self.ax_filter.set_xlim(0, min(3000, max(freqs_filter)))
                
            except Exception as e:
                pass
        else:
            self.filter_line.set_data([], [])
        
        # Обновляем водопад
        if len(self.waterfall_data) > 1:
            waterfall_array = np.array(list(self.waterfall_data))
            
            self.ax_waterfall.clear()
            im = self.ax_waterfall.imshow(waterfall_array, aspect='auto', cmap='viridis', 
                                        origin='lower', interpolation='bilinear')
            self.ax_waterfall.set_xlabel('Частота (бины)')
            self.ax_waterfall.set_ylabel('Время (кадры)')
            self.ax_waterfall.set_title('Спектрограмма-водопад')
        
        return []
    
    def close(self):
        """Закрытие визуализатора"""
        plt.close(self.fig)

def print_help():
    """Справка по управлению"""
    print("=" * 100)
    print("🎵 REAL-TIME FOURIER ANALYZER С ФИЛЬТРАЦИЕЙ 🎵")
    print("=" * 100)
    print("СИГНАЛ:")
    print("  1-4    - Тип несущего (синус/импульс/треугольный/пилообразный)")
    print("  q/a    - Частота несущего +/- (10 Гц)")
    print("  w/s    - Амплитуда несущего +/- (0.05)")
    print("  e/d    - Фаза несущего +/- (π/8)")
    print("  r/f    - Скважность несущего +/- (0.1)")
    print()
    print("МОДУЛЯЦИЯ:")
    print("  m      - Переключение AM/FM")
    print("  n      - Включить/выключить модуляцию")
    print("  5-8    - Тип модулирующего (синус/импульс/треугольный/пилообразный)")
    print("  t/g    - Частота модулирующего +/- (1 Гц)")
    print("  y/h    - Глубина модуляции +/- (0.05)")
    print()
    print("ФИЛЬТРАЦИЯ:")
    print("  SPACE  - Включить/выключить фильтр")
    print("  z/x    - Тип фильтра (НЧ/ВЧ/полосовой)")
    print("  u/j    - Частота среза +/- (50 Гц)")
    print("  i/k    - Нижняя частота полосы +/- (50 Гц) [полосовой]")
    print("  o/l    - Верхняя частота полосы +/- (50 Гц) [полосовой]")
    print("  p/;    - Порядок фильтра +/- (1)")
    print()
    print("  ?      - Эта справка")
    print("  ESC    - Выход")
    print("=" * 100)

def main():
    """Основная функция"""
    print_help()
    
    analyzer = RealTimeFourierAnalyzer(sample_rate=44100, buffer_size=44100)
    keyboard = KeyboardController()
    
    try:
        analyzer.start()
        visualizer = RealTimeVisualizer(analyzer)
        
        print("\n🎵 Real-time Fourier Analyzer запущен!")
        print("Используйте клавиши для управления. Нажмите '?' для справки.")
        
        while analyzer.running:
            key = keyboard.get_key()
            
            if key:
                # Управление сигналом
                if key == '1':
                    analyzer.carrier_type = 'sine'
                elif key == '2':
                    analyzer.carrier_type = 'pulse'
                elif key == '3':
                    analyzer.carrier_type = 'triangle'
                elif key == '4':
                    analyzer.carrier_type = 'sawtooth'
                elif key == '5':
                    analyzer.mod_type = 'sine'
                elif key == '6':
                    analyzer.mod_type = 'pulse'
                elif key == '7':
                    analyzer.mod_type = 'triangle'
                elif key == '8':
                    analyzer.mod_type = 'sawtooth'
                
                # Частота несущего
                elif key == 'q':
                    analyzer.carrier_frequency = min(analyzer.carrier_frequency + 10, 5000)
                elif key == 'a':
                    analyzer.carrier_frequency = max(analyzer.carrier_frequency - 10, 1)
                
                # Амплитуда несущего
                elif key == 'w':
                    analyzer.carrier_amplitude = min(analyzer.carrier_amplitude + 0.05, 1.0)
                elif key == 's':
                    analyzer.carrier_amplitude = max(analyzer.carrier_amplitude - 0.05, 0.0)
                
                # Фаза несущего
                elif key == 'e':
                    analyzer.carrier_phase += np.pi / 8
                    if analyzer.carrier_phase > 2 * np.pi:
                        analyzer.carrier_phase -= 2 * np.pi
                elif key == 'd':
                    analyzer.carrier_phase -= np.pi / 8
                    if analyzer.carrier_phase < 0:
                        analyzer.carrier_phase += 2 * np.pi
                
                # Скважность несущего
                elif key == 'r':
                    analyzer.carrier_duty_cycle = min(analyzer.carrier_duty_cycle + 0.1, 1)
                elif key == 'f':
                    analyzer.carrier_duty_cycle = max(analyzer.carrier_duty_cycle - 0.1, 0)
                
                # Модуляция
                elif key == 'm':
                    analyzer.modulation_mode = 'FM' if analyzer.modulation_mode == 'AM' else 'AM'
                elif key == 'n':
                    analyzer.modulation_enabled = not analyzer.modulation_enabled
                
                # Частота модулирующего
                elif key == 't':
                    analyzer.mod_frequency = min(analyzer.mod_frequency + 1, 100)
                elif key == 'g':
                    analyzer.mod_frequency = max(analyzer.mod_frequency - 1, 0.1)
                
                # Амплитуда модулирующего
                elif key == 'y':
                    analyzer.mod_amplitude = min(analyzer.mod_amplitude + 0.05, 1.0)
                elif key == 'h':
                    analyzer.mod_amplitude = max(analyzer.mod_amplitude - 0.05, 0.0)
                
                # Фильтрация
                elif key == ' ':  # Пробел
                    analyzer.filter_enabled = not analyzer.filter_enabled
                elif key == 'z':
                    filter_types = ['lowpass', 'highpass', 'bandpass']
                    current_idx = filter_types.index(analyzer.filter_type)
                    analyzer.filter_type = filter_types[(current_idx + 1) % len(filter_types)]
                elif key == 'x':
                    filter_types = ['lowpass', 'highpass', 'bandpass']
                    current_idx = filter_types.index(analyzer.filter_type)
                    analyzer.filter_type = filter_types[(current_idx - 1) % len(filter_types)]
                
                # Частота среза
                elif key == 'u':
                    analyzer.filter_cutoff = min(analyzer.filter_cutoff + 50, analyzer.sample_rate // 2 - 100)
                elif key == 'j':
                    analyzer.filter_cutoff = max(analyzer.filter_cutoff - 50, 50)
                
                # Полосовой фильтр - нижняя частота
                elif key == 'i':
                    analyzer.filter_low = min(analyzer.filter_low + 50, analyzer.filter_high - 100)
                elif key == 'k':
                    analyzer.filter_low = max(analyzer.filter_low - 50, 50)
                
                # Полосовой фильтр - верхняя частота  
                elif key == 'o':
                    analyzer.filter_high = min(analyzer.filter_high + 50, analyzer.sample_rate // 2 - 100)
                elif key == 'l':
                    analyzer.filter_high = max(analyzer.filter_high - 50, analyzer.filter_low + 100)
                
                # Порядок фильтра
                elif key == 'p':
                    analyzer.filter_order = min(analyzer.filter_order + 1, 10)
                elif key == ';':
                    analyzer.filter_order = max(analyzer.filter_order - 1, 1)
                
                # Справка и выход
                elif key == '?':
                    print()
                    print_help()
                elif key == '\x1b' or key == '\x03':  # ESC или Ctrl+C
                    break
            
            # Вывод статуса
            analyzer.print_status()
            
            # Пауза для обновления
            plt.pause(0.01)
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\n🎵 Остановка Real-time Fourier Analyzer...")
        analyzer.stop()
        if 'visualizer' in locals():
            visualizer.close()
        print("Анализатор остановлен.")

if __name__ == "__main__":
    main()