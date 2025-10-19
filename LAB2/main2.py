def demonstrate_fourier_analysis():
    """Демонстрация анализа Фурье различных сигналов"""
    
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА: АНАЛИЗ ФУРЬЕ СИГНАЛОВ")
    print("=" * 80)
    
    generator = SignalGenerator(sample_rate=8000) 
    analyzer = FourierAnalyzer()

    print("\n1. Анализ простых сигналов")
    print("-" * 40)
    
    duration = 0.5  
   
    print("Анализ синусоидального сигнала (200 Гц)...")
    t, sine_signal = generator.generate_signal(duration, 200, signal_type='sine')
    

    t_short = t[:500]  
    sine_short = sine_signal[:500]
    
    print("Прямое ДПФ:")
    result_dft = analyzer.analyze_spectrum(t_short, sine_short, method='dft', 
                                         title="Синусоидальный сигнал (200 Гц) - Прямое ДПФ")
    
    print("БПФ")
    result_scipy_fft = analyzer.analyze_spectrum(t, sine_signal, method='fft', 
                                               title="Синусоидальный сигнал (200 Гц) - БПФ")
    
   
    print(f"Максимальная ошибка БПФ: {np.max(np.abs(result_scipy_fft['error'])):.2e}")
    
    print("\nАнализ импульсного сигнала (150 Гц)...")
    t, pulse_signal = generator.generate_signal(duration, 150, signal_type='pulse', duty_cycle=0.3)
    analyzer.analyze_spectrum(t, pulse_signal, method='fft', 
                            title="Импульсный сигнал (150 Гц, скважность 0.3)")
    
    print("\nАнализ треугольного сигнала (100 Гц)...")
    t, triangle_signal = generator.generate_signal(duration, 100, signal_type='triangle')
    analyzer.analyze_spectrum(t, triangle_signal, method='fft', 
                            title="Треугольный сигнал (100 Гц)")
    
    # Пилообразный сигнал
    print("\nАнализ пилообразного сигнала (120 Гц)...")
    t, sawtooth_signal = generator.generate_signal(duration, 120, signal_type='sawtooth')
    analyzer.analyze_spectrum(t, sawtooth_signal, method='fft', 
                            title="Пилообразный сигнал (120 Гц)")
    
    # 2. Анализ модулированных сигналов
    print("\n2. Анализ модулированных сигналов")
    print("-" * 40)
    
    # AM модуляция
    print("AM модуляция (несущая 500 Гц, модулирующая 50 Гц)...")
    t, am_signal, carrier, mod = generator.generate_modulated_signal(
        duration, carrier_freq=500, mod_freq=50, modulation_type='AM'
    )
    analyzer.analyze_spectrum(t, am_signal, method='fft', 
                            title="AM сигнал (500 Гц + 50 Гц)")
    
    # FM модуляция
    print("\nFM модуляция (несущая 600 Гц, модулирующая 30 Гц)...")
    t, fm_signal, _, _ = generator.generate_modulated_signal(
        duration, carrier_freq=600, mod_freq=30, modulation_type='FM'
    )
    analyzer.analyze_spectrum(t, fm_signal, method='fft', 
                            title="FM сигнал (600 Гц + 30 Гц)")
    
    # 3. Демонстрация различных методов преобразования
    print("\n3. Сравнение методов преобразования")
    print("-" * 40)
    
    # Создание композитного сигнала
    t_comp = np.linspace(0, 1, 1024)  # Степень 2 для эффективного БПФ
    composite_signal = (
        0.5 * np.sin(2 * np.pi * 50 * t_comp) +      # НЧ компонента
        0.3 * np.sin(2 * np.pi * 200 * t_comp) +     # СЧ компонента
        0.4 * np.sin(2 * np.pi * 800 * t_comp)       # ВЧ компонента
    )
    
    print("Композитный сигнал (прямое ДПФ):")
    result_comp_dft = analyzer.analyze_spectrum(t_comp[:256], composite_signal[:256], 
                                              method='dft', 
                                              title="Композитный сигнал - Прямое ДПФ")
    
    print("Композитный сигнал (собственное БПФ):")
    result_comp_fft = analyzer.analyze_spectrum(t_comp, composite_signal, 
                                              method='fft', 
                                              title="Композитный сигнал - Собственное БПФ")
    
    print("Композитный сигнал (библиотечное БПФ):")
    result_comp_scipy = analyzer.analyze_spectrum(t_comp, composite_signal, 
                                                method='scipy_fft', 
                                                title="Композитный сигнал - Библиотечное БПФ")
    
    # Сравнение времени выполнения
    import time
    
    print("\n4. Сравнение скорости выполнения")
    print("-" * 40)
    
    test_signal = composite_signal[:256]
    
    # Время выполнения ДПФ
    start_time = time.time()
    dft_result = analyzer.dft_direct(test_signal)
    dft_time = time.time() - start_time
    
    # Время выполнения собственного БПФ
    start_time = time.time()
    fft_result = analyzer.fft_custom(test_signal.astype(complex))
    fft_time = time.time() - start_time
    
    # Время выполнения библиотечного БПФ
    start_time = time.time()
    scipy_fft_result = fft(test_signal)
    scipy_fft_time = time.time() - start_time
    
    print(f"Время выполнения ДПФ: {dft_time:.4f} сек")
    print(f"Время выполнения собственного БПФ: {fft_time:.4f} сек")
    print(f"Время выполнения библиотечного БПФ: {scipy_fft_time:.6f} сек")
    print(f"Ускорение собственного БПФ vs ДПФ: {dft_time/fft_time:.1f}x")
    print(f"Ускорение библиотечного БПФ vs ДПФ: {dft_time/scipy_fft_time:.1f}x")
    print(f"Отношение времени собственного к библиотечному БПФ: {fft_time/scipy_fft_time:.1f}x")
    
    # Проверка точности
    max_error_custom = np.max(np.abs(dft_result - fft_result))
    max_error_scipy = np.max(np.abs(dft_result - scipy_fft_result))
    max_error_comparison = np.max(np.abs(fft_result - scipy_fft_result))
    
    print(f"Максимальная разность ДПФ vs собственное БПФ: {max_error_custom:.2e}")
    print(f"Максимальная разность ДПФ vs библиотечное БПФ: {max_error_scipy:.2e}")
    print(f"Максимальная разность собственное vs библиотечное БПФ: {max_error_comparison:.2e}")
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("Проанализированы различные типы сигналов с использованием:")
    print("- Прямого дискретного преобразования Фурье (ДПФ)")
    print("- Собственной реализации быстрого преобразования Фурье (БПФ)")  
    print("- Библиотечного БПФ (scipy.fft)")
    print("- Обратных преобразований для восстановления сигналов")
    print("- Построения амплитудных и фазовых спектров")
    print("=" * 80)
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import warnings
warnings.filterwarnings('ignore')

class SignalGenerator:
    """Класс для генерации сигналов из лабораторной работы №1"""
    
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
    
    def generate_signal(self, duration, frequency, amplitude=1.0, phase=0.0, 
                       signal_type='sine', duty_cycle=0.5):
        """Генерация сигнала заданной формы"""
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        n_array = np.arange(len(t))
        phi = self.phase_array(frequency, phase, n_array, self.sample_rate)
        return t, self.waveform(amplitude, phi, signal_type, duty_cycle)
    
    def generate_modulated_signal(self, duration, carrier_freq, mod_freq, 
                                carrier_amp=1.0, mod_amp=1.0, modulation_type='AM',
                                carrier_type='sine', mod_type='sine'):
        """Генерация модулированного сигнала"""
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        n_array = np.arange(len(t))
        

        mod_phi = self.phase_array(mod_freq, 0, n_array, self.sample_rate)
        mod_signal = self.waveform(mod_amp, mod_phi, mod_type)
        
        if modulation_type == 'FM':
   
            freq_deviation = mod_freq * 50
            instantaneous_freq = carrier_freq + freq_deviation * mod_signal
            carrier_phi = np.mod(
                2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate,
                2 * np.pi
            )
        else:
     
            carrier_phi = self.phase_array(carrier_freq, 0, n_array, self.sample_rate)
        
        carrier_signal = self.waveform(carrier_amp, carrier_phi, carrier_type)
        
        if modulation_type == 'AM':
         
            modulated_signal = carrier_signal * (1 + mod_signal)
        else:
          
            modulated_signal = carrier_signal
        
        return t, modulated_signal, carrier_signal, mod_signal

class FourierAnalyzer:
    """Класс для анализа Фурье"""
    
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
        
        if N <= 1:
            return signal
        
        if N & (N - 1) != 0:
            next_power_of_2 = 1 << (N - 1).bit_length()
            padded_signal = np.zeros(next_power_of_2, dtype=complex)
            padded_signal[:N] = signal
            result = self.fft_custom(padded_signal)
            return result[:N] if N < next_power_of_2 else result
        
        
        even = self.fft_custom(signal[0::2])
        odd = self.fft_custom(signal[1::2])
        
        T = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        spectrum = np.zeros(N, dtype=complex)
        
        for k in range(N // 2):
            t = T[k] * odd[k]
            spectrum[k] = even[k] + t
            spectrum[k + N // 2] = even[k] - t
        
        return spectrum
    
    def ifft_custom(self, spectrum):
        """Обратное быстрое преобразование Фурье"""
        
        N = len(spectrum)
        conjugated = np.conj(spectrum)
        fft_result = self.fft_custom(conjugated)
        return np.conj(fft_result) / N
    
    def generate_frequency_array(self, N, sample_rate):
        """Генерация массива частот для спектра"""
        freqs = np.zeros(N)
        for k in range(N):
            if k < N // 2:
                freqs[k] = k * sample_rate / N
            else:
                freqs[k] = (k - N) * sample_rate / N
        return freqs
    
    def analyze_spectrum(self, t, signal, method='fft', title="Спектральный анализ"):
        """Полный спектральный анализ сигнала"""
        dt = t[1] - t[0]
        sample_rate = 1.0 / dt
        N = len(signal)
        
        if method == 'dft':
            if N > 1000:
                print(f"Сигнал слишком длинный ({N} точек) для прямого ДПФ. Используется БПФ.")
                spectrum = fft(signal)
                reconstructed = np.real(ifft(spectrum))
                method_name = "БПФ"
            else:
                spectrum = self.dft_direct(signal)
                reconstructed = np.real(self.idft_direct(spectrum))
                method_name = "Прямое ДПФ"
        elif method == 'fft':
            spectrum = fft(signal)
            reconstructed = np.real(ifft(spectrum))
            method_name = "БПФ"
        else:
            raise ValueError("Неподдерживаемый метод. Используйте 'dft', 'fft'")
        
        
        if method == 'fft':
            freqs = fftfreq(N, dt)
        else:
            freqs = self.generate_frequency_array(N, sample_rate)
        
        
        amplitude_spectrum = np.abs(spectrum)
        phase_spectrum = np.angle(spectrum)
        
  
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{title} ({method_name})", fontsize=14, fontweight='bold')
        
        axes[0, 0].plot(t, signal, 'b-', label='Исходный сигнал', alpha=0.8, linewidth=2)
        axes[0, 0].plot(t, reconstructed, 'r--', label='Восстановленный сигнал', alpha=0.8, linewidth=1.5)
        axes[0, 0].set_xlabel('Время (с)')
        axes[0, 0].set_ylabel('Амплитуда')
        axes[0, 0].set_title('Сигналы')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
  
        positive_freqs = freqs[:N//2]
        positive_amplitude = amplitude_spectrum[:N//2]
        axes[0, 1].plot(positive_freqs, positive_amplitude, 'g-', linewidth=1.5)
        axes[0, 1].set_xlabel('Частота (Гц)')
        axes[0, 1].set_ylabel('Амплитуда')
        axes[0, 1].set_title('Амплитудный спектр')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(0, min(2000, max(positive_freqs)))
 
        axes[1, 0].plot(positive_freqs, phase_spectrum[:N//2], 'm-', linewidth=1.5)
        axes[1, 0].set_xlabel('Частота (Гц)')
        axes[1, 0].set_ylabel('Фаза (рад)')
        axes[1, 0].set_title('Фазовый спектр')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, min(2000, max(positive_freqs)))
        
      
        error = signal - reconstructed
        axes[1, 1].plot(t, error, 'r-', linewidth=1)
        axes[1, 1].set_xlabel('Время (с)')
        axes[1, 1].set_ylabel('Ошибка')
        axes[1, 1].set_title(f'Ошибка восстановления (max: {np.max(np.abs(error)):.2e})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'spectrum': spectrum,
            'freqs': freqs,
            'amplitude_spectrum': amplitude_spectrum,
            'phase_spectrum': phase_spectrum,
            'reconstructed': reconstructed,
            'error': error,
            'method': method_name
        }



def demonstrate_fourier_analysis():
    """Демонстрация анализа Фурье различных сигналов"""

    print("=" * 80)
    
    generator = SignalGenerator(sample_rate=8000)
    analyzer = FourierAnalyzer()
    

    print("\n1. Анализ простых сигналов")
    print("-" * 40)
    
    duration = 0.5  
    

    print("Анализ синусоидального сигнала (200 Гц)...")
    t, sine_signal = generator.generate_signal(duration, 200, signal_type='sine')
    

    t_short = t[:500]  
    sine_short = sine_signal[:500]
    
    print("Прямое ДПФ:")
    result_dft = analyzer.analyze_spectrum(t_short, sine_short, method='dft', 
                                         title="Синусоидальный сигнал (200 Гц) - Прямое ДПФ")
    
    print("БПФ:")
    result_fft = analyzer.analyze_spectrum(t, sine_signal, method='fft', 
                                         title="Синусоидальный сигнал (200 Гц) - Scipy БПФ")
    
    print("\nАнализ импульсного сигнала (150 Гц)...")
    t, pulse_signal = generator.generate_signal(duration, 150, signal_type='pulse', duty_cycle=0.3)
    analyzer.analyze_spectrum(t, pulse_signal, title="Импульсный сигнал (150 Гц, скважность 0.3)")
    
    print("\nАнализ треугольного сигнала (100 Гц)...")
    t, triangle_signal = generator.generate_signal(duration, 100, signal_type='triangle')
    analyzer.analyze_spectrum(t, triangle_signal, title="Треугольный сигнал (100 Гц)")
    
    print("\nАнализ пилообразного сигнала (120 Гц)...")
    t, sawtooth_signal = generator.generate_signal(duration, 120, signal_type='sawtooth')
    analyzer.analyze_spectrum(t, sawtooth_signal, title="Пилообразный сигнал (120 Гц)")
    
    print("\n2. Анализ модулированных сигналов")
    print("-" * 40)
    

    print("AM модуляция (несущая 500 Гц, модулирующая 50 Гц)...")
    t, am_signal, carrier, mod = generator.generate_modulated_signal(
        duration, carrier_freq=500, mod_freq=50, modulation_type='AM'
    )
    analyzer.analyze_spectrum(t, am_signal, title="AM сигнал (500 Гц + 50 Гц)")
    

    print("\nFM модуляция (несущая 600 Гц, модулирующая 30 Гц)...")
    t, fm_signal, _, _ = generator.generate_modulated_signal(
        duration, carrier_freq=600, mod_freq=30, modulation_type='FM'
    )
    analyzer.analyze_spectrum(t, fm_signal, title="FM сигнал (600 Гц + 30 Гц)")
    

    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("Проанализированы различные типы сигналов с использованием:")
    print("- Прямого дискретного преобразования Фурье (ДПФ)")
    print("- Быстрого преобразования Фурье (БПФ)")  
    print("- Цифровой фильтрации (НЧ, ВЧ, полосовой)")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_fourier_analysis()