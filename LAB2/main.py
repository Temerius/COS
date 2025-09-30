import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class SignalGenerator:
    
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
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        n_array = np.arange(len(t))
        phi = self.phase_array(frequency, phase, n_array, self.sample_rate)
        return t, self.waveform(amplitude, phi, signal_type, duty_cycle)
    
    def generate_modulated_signal(self, duration, carrier_freq, mod_freq, 
                                carrier_amp=1.0, mod_amp=1.0, modulation_type='AM',
                                carrier_type='sine', mod_type='sine'):
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
        N = len(signal)
        spectrum = np.zeros(N, dtype=complex)
        
        for k in range(N):
            for n in range(N):
                spectrum[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
        
        return spectrum
    
    def idft_direct(self, spectrum):
        N = len(spectrum)
        signal = np.zeros(N, dtype=complex)
        
        for n in range(N):
            for k in range(N):
                signal[n] += spectrum[k] * np.exp(2j * np.pi * k * n / N)
            signal[n] /= N
        
        return signal
    
    def analyze_spectrum(self, t, signal, use_fft=True, title="Спектральный анализ"):
        dt = t[1] - t[0]
        N = len(signal)
        
        if use_fft:
            spectrum = fft(signal)
            method = "БПФ (FFT)"
        else:
            if N > 1000:
                print(f"Сигнал слишком длинный ({N} точек) для прямого ДПФ. Используется БПФ.")
                spectrum = fft(signal)
                method = "БПФ (FFT)"
            else:
                spectrum = self.dft_direct(signal)
                method = "Прямое ДПФ"
        
        freqs = fftfreq(N, dt)
        
        amplitude_spectrum = np.abs(spectrum)
        phase_spectrum = np.angle(spectrum)
        
        if use_fft or N > 1000:
            reconstructed = np.real(ifft(spectrum))
        else:
            reconstructed = np.real(self.idft_direct(spectrum))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{title} ({method})", fontsize=14, fontweight='bold')
        
        axes[0, 0].plot(t, signal, 'b-', label='Исходный сигнал', alpha=0.8)
        axes[0, 0].plot(t, reconstructed, 'r--', label='Восстановленный сигнал', alpha=0.8)
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
            'method': method
        }

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
    
    def analyze_filter_response(self, b, a, sample_rate, title="Характеристика фильтра"):
        """Анализ частотной характеристики фильтра"""
        w, h = signal.freqz(b, a, worN=8000)
        freqs = w * sample_rate / (2 * np.pi)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # АЧХ
        ax1.plot(freqs, 20 * np.log10(abs(h)), 'b-', linewidth=2)
        ax1.set_xlabel('Частота (Гц)')
        ax1.set_ylabel('Амплитуда (дБ)')
        ax1.set_title('Амплитудно-частотная характеристика (АЧХ)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, min(2000, max(freqs)))
        
        # ФЧХ
        angles = np.unwrap(np.angle(h))
        ax2.plot(freqs, angles, 'r-', linewidth=2)
        ax2.set_xlabel('Частота (Гц)')
        ax2.set_ylabel('Фаза (рад)')
        ax2.set_title('Фазо-частотная характеристика (ФЧХ)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(2000, max(freqs)))
        
        plt.tight_layout()
        plt.show()
    
    def filter_signal(self, t, signal_data, filter_type, sample_rate, **filter_params):
        """Фильтрация сигнала с визуализацией"""
        
        if filter_type == 'lowpass':
            b, a = self.design_lowpass_filter(
                filter_params['cutoff'], sample_rate, filter_params.get('order', 5)
            )
            title = f"НЧ-фильтр (fc = {filter_params['cutoff']} Гц)"
        elif filter_type == 'highpass':
            b, a = self.design_highpass_filter(
                filter_params['cutoff'], sample_rate, filter_params.get('order', 5)
            )
            title = f"ВЧ-фильтр (fc = {filter_params['cutoff']} Гц)"
        elif filter_type == 'bandpass':
            b, a = self.design_bandpass_filter(
                filter_params['low'], filter_params['high'], 
                sample_rate, filter_params.get('order', 5)
            )
            title = f"Полосовой фильтр ({filter_params['low']}-{filter_params['high']} Гц)"
        else:
            raise ValueError("Неподдерживаемый тип фильтра")
        
        # Применение фильтра
        filtered_signal = self.apply_filter(signal_data, b, a)
        
        # Спектральный анализ
        dt = t[1] - t[0]
        N = len(signal_data)
        freqs = fftfreq(N, dt)
        
        original_spectrum = fft(signal_data)
        filtered_spectrum = fft(filtered_signal)
        
        # Визуализация результатов
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Цифровая фильтрация: {title}", fontsize=14, fontweight='bold')
        
        # Временные сигналы
        axes[0, 0].plot(t, signal_data, 'b-', label='Исходный сигнал', alpha=0.8)
        axes[0, 0].plot(t, filtered_signal, 'r-', label='Отфильтрованный сигнал', alpha=0.8)
        axes[0, 0].set_xlabel('Время (с)')
        axes[0, 0].set_ylabel('Амплитуда')
        axes[0, 0].set_title('Сигналы')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Спектры
        positive_freqs = freqs[:N//2]
        axes[0, 1].plot(positive_freqs, np.abs(original_spectrum[:N//2]), 
                       'b-', label='Исходный спектр', alpha=0.8)
        axes[0, 1].plot(positive_freqs, np.abs(filtered_spectrum[:N//2]), 
                       'r-', label='Отфильтрованный спектр', alpha=0.8)
        axes[0, 1].set_xlabel('Частота (Гц)')
        axes[0, 1].set_ylabel('Амплитуда')
        axes[0, 1].set_title('Спектры')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(0, min(2000, max(positive_freqs)))
        
        # Частотная характеристика фильтра
        w, h = signal.freqz(b, a, worN=8000)
        filter_freqs = w * sample_rate / (2 * np.pi)
        axes[1, 0].plot(filter_freqs, 20 * np.log10(abs(h)), 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Частота (Гц)')
        axes[1, 0].set_ylabel('Амплитуда (дБ)')
        axes[1, 0].set_title('АЧХ фильтра')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, min(2000, max(filter_freqs)))
        
        # Разность сигналов
        difference = signal_data - filtered_signal
        axes[1, 1].plot(t, difference, 'm-', linewidth=1)
        axes[1, 1].set_xlabel('Время (с)')
        axes[1, 1].set_ylabel('Амплитуда')
        axes[1, 1].set_title('Разность (исходный - отфильтрованный)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return filtered_signal, b, a

def demonstrate_fourier_analysis():
    """Демонстрация анализа Фурье различных сигналов"""
    
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА: АНАЛИЗ ФУРЬЕ СИГНАЛОВ")
    print("=" * 80)
    
    generator = SignalGenerator(sample_rate=8000)  # Уменьшенная частота дискретизации
    analyzer = FourierAnalyzer()
    filters = DigitalFilters()
    
    # 1. Анализ простых сигналов
    print("\n1. Анализ простых сигналов")
    print("-" * 40)
    
    duration = 0.5  # секунд
    
    # Синусоидальный сигнал
    print("Анализ синусоидального сигнала (200 Гц)...")
    t, sine_signal = generator.generate_signal(duration, 200, signal_type='sine')
    
    # Сравнение ДПФ и БПФ для короткого сигнала
    t_short = t[:500]  # Короткий сигнал для прямого ДПФ
    sine_short = sine_signal[:500]
    
    print("Прямое ДПФ:")
    result_dft = analyzer.analyze_spectrum(t_short, sine_short, use_fft=False, 
                                         title="Синусоидальный сигнал (200 Гц) - Прямое ДПФ")
    
    print("БПФ:")
    result_fft = analyzer.analyze_spectrum(t, sine_signal, use_fft=True, 
                                         title="Синусоидальный сигнал (200 Гц) - БПФ")
    
    # Импульсный сигнал
    print("\nАнализ импульсного сигнала (150 Гц)...")
    t, pulse_signal = generator.generate_signal(duration, 150, signal_type='pulse', duty_cycle=0.3)
    analyzer.analyze_spectrum(t, pulse_signal, title="Импульсный сигнал (150 Гц, скважность 0.3)")
    
    # Треугольный сигнал
    print("\nАнализ треугольного сигнала (100 Гц)...")
    t, triangle_signal = generator.generate_signal(duration, 100, signal_type='triangle')
    analyzer.analyze_spectrum(t, triangle_signal, title="Треугольный сигнал (100 Гц)")
    
    # Пилообразный сигнал
    print("\nАнализ пилообразного сигнала (120 Гц)...")
    t, sawtooth_signal = generator.generate_signal(duration, 120, signal_type='sawtooth')
    analyzer.analyze_spectrum(t, sawtooth_signal, title="Пилообразный сигнал (120 Гц)")
    
    # 2. Анализ модулированных сигналов
    print("\n2. Анализ модулированных сигналов")
    print("-" * 40)
    
    # AM модуляция
    print("AM модуляция (несущая 500 Гц, модулирующая 50 Гц)...")
    t, am_signal, carrier, mod = generator.generate_modulated_signal(
        duration, carrier_freq=500, mod_freq=50, modulation_type='AM'
    )
    analyzer.analyze_spectrum(t, am_signal, title="AM сигнал (500 Гц + 50 Гц)")
    
    # FM модуляция
    print("\nFM модуляция (несущая 600 Гц, модулирующая 30 Гц)...")
    t, fm_signal, _, _ = generator.generate_modulated_signal(
        duration, carrier_freq=600, mod_freq=30, modulation_type='FM'
    )
    analyzer.analyze_spectrum(t, fm_signal, title="FM сигнал (600 Гц + 30 Гц)")
    
    # 3. Цифровая фильтрация
    print("\n3. Цифровая фильтрация сигналов")
    print("-" * 40)
    
    # Создание композитного сигнала для фильтрации
    t = np.linspace(0, 1, generator.sample_rate)
    composite_signal = (
        0.5 * np.sin(2 * np.pi * 50 * t) +      # НЧ компонента
        0.3 * np.sin(2 * np.pi * 200 * t) +     # СЧ компонента
        0.4 * np.sin(2 * np.pi * 800 * t) +     # ВЧ компонента
        0.1 * np.random.randn(len(t))            # Шум
    )
    
    print("Исходный композитный сигнал:")
    analyzer.analyze_spectrum(t, composite_signal, title="Композитный сигнал (50+200+800 Гц + шум)")
    
    # НЧ-фильтрация
    print("\nПрименение НЧ-фильтра (fc = 150 Гц)...")
    filtered_lp, _, _ = filters.filter_signal(
        t, composite_signal, 'lowpass', generator.sample_rate, cutoff=150, order=6
    )
    
    # ВЧ-фильтрация
    print("\nПрименение ВЧ-фильтра (fc = 300 Гц)...")
    filtered_hp, _, _ = filters.filter_signal(
        t, composite_signal, 'highpass', generator.sample_rate, cutoff=300, order=6
    )
    
    # Полосовая фильтрация
    print("\nПрименение полосового фильтра (150-300 Гц)...")
    filtered_bp, _, _ = filters.filter_signal(
        t, composite_signal, 'bandpass', generator.sample_rate, 
        low=150, high=300, order=4
    )
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("Проанализированы различные типы сигналов с использованием:")
    print("- Прямого дискретного преобразования Фурье (ДПФ)")
    print("- Быстрого преобразования Фурье (БПФ)")  
    print("- Цифровой фильтрации (НЧ, ВЧ, полосовой)")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_fourier_analysis()