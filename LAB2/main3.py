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
    
    def noise_wave(self, amplitude, n_samples):
        return amplitude * (2 * np.random.random(n_samples) - 1)

    def waveform(self, amplitude, phi, signal_type, duty_cycle=0.5):
        if signal_type == 'sine':
            return self.sine_wave(amplitude, phi)
        elif signal_type == 'pulse':
            return self.pulse_wave(amplitude, phi, duty_cycle)
        elif signal_type == 'triangle':
            return self.triangle_wave(amplitude, phi)
        elif signal_type == 'sawtooth':
            return self.sawtooth_wave(amplitude, phi)
        elif signal_type == 'noise':
            return self.noise_wave(amplitude, len(phi))
        else:
            return np.zeros_like(phi)

class FourierAnalyzer:
    
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
    
    def fft_custom(self, signal):
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
        N = len(spectrum)
        conjugated = np.conj(spectrum)
        fft_result = self.fft_custom(conjugated)
        return np.conj(fft_result) / N

class DigitalFilters:
    
    def __init__(self):
        pass
    
    def design_lowpass_filter(self, cutoff_freq, sample_rate, order=5):
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        return b, a
    
    def design_highpass_filter(self, cutoff_freq, sample_rate, order=5):
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        return b, a
    
    def design_bandpass_filter(self, low_freq, high_freq, sample_rate, order=5):
        nyquist = sample_rate / 2
        low_normalized = low_freq / nyquist
        high_normalized = high_freq / nyquist
        b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
        return b, a
    
    def apply_filter(self, signal_data, b, a):
        return signal.filtfilt(b, a, signal_data)

class KeyboardController:
    
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
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None

class RealTimeFourierAnalyzer:
    
    def __init__(self, sample_rate=44100, buffer_size=44100):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.dt = 1.0 / sample_rate
        self.t = 0.0    
        
        self.carrier_frequency = 10000.0
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

        self.filter_enabled = False
        self.filter_type = 'lowpass'
        self.filter_cutoff = 1000.0
        self.filter_low = 500.0
        self.filter_high = 1500.0
        self.filter_order = 5
        
        self.generator = SignalGenerator(sample_rate)
        self.analyzer = FourierAnalyzer()
        self.filters = DigitalFilters()
        
        self.running = True
        
        self.buffer_size_vis = 2048
        self.time_buffer = deque(maxlen=self.buffer_size_vis)
        self.carrier_buffer = deque(maxlen=self.buffer_size_vis)
        self.mod_buffer = deque(maxlen=self.buffer_size_vis)
        self.final_buffer = deque(maxlen=self.buffer_size_vis)
        
        self.stream = sd.OutputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=sample_rate,
            blocksize=buffer_size
        )
        
        for i in range(self.buffer_size_vis):
            self.time_buffer.append(i * self.dt)
            self.carrier_buffer.append(0.0)
            self.mod_buffer.append(0.0)
            self.final_buffer.append(0.0)
    
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(f"Audio status: {status}")

        n_array = np.arange(frames)
        N = self.sample_rate

        mod_phi = self.generator.phase_array(self.mod_frequency, self.mod_phase, n_array, N)
        mod_signal = self.generator.waveform(self.mod_amplitude, mod_phi, self.mod_type, self.mod_duty_cycle)

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

        if self.modulation_enabled and self.modulation_mode == 'AM':
            modulated_signal = carrier_signal * (1 + mod_signal)
        else:
            modulated_signal = carrier_signal
        
        if self.filter_enabled:
            try:
                if self.filter_type == 'lowpass':
                    b, a = self.filters.design_lowpass_filter(self.filter_cutoff, self.sample_rate, self.filter_order)
                elif self.filter_type == 'highpass':
                    b, a = self.filters.design_highpass_filter(self.filter_cutoff, self.sample_rate, self.filter_order)
                elif self.filter_type == 'bandpass':
                    b, a = self.filters.design_bandpass_filter(self.filter_low, self.filter_high, self.sample_rate, self.filter_order)
                
                filtered_signal = self.filters.apply_filter(modulated_signal, b, a)
            except:
                filtered_signal = modulated_signal
        else:
            filtered_signal = modulated_signal
        
        outdata[:, 0] = filtered_signal.astype(np.float32)
        
        for i in range(frames):
            self.time_buffer.append(self.t + i * self.dt)
            self.carrier_buffer.append(carrier_signal[i])
            self.mod_buffer.append(mod_signal[i])
            self.final_buffer.append(filtered_signal[i])
        
        self.t += frames * self.dt
    
    def start(self):
        self.stream.start()
    
    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
    
    def get_data_for_plot(self):
        time_arr = np.array(self.time_buffer)
        carrier_arr = np.array(self.carrier_buffer)
        mod_arr = np.array(self.mod_buffer)
        final_arr = np.array(self.final_buffer)
        
        if len(final_arr) > 0:
            spectrum = fft(final_arr)
            freqs = fftfreq(len(final_arr), self.dt)
            
            positive_mask = freqs >= 0
            freqs_positive = freqs[positive_mask]
            spectrum_positive = spectrum[positive_mask]
            
            magnitude = np.abs(spectrum_positive)
            phase = np.angle(spectrum_positive)
            
            return time_arr, carrier_arr, mod_arr, final_arr, freqs_positive, magnitude, phase
        else:
            return time_arr, carrier_arr, mod_arr, final_arr, np.array([]), np.array([]), np.array([])
    
    def print_status(self):
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
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
        plt.ion()
        
        self.fig = plt.figure(figsize=(16, 10))
        
        self.ax_carrier = plt.subplot(3, 2, 1)
        self.line_carrier, = self.ax_carrier.plot([], [], 'b-', linewidth=1.5)
        self.ax_carrier.set_xlim(0, 0.05)
        self.ax_carrier.set_ylim(-1.5, 1.5)
        self.ax_carrier.set_xlabel('Время (с)')
        self.ax_carrier.set_ylabel('Амплитуда')
        self.ax_carrier.set_title('Несущий сигнал')
        self.ax_carrier.grid(True, alpha=0.3)
        
        self.ax_mod = plt.subplot(3, 2, 3)
        self.line_mod, = self.ax_mod.plot([], [], 'g-', linewidth=1.5)
        self.ax_mod.set_xlim(0, 0.05)
        self.ax_mod.set_ylim(-1.5, 1.5)
        self.ax_mod.set_xlabel('Время (с)')
        self.ax_mod.set_ylabel('Амплитуда')
        self.ax_mod.set_title('Модулирующий сигнал')
        self.ax_mod.grid(True, alpha=0.3)
        
        self.ax_final = plt.subplot(3, 2, 5)
        self.line_final, = self.ax_final.plot([], [], 'r-', linewidth=1.5)
        self.ax_final.set_xlim(0, 0.05)
        self.ax_final.set_ylim(-2, 2)
        self.ax_final.set_xlabel('Время (с)')
        self.ax_final.set_ylabel('Амплитуда')
        self.ax_final.set_title('Финальный сигнал')
        self.ax_final.grid(True, alpha=0.3)
        
        self.ax_magnitude = plt.subplot(3, 2, 2)
        self.line_magnitude, = self.ax_magnitude.plot([], [], 'c-', linewidth=1.5)
        self.ax_magnitude.set_xlim(0, 2000)
        self.ax_magnitude.set_ylim(0, 1)
        self.ax_magnitude.set_xlabel('Частота (Гц)')
        self.ax_magnitude.set_ylabel('Амплитуда')
        self.ax_magnitude.set_title('БПФ: Амплитуда от частоты')
        self.ax_magnitude.grid(True, alpha=0.3)
        
        self.ax_phase = plt.subplot(3, 2, 4)
        self.line_phase, = self.ax_phase.plot([], [], 'm-', linewidth=1.5)
        self.ax_phase.set_xlim(0, 2000)
        self.ax_phase.set_ylim(-np.pi, np.pi)
        self.ax_phase.set_xlabel('Частота (Гц)')
        self.ax_phase.set_ylabel('Фаза (рад)')
        self.ax_phase.set_title('БПФ: Фаза от частоты')
        self.ax_phase.grid(True, alpha=0.3)
        
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot, interval=50, blit=False
        )
        
        try:
            self.fig.canvas.set_window_title('Real-time Fourier Analyzer')
        except:
            try:
                self.fig.canvas.manager.set_window_title('Real-time Fourier Analyzer')
            except:
                pass
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update_plot(self, frame):
        if not self.analyzer.running:
            return []
        
        time_data, carrier_data, mod_data, final_data, freqs, magnitude, phase = self.analyzer.get_data_for_plot()
        
        if len(time_data) > 0:
            window_size = min(1000, len(time_data))
            time_window = time_data[-window_size:]
            carrier_window = carrier_data[-window_size:]
            mod_window = mod_data[-window_size:]
            final_window = final_data[-window_size:]
   
            if len(time_window) > 0:
                time_normalized = time_window - time_window[0]
                
                self.line_carrier.set_data(time_normalized, carrier_window)
                self.line_mod.set_data(time_normalized, mod_window)
                self.line_final.set_data(time_normalized, final_window)
                
                if len(time_normalized) > 0:
                    max_time = time_normalized[-1]
                    self.ax_carrier.set_xlim(0, max_time)
                    self.ax_mod.set_xlim(0, max_time)
                    self.ax_final.set_xlim(0, max_time)
                    
                    max_amp_carrier = max(1.5, np.max(np.abs(carrier_window)) * 1.2)
                    max_amp_mod = max(1.5, np.max(np.abs(mod_window)) * 1.2)
                    max_amp_final = max(2.0, np.max(np.abs(final_window)) * 1.2)
                    
                    self.ax_carrier.set_ylim(-max_amp_carrier, max_amp_carrier)
                    self.ax_mod.set_ylim(-max_amp_mod, max_amp_mod)
                    self.ax_final.set_ylim(-max_amp_final, max_amp_final)
        
        if len(freqs) > 0 and len(magnitude) > 0:
            max_freq_display = 2000
            freq_mask = freqs <= max_freq_display
            freqs_display = freqs[freq_mask]
            magnitude_display = magnitude[freq_mask]
            phase_display = phase[freq_mask]
            
            if np.max(magnitude_display) > 0:
                magnitude_normalized = magnitude_display / np.max(magnitude_display)
            else:
                magnitude_normalized = magnitude_display
            
            self.line_magnitude.set_data(freqs_display, magnitude_normalized)
            self.line_phase.set_data(freqs_display, phase_display)
            
            self.ax_magnitude.set_xlim(0, max_freq_display)
            self.ax_phase.set_xlim(0, max_freq_display)
        
        return []
    
    def close(self):
        plt.close(self.fig)

def print_help():
    print("=" * 100)
    print("🎵 REAL-TIME FOURIER ANALYZER С ФИЛЬТРАЦИЕЙ 🎵")
    print("=" * 100)
    print("СИГНАЛ:")
    print("  1-5    - Тип несущего (синус/импульс/треугольный/пилообразный/шум)")
    print("  q/a    - Частота несущего +/- (10 Гц)")
    print("  w/s    - Амплитуда несущего +/- (0.05)")
    print("  e/d    - Фаза несущего +/- (π/8)")
    print("  r/f    - Скважность несущего +/- (0.1)")
    print()
    print("МОДУЛЯЦИЯ:")
    print("  m      - Переключение AM/FM")
    print("  n      - Включить/выключить модуляцию")
    print("  6-0    - Тип модулирующего (синус/импульс/треугольный/пилообразный/шум)")
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
                if key == '1':
                    analyzer.carrier_type = 'sine'
                elif key == '2':
                    analyzer.carrier_type = 'pulse'
                elif key == '3':
                    analyzer.carrier_type = 'triangle'
                elif key == '4':
                    analyzer.carrier_type = 'sawtooth'
                elif key == '5':
                    analyzer.carrier_type = 'noise'
                elif key == '6':
                    analyzer.mod_type = 'sine'
                elif key == '7':
                    analyzer.mod_type = 'pulse'
                elif key == '8':
                    analyzer.mod_type = 'triangle'
                elif key == '9':
                    analyzer.mod_type = 'sawtooth'
                elif key == '0':
                    analyzer.mod_type = 'noise'
                elif key == 'q':
                    analyzer.carrier_frequency = min(analyzer.carrier_frequency + 10, 10000)
                elif key == 'a':
                    analyzer.carrier_frequency = max(analyzer.carrier_frequency - 10, 1)
                elif key == 'w':
                    analyzer.carrier_amplitude = min(analyzer.carrier_amplitude + 0.05, 1.0)
                elif key == 's':
                    analyzer.carrier_amplitude = max(analyzer.carrier_amplitude - 0.05, 0.0)
                elif key == 'e':
                    analyzer.carrier_phase += np.pi / 8
                    if analyzer.carrier_phase > 2 * np.pi:
                        analyzer.carrier_phase -= 2 * np.pi
                elif key == 'd':
                    analyzer.carrier_phase -= np.pi / 8
                    if analyzer.carrier_phase < 0:
                        analyzer.carrier_phase += 2 * np.pi
                elif key == 'r':
                    analyzer.carrier_duty_cycle = min(analyzer.carrier_duty_cycle + 0.1, 1)
                elif key == 'f':
                    analyzer.carrier_duty_cycle = max(analyzer.carrier_duty_cycle - 0.1, 0)
                elif key == 'm':
                    analyzer.modulation_mode = 'FM' if analyzer.modulation_mode == 'AM' else 'AM'
                elif key == 'n':
                    analyzer.modulation_enabled = not analyzer.modulation_enabled
                elif key == 't':
                    analyzer.mod_frequency = min(analyzer.mod_frequency + 1, 100)
                elif key == 'g':
                    analyzer.mod_frequency = max(analyzer.mod_frequency - 1, 0.1)
                elif key == 'y':
                    analyzer.mod_amplitude = min(analyzer.mod_amplitude + 0.05, 1.0)
                elif key == 'h':
                    analyzer.mod_amplitude = max(analyzer.mod_amplitude - 0.05, 0.0)
                elif key == ' ': 
                    analyzer.filter_enabled = not analyzer.filter_enabled
                elif key == 'z':
                    filter_types = ['lowpass', 'highpass', 'bandpass']
                    current_idx = filter_types.index(analyzer.filter_type)
                    analyzer.filter_type = filter_types[(current_idx + 1) % len(filter_types)]
                elif key == 'x':
                    filter_types = ['lowpass', 'highpass', 'bandpass']
                    current_idx = filter_types.index(analyzer.filter_type)
                    analyzer.filter_type = filter_types[(current_idx - 1) % len(filter_types)]
                elif key == 'u':
                    analyzer.filter_cutoff = min(analyzer.filter_cutoff + 50, analyzer.sample_rate // 2 - 100)
                elif key == 'j':
                    analyzer.filter_cutoff = max(analyzer.filter_cutoff - 50, 50)
                elif key == 'i':
                    analyzer.filter_low = min(analyzer.filter_low + 50, analyzer.filter_high - 100)
                elif key == 'k':
                    analyzer.filter_low = max(analyzer.filter_low - 50, 50)
                elif key == 'o':
                    analyzer.filter_high = min(analyzer.filter_high + 50, analyzer.sample_rate // 2 - 100)
                elif key == 'l':
                    analyzer.filter_high = max(analyzer.filter_high - 50, analyzer.filter_low + 100)
                elif key == 'p':
                    analyzer.filter_order = min(analyzer.filter_order + 1, 10)
                elif key == ';':
                    analyzer.filter_order = max(analyzer.filter_order - 1, 1)
                elif key == '?':
                    print()
                    print_help()
                elif key == '\x1b' or key == '\x03': 
                    break
            
            analyzer.print_status()
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