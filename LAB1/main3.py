import numpy as np
import sounddevice as sd
import threading
import time
import sys
import msvcrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from main import KeyboardController

class ModulatedGenerator:
    def __init__(self, sample_rate=44100, buffer_size=44100):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.t = 0.0
        self.dt = 1.0 / sample_rate
        
        self.carrier_frequency = 800.0
        self.carrier_amplitude = 1
        self.carrier_phase = 0.0
        self.carrier_duty_cycle = 0.5
        self.carrier_type = 'pulse'
        
        self.mod_frequency = 2.0
        self.mod_amplitude = 1
        self.mod_phase = 0.0
        self.mod_duty_cycle = 0.5
        self.mod_type = 'triangle'
        
        self.modulation_mode = 'FM'
        self.modulation_enabled = True
        
        self.running = True
        
        self.plot_buffer_size = 2048
        self.plot_buffer = deque(maxlen=self.plot_buffer_size)
        self.carrier_buffer = deque(maxlen=self.plot_buffer_size)
        self.mod_buffer = deque(maxlen=self.plot_buffer_size)
        self.time_buffer = deque(maxlen=self.plot_buffer_size)
        
        for i in range(self.plot_buffer_size):
            self.plot_buffer.append(0.0)
            self.carrier_buffer.append(0.0)
            self.mod_buffer.append(0.0)
            self.time_buffer.append(i * self.dt)
        
        self.stream = sd.OutputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=sample_rate,
            blocksize=buffer_size
        )
    
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
    
    # def generate_signal(self, t_array, signal_type, frequency, amplitude, phase, duty_cycle):
    #     if signal_type == 'sine':
    #         return self.generate_sine(t_array, frequency, amplitude, phase)
    #     elif signal_type == 'pulse':
    #         return self.generate_pulse(t_array, frequency, amplitude, phase, duty_cycle)
    #     elif signal_type == 'triangle':
    #         return self.generate_triangle(t_array, frequency, amplitude, phase)
    #     elif signal_type == 'sawtooth':
    #         return self.generate_sawtooth(t_array, frequency, amplitude, phase)
    #     else:
    #         return np.zeros(len(t_array))
    
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(f"Audio status: {status}")

        n_array = np.arange(frames)
        N = self.sample_rate

        mod_phi = self.phase_array(self.mod_frequency, self.mod_phase, n_array, N)
        mod_signal = self.waveform(self.mod_amplitude, mod_phi, self.mod_type, self.mod_duty_cycle)

        if self.modulation_enabled and self.modulation_mode == 'FM':
            freq_deviation = self.mod_frequency * 50
            instantaneous_freq = self.carrier_frequency + freq_deviation * mod_signal
            carrier_phi = np.mod(
                2 * np.pi * np.cumsum(instantaneous_freq) / N + self.carrier_phase,
                2 * np.pi
            )
        else:
            carrier_phi = self.phase_array(self.carrier_frequency, self.carrier_phase, n_array, N)

        carrier_signal = self.waveform(self.carrier_amplitude, carrier_phi, self.carrier_type, self.carrier_duty_cycle)

        if self.modulation_enabled and self.modulation_mode == 'AM':
            modulated_signal = carrier_signal * (1 + mod_signal)
        else:
            modulated_signal = carrier_signal

        outdata[:, 0] = modulated_signal.astype(np.float32)

        for i in range(frames):
            self.plot_buffer.append(modulated_signal[i])
            self.carrier_buffer.append(carrier_signal[i])
            self.mod_buffer.append(mod_signal[i])
            self.time_buffer.append(self.t + i * self.dt)

        self.t += frames * self.dt
    
    def start(self):
        self.stream.start()
    
    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
    
    def get_plot_data(self):
        return (np.array(self.time_buffer), np.array(self.plot_buffer), 
                np.array(self.carrier_buffer), np.array(self.mod_buffer))
    
    def print_status(self):
        mod_status = "ON" if self.modulation_enabled else "OFF"
        print(f"\rМодуляция: {mod_status} ({self.modulation_mode}) | "
              f"Несущая: {self.carrier_type} f={self.carrier_frequency:.1f}Гц a={self.carrier_amplitude:.2f} | "
              f"Модулирующая: {self.mod_type} f={self.mod_frequency:.1f}Гц a={self.mod_amplitude:.2f}",
              end='', flush=True)

class ModulatedPlotter:
    def __init__(self, modulated_generator):
        self.generator = modulated_generator
        
        plt.ion()
        self.fig, (self.ax_mod, self.ax_carrier, self.ax_modulating) = plt.subplots(3, 1, figsize=(14, 12))
        
        self.modulated_line, = self.ax_mod.plot([], [], 'r-', linewidth=2)
        self.carrier_line, = self.ax_carrier.plot([], [], 'b-', linewidth=1.5)
        self.modulating_line, = self.ax_modulating.plot([], [], 'g-', linewidth=1.5)
        
        axes_config = [
            (self.ax_mod, 'Модулированный сигнал', 'red'),
            (self.ax_carrier, 'Несущий сигнал', 'blue'),
            (self.ax_modulating, 'Модулирующий сигнал', 'green')
        ]
        
        for ax, title, color in axes_config:
            ax.set_xlim(0, 0.02)
            ax.set_ylim(-2, 2)
            ax.set_xlabel('Время (с)')
            ax.set_ylabel('Амплитуда')
            ax.set_title(title, color=color, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        try:
            self.fig.canvas.set_window_title('Генератор с модуляцией - Визуализация')
        except AttributeError:
            try:
                self.fig.canvas.manager.set_window_title('Генератор с модуляцией - Визуализация')
            except:
                pass
        
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot, interval=50, blit=False
        )
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update_plot(self, frame):
        if not self.generator.running:
            return [self.modulated_line, self.carrier_line, self.modulating_line]
        
        time_data, modulated_data, carrier_data, mod_data = self.generator.get_plot_data()
        
        if len(time_data) > 0:
            window_size = min(1500, len(time_data))
            
            time_window = time_data[-window_size:]
            modulated_window = modulated_data[-window_size:]
            carrier_window = carrier_data[-window_size:]
            mod_window = mod_data[-window_size:]
            
            if len(time_window) > 0:
                time_normalized = time_window - time_window[0]
                
                self.modulated_line.set_data(time_normalized, modulated_window)
                self.carrier_line.set_data(time_normalized, carrier_window)
                self.modulating_line.set_data(time_normalized, mod_window)
                
                for ax in [self.ax_mod, self.ax_carrier, self.ax_modulating]:
                    ax.set_xlim(0, time_normalized[-1])
                
                max_mod = max(2.0, np.max(np.abs(modulated_window)) * 1.2) if len(modulated_window) > 0 else 2.0
                max_carrier = max(1.5, np.max(np.abs(carrier_window)) * 1.2) if len(carrier_window) > 0 else 1.5
                max_modulating = max(1.0, np.max(np.abs(mod_window)) * 1.2) if len(mod_window) > 0 else 1.0
                
                self.ax_mod.set_ylim(-max_mod, max_mod)
                self.ax_carrier.set_ylim(-max_carrier, max_carrier)
                self.ax_modulating.set_ylim(-max_modulating, max_modulating)
        
        return [self.modulated_line, self.carrier_line, self.modulating_line]
    
    def close(self):
        plt.close(self.fig)

def print_help():
    print("=" * 100)
    print("ГЕНЕРАТОР С МОДУЛЯЦИЕЙ СИГНАЛОВ")
    print("=" * 100)
    print("Режимы модуляции:")
    print("  m - Переключение режима модуляции (AM/FM)")
    print("  n - Включить/выключить модуляцию")
    print()
    print("Настройка НЕСУЩЕГО сигнала:")
    print("  1-4 - Тип несущего (синус/импульс/треугольный/пилообразный)")
    print("  q/a - Частота несущего +/- (10 Гц)")
    print("  w/s - Амплитуда несущего +/- (0.05)")
    print("  e/d - Фаза несущего +/- (π/8)")
    print("  r/f - Скважность несущего +/- (0.1)")
    print()
    print("Настройка МОДУЛИРУЮЩЕГО сигнала (SHIFT+клавиша):")
    print("  !/@/#/$ - Тип модулирующего (синус/импульс/треугольный/пилообразный)")
    print("  Q/A - Частота модулирующего +/- (1 Гц)")
    print("  W/S - Глубина модуляции +/- (0.05)")
    print("  E/D - Фаза модулирующего +/- (π/8)")
    print("  R/F - Скважность модулирующего +/- (0.1)")
    print()
    print("  h - Эта справка | ESC/Ctrl+C - Выход")
    print("=" * 100)


def main():
    print_help()
    
    generator = ModulatedGenerator()
    keyboard = KeyboardController()
    
    try:
        generator.start()
        plotter = ModulatedPlotter(generator)
        
        print("\nГенератор с модуляцией запущен!")
        print("По умолчанию: AM модуляция включена")
        
        while generator.running:
            key = keyboard.get_key()
            
            if key:
                if key == 'm':
                    generator.modulation_mode = 'FM' if generator.modulation_mode == 'AM' else 'AM'
                elif key == 'n':
                    generator.modulation_enabled = not generator.modulation_enabled
                
                elif key == '1':
                    generator.carrier_type = 'sine'
                elif key == '2':
                    generator.carrier_type = 'pulse'
                elif key == '3':
                    generator.carrier_type = 'triangle'
                elif key == '4':
                    generator.carrier_type = 'sawtooth'
                
                elif key == '5':
                    generator.mod_type = 'sine'
                elif key == '6':
                    generator.mod_type = 'pulse'
                elif key == '7':
                    generator.mod_type = 'triangle'
                elif key == '8':
                    generator.mod_type = 'sawtooth'
                
                elif key == 'q':
                    generator.carrier_frequency = min(generator.carrier_frequency + 10, 5000)
                elif key == 'a':
                    generator.carrier_frequency = max(generator.carrier_frequency - 10, 1)
                elif key == 'Q':
                    generator.mod_frequency = min(generator.mod_frequency + 1, 5000)
                elif key == 'A':
                    generator.mod_frequency = max(generator.mod_frequency - 1, 1)
                
                elif key == 'w':
                    generator.carrier_amplitude = min(generator.carrier_amplitude + 0.05, 1.0)
                elif key == 's':
                    generator.carrier_amplitude = max(generator.carrier_amplitude - 0.05, 0.0)
                elif key == 'W':
                    generator.mod_amplitude = min(generator.mod_amplitude + 0.05, 1.0)
                elif key == 'S':
                    generator.mod_amplitude = max(generator.mod_amplitude - 0.05, 0.0)
                
                elif key == 'e':
                    generator.carrier_phase += np.pi / 8
                    if generator.carrier_phase > 2 * np.pi:
                        generator.carrier_phase -= 2 * np.pi
                elif key == 'd':
                    generator.carrier_phase -= np.pi / 8
                    if generator.carrier_phase < 0:
                        generator.carrier_phase += 2 * np.pi
                elif key == 'E':
                    generator.mod_phase += np.pi / 8
                    if generator.mod_phase > 2 * np.pi:
                        generator.mod_phase -= 2 * np.pi
                elif key == 'D':
                    generator.mod_phase -= np.pi / 8
                    if generator.mod_phase < 0:
                        generator.mod_phase += 2 * np.pi
                
                elif key == 'r':
                    generator.carrier_duty_cycle = min(generator.carrier_duty_cycle + 0.1, 1)
                elif key == 'f':
                    generator.carrier_duty_cycle = max(generator.carrier_duty_cycle - 0.1, 0)
                elif key == 'R':
                    generator.mod_duty_cycle = min(generator.mod_duty_cycle + 0.1, 1)
                elif key == 'F':
                    generator.mod_duty_cycle = max(generator.mod_duty_cycle - 0.1, 0)
            
                
                elif key == '\x1b' or key == '\x03':
                    break
            
            generator.print_status()
            plt.pause(0.05)
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nОстановка генератора с модуляцией...")
        generator.stop()
        if 'plotter' in locals():
            plotter.close()
        print("Генератор остановлен.")

if __name__ == "__main__":
    main()