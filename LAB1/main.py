import numpy as np
import sounddevice as sd
import threading
import time
import sys
import msvcrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

class SignalGenerator:
    def __init__(self, sample_rate=44100, buffer_size=1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.t = 0.0
        self.dt = 1.0 / sample_rate
        
        self.frequency = 440.0     
        self.amplitude = 0.5       
        self.phase = 0.0           
        self.s = 2
        
        self.signal_type = 'sine'  
        self.running = True
        
        self.plot_buffer_size = 2048
        self.plot_buffer = deque(maxlen=self.plot_buffer_size)
        self.time_buffer = deque(maxlen=self.plot_buffer_size)
        
        for i in range(self.plot_buffer_size):
            self.plot_buffer.append(0.0)
            self.time_buffer.append(i * self.dt)
        
        self.stream = sd.OutputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=sample_rate,
            blocksize=buffer_size
        )
    
    def generate_sine(self, t_array):
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t_array + self.phase)
    
    def generate_pulse(self, t_array):
        period = 1.0 / self.frequency
        mod_t = np.mod(t_array, period)
        return np.where(mod_t < period / self.s, self.amplitude, -self.amplitude)
    
    def generate_triangle(self, t_array):
        period = 1.0 / self.frequency
        mod_t = np.mod(t_array - period/4, period)
        return 4 * self.amplitude * self.frequency * np.abs(mod_t - period/2) - self.amplitude
    
    def generate_sawtooth(self, t_array):
        period = 1.0 / self.frequency
        mod_t = np.mod(t_array, period)
        return 2 * self.amplitude * self.frequency * mod_t - self.amplitude
    
    def generate_noise(self, t_array):
        return self.amplitude * np.random.uniform(-1, 1, len(t_array))
    
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        
        t_array = np.arange(frames) * self.dt + self.t
        
        if self.signal_type == 'sine':
            signal = self.generate_sine(t_array)
        elif self.signal_type == 'pulse':
            signal = self.generate_pulse(t_array)
        elif self.signal_type == 'triangle':
            signal = self.generate_triangle(t_array)
        elif self.signal_type == 'sawtooth':
            signal = self.generate_sawtooth(t_array)
        elif self.signal_type == 'noise':
            signal = self.generate_noise(t_array)
        else:
            signal = np.zeros(frames)
        
        outdata[:, 0] = signal.astype(np.float32)
        
        for i in range(len(signal)):
            self.plot_buffer.append(signal[i])
            self.time_buffer.append(self.t + i * self.dt)
        
        self.t += frames * self.dt
    
    def start(self):
        self.stream.start()
    
    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
    
    def print_status(self):
        print(f"\rТип: {self.signal_type:8} | Частота: {self.frequency:6.1f} Гц | "
              f"Амплитуда: {self.amplitude:.2f} | Фаза: {self.phase:.2f} | "
              f"Скважность: {self.s:.2f}", end='', flush=True)
    
    def get_plot_data(self):
        """Получить данные для построения графика"""
        return np.array(self.time_buffer), np.array(self.plot_buffer)

class RealTimePlotter:
    def __init__(self, signal_generator):
        self.signal_generator = signal_generator
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [], 'b-', linewidth=1)
        
        self.ax.set_xlim(0, 0.02) 
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_xlabel('Время (с)')
        self.ax.set_ylabel('Амплитуда')
        self.ax.set_title('Реальное время - Генератор сигналов')
        self.ax.grid(True, alpha=0.3)
        
        try:
            self.fig.canvas.set_window_title('Генератор сигналов - Визуализация')
        except AttributeError:

            try:
                self.fig.canvas.manager.set_window_title('Генератор сигналов - Визуализация')
            except:
                pass
        
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot, interval=50, blit=False
        )
        
        plt.show(block=False)
    
    def update_plot(self, frame):
        if not self.signal_generator.running:
            return self.line,
        
        time_data, signal_data = self.signal_generator.get_plot_data()
        
        if len(time_data) > 0:
            window_size = max(int(4 * self.signal_generator.sample_rate / self.signal_generator.frequency), 1000)
            window_size = min(window_size, len(time_data))
            
            time_window = time_data[-window_size:]
            signal_window = signal_data[-window_size:]
            
            if len(time_window) > 0:
                time_normalized = time_window - time_window[0]
                
                self.line.set_data(time_normalized, signal_window)
                
                self.ax.set_xlim(0, time_normalized[-1])
                max_amp = max(1.5, abs(self.signal_generator.amplitude) * 1.2)
                self.ax.set_ylim(-max_amp, max_amp)
                
                self.ax.set_title(f'Тип: {self.signal_generator.signal_type} | '
                                f'Частота: {self.signal_generator.frequency:.1f} Гц | '
                                f'Амплитуда: {self.signal_generator.amplitude:.2f}')
        
        return self.line,
    
    def close(self):
        plt.close(self.fig)

class KeyboardController:
    def __init__(self):
        pass  
    
    def __del__(self):
        pass 
    
    def get_key(self):
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

def print_help():
    """Вывод справки по управлению"""
    print("=" * 80)
    print("Выбор сигнала:")
    print("  1 - Синусоида")
    print("  2 - Импульс")
    print("  3 - Треугольный")
    print("  4 - Пилообразный")
    print("  5 - Шум")
    print()
    print("Управление параметрами:")
    print("  q/a - Частота +/- (10 Гц)")
    print("  w/s - Амплитуда +/- (0.05)")
    print("  e/d - Фаза +/- (π/8)")
    print("  r/f - Скважность +/- (0.1) [только для импульса]")
    print()
    print("  h - Показать эту справку")
    print("  ESC или Ctrl+C - Выход")
    print("=" * 80)

def main():
    print_help()
    
    generator = SignalGenerator()
    keyboard = KeyboardController()
    
    try:
        # Запускаем аудио
        generator.start()
        
        # Запускаем визуализацию
        plotter = RealTimePlotter(generator)
        
        print("\nГенератор запущен! График отображается в отдельном окне.")
        print("Нажмите 'h' для справки.")
        
        while generator.running:
            key = keyboard.get_key()
            
            if key:
                if key == '1':
                    generator.signal_type = 'sine'
                elif key == '2':
                    generator.signal_type = 'pulse'
                elif key == '3':
                    generator.signal_type = 'triangle'
                elif key == '4':
                    generator.signal_type = 'sawtooth'
                elif key == '5':
                    generator.signal_type = 'noise'
                
                elif key == 'q':
                    generator.frequency = min(generator.frequency + 10, 5000)
                elif key == 'a':
                    generator.frequency = max(generator.frequency - 10, 20)
                
                elif key == 'w':
                    generator.amplitude = min(generator.amplitude + 0.05, 10.0)
                elif key == 's':
                    generator.amplitude = max(generator.amplitude - 0.05, 0.0)

                elif key == 'e':
                    generator.phase += np.pi / 8
                    if generator.phase > 2 * np.pi:
                        generator.phase -= 2 * np.pi
                elif key == 'd':
                    generator.phase -= np.pi / 8
                    if generator.phase < 0:
                        generator.phase += 2 * np.pi
                
                elif key == 'r':
                    generator.s = min(generator.s + 0.1, 10.0)
                elif key == 'f':
                    generator.s = max(generator.s - 0.1, 1)
                
                elif key == 'h':
                    print()
                    print_help()
                
                elif key == '\x1b' or key == '\x03':
                    break
            
            generator.print_status()
            
            plt.pause(0.001)
            
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nОстановка генератора...")
        generator.stop()
        if 'plotter' in locals():
            plotter.close()
        print("Генератор остановлен.")

if __name__ == "__main__":
    main()