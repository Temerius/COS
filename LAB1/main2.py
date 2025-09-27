import numpy as np
import sounddevice as sd
import threading
import time
import sys
import msvcrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from main import SignalGenerator, RealTimePlotter, KeyboardController

class PolyphonicGenerator:
    def __init__(self, sample_rate=44100, buffer_size=1024, max_voices=4):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_voices = max_voices
        self.t = 0.0
        self.dt = 1.0 / sample_rate
        
        self.voices = []
        for i in range(max_voices):
            voice = {
                'frequency': 440.0 + i * 110.0,
                'amplitude': 0.2,
                'phase': 0.0,
                's': 2.0,
                'signal_type': 'sine',
                'active': i == 0
            }
            self.voices.append(voice)
        
        self.current_voice = 0
        self.running = True
        
        self.plot_buffer_size = 2048
        self.plot_buffer = deque(maxlen=self.plot_buffer_size)
        self.time_buffer = deque(maxlen=self.plot_buffer_size)
        self.voice_buffers = [deque(maxlen=self.plot_buffer_size) for _ in range(max_voices)]
        
        for i in range(self.plot_buffer_size):
            self.plot_buffer.append(0.0)
            self.time_buffer.append(i * self.dt)
            for j in range(max_voices):
                self.voice_buffers[j].append(0.0)
        
        self.stream = sd.OutputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=sample_rate,
            blocksize=buffer_size
        )
    
    def generate_sine(self, t_array, voice):
        return voice['amplitude'] * np.sin(2 * np.pi * voice['frequency'] * t_array + voice['phase'])
    
    def generate_pulse(self, t_array, voice):
        period = 1.0 / voice['frequency']
        mod_t = np.mod(t_array, period)
        return np.where(mod_t < period / voice['s'], voice['amplitude'], -voice['amplitude'])
    
    def generate_triangle(self, t_array, voice):
        period = 1.0 / voice['frequency']
        mod_t = np.mod(t_array - period/4, period)
        return 4 * voice['amplitude'] * voice['frequency'] * np.abs(mod_t - period/2) - voice['amplitude']
    
    def generate_sawtooth(self, t_array, voice):
        period = 1.0 / voice['frequency']
        mod_t = np.mod(t_array, period)
        return 2 * voice['amplitude'] * voice['frequency'] * mod_t - voice['amplitude']
    
    def generate_noise(self, t_array, voice):
        return voice['amplitude'] * np.random.uniform(-1, 1, len(t_array))
    
    def generate_voice_signal(self, t_array, voice):
        if not voice['active']:
            return np.zeros(len(t_array))
            
        if voice['signal_type'] == 'sine':
            return self.generate_sine(t_array, voice)
        elif voice['signal_type'] == 'pulse':
            return self.generate_pulse(t_array, voice)
        elif voice['signal_type'] == 'triangle':
            return self.generate_triangle(t_array, voice)
        elif voice['signal_type'] == 'sawtooth':
            return self.generate_sawtooth(t_array, voice)
        elif voice['signal_type'] == 'noise':
            return self.generate_noise(t_array, voice)
        else:
            return np.zeros(len(t_array))
    
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        
        t_array = np.arange(frames) * self.dt + self.t
        
        combined_signal = np.zeros(frames)
        
        for i, voice in enumerate(self.voices):
            voice_signal = self.generate_voice_signal(t_array, voice)
            combined_signal += voice_signal
            
            for j in range(len(voice_signal)):
                self.voice_buffers[i].append(voice_signal[j])
        
        active_voices = sum(1 for voice in self.voices if voice['active'])
        if active_voices > 0:
            combined_signal /= np.sqrt(active_voices)
        
        outdata[:, 0] = combined_signal.astype(np.float32)
        
        for i in range(len(combined_signal)):
            self.plot_buffer.append(combined_signal[i])
            self.time_buffer.append(self.t + i * self.dt)
        
        self.t += frames * self.dt
    
    def start(self):
        self.stream.start()
    
    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
    
    def get_plot_data(self):
        return np.array(self.time_buffer), np.array(self.plot_buffer)
    
    def get_voice_plot_data(self, voice_index):
        return np.array(self.time_buffer), np.array(self.voice_buffers[voice_index])
    
    def print_status(self):
        active_count = sum(1 for voice in self.voices if voice['active'])
        current = self.voices[self.current_voice]
        
        voice_status = "Голоса: "
        for i, voice in enumerate(self.voices):
            marker = "*" if i == self.current_voice else " "
            status = "ON" if voice['active'] else "OFF"
            voice_status += f"{marker}{i+1}:{status} "
        
        print(f"\r{voice_status}| Активных: {active_count} | "
              f"Текущий[{self.current_voice+1}]: {current['signal_type']} "
              f"f={current['frequency']:.0f} a={current['amplitude']:.2f}", 
              end='', flush=True)

class PolyphonicPlotter:
    def __init__(self, polyphonic_generator):
        self.generator = polyphonic_generator
        
        plt.ion()
        self.fig, (self.ax_combined, self.ax_voices) = plt.subplots(2, 1, figsize=(14, 10))
        
        self.combined_line, = self.ax_combined.plot([], [], 'r-', linewidth=2, label='Полифония')
        self.voice_lines = []
        colors = ['b-', 'g-', 'm-', 'c-']
        for i in range(self.generator.max_voices):
            line, = self.ax_voices.plot([], [], colors[i], linewidth=1, alpha=0.7, label=f'Голос {i+1}')
            self.voice_lines.append(line)
        
        for ax in [self.ax_combined, self.ax_voices]:
            ax.set_xlim(0, 0.02)
            ax.set_ylim(-2, 2)
            ax.set_xlabel('Время (с)')
            ax.set_ylabel('Амплитуда')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        self.ax_combined.set_title('Полифонический сигнал')
        self.ax_voices.set_title('Отдельные голоса')
        
        try:
            self.fig.canvas.set_window_title('Полифонический генератор - Визуализация')
        except AttributeError:
            try:
                self.fig.canvas.manager.set_window_title('Полифонический генератор - Визуализация')
            except:
                pass
        
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot, interval=50, blit=False
        )
        
        plt.show(block=False)
    
    def update_plot(self, frame):
        if not self.generator.running:
            return [self.combined_line] + self.voice_lines
        
        time_data, combined_data = self.generator.get_plot_data()
        
        if len(time_data) > 0:
            window_size = min(1000, len(time_data))
            time_window = time_data[-window_size:]
            combined_window = combined_data[-window_size:]
            
            if len(time_window) > 0:
                time_normalized = time_window - time_window[0]
                
                self.combined_line.set_data(time_normalized, combined_window)
                
                for i, line in enumerate(self.voice_lines):
                    _, voice_data = self.generator.get_voice_plot_data(i)
                    voice_window = voice_data[-window_size:]
                    
                    if self.generator.voices[i]['active']:
                        line.set_data(time_normalized, voice_window)
                        line.set_alpha(0.8)
                    else:
                        line.set_data([], [])
                        line.set_alpha(0.3)
                
                for ax in [self.ax_combined, self.ax_voices]:
                    ax.set_xlim(0, time_normalized[-1])
                    max_amp = 2.5
                    ax.set_ylim(-max_amp, max_amp)
        
        return [self.combined_line] + self.voice_lines
    
    def close(self):
        plt.close(self.fig)

def print_help():
    print("=" * 90)
    print("Управление голосами:")
    print()
    print("Выбор типа сигнала для текущего голоса:")
    print("  1-5 - Синусоида/Импульс/Треугольный/Пилообразный/Шум")
    print()
    print("Параметры текущего голоса:")
    print("  q/a - Частота +/- (10 Гц)")
    print("  w/s - Амплитуда +/- (0.02)")
    print("  e/d - Фаза +/- (π/8)")
    print("  r/f - Скважность +/- (0.1)")
    print()
    print("  h - Эта справка | ESC/Ctrl+C - Выход")
    print("=" * 90)

def setup_chord_preset(generator, frequencies, signal_type='sine'):
    for i, freq in enumerate(frequencies):
        if i < len(generator.voices):
            generator.voices[i]['frequency'] = freq
            generator.voices[i]['signal_type'] = signal_type
            generator.voices[i]['amplitude'] = 0.15
            generator.voices[i]['active'] = True
    
    for i in range(len(frequencies), len(generator.voices)):
        generator.voices[i]['active'] = False

def main():
    print_help()
    
    generator = PolyphonicGenerator()
    keyboard = KeyboardController()
    
    try:
        generator.start()
        plotter = PolyphonicPlotter(generator)
        
        print("\nПолифонический генератор запущен!")
        
        while generator.running:
            key = keyboard.get_key()
            
            if key:
                current_voice = generator.voices[generator.current_voice]
                
                if key == '\t':
                    generator.current_voice = (generator.current_voice + 1) % generator.max_voices
                elif key == ' ':
                    current_voice['active'] = not current_voice['active']
                elif key == '`':
                    all_active = all(voice['active'] for voice in generator.voices)
                    for voice in generator.voices:
                        voice['active'] = not all_active
                
                elif key == '1':
                    current_voice['signal_type'] = 'sine'
                elif key == '2':
                    current_voice['signal_type'] = 'pulse'
                elif key == '3':
                    current_voice['signal_type'] = 'triangle'
                elif key == '4':
                    current_voice['signal_type'] = 'sawtooth'
                elif key == '5':
                    current_voice['signal_type'] = 'noise'
                
                elif key == 'q':
                    current_voice['frequency'] = min(current_voice['frequency'] + 10, 5000)
                elif key == 'a':
                    current_voice['frequency'] = max(current_voice['frequency'] - 10, 20)
                
                elif key == 'w':
                    current_voice['amplitude'] = min(current_voice['amplitude'] + 0.02, 1.0)
                elif key == 's':
                    current_voice['amplitude'] = max(current_voice['amplitude'] - 0.02, 0.0)
                
                elif key == 'e':
                    current_voice['phase'] += np.pi / 8
                    if current_voice['phase'] > 2 * np.pi:
                        current_voice['phase'] -= 2 * np.pi
                elif key == 'd':
                    current_voice['phase'] -= np.pi / 8
                    if current_voice['phase'] < 0:
                        current_voice['phase'] += 2 * np.pi
                
                elif key == 'r':
                    current_voice['s'] = min(current_voice['s'] + 0.1, 10.0)
                elif key == 'f':
                    current_voice['s'] = max(current_voice['s'] - 0.1, 1)
                
                elif key == 'h':
                    print()
                    print_help()
                
                elif key == '\x1b' or key == '\x03':
                    break
            
            generator.print_status()
            plt.pause(0.05)
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nОстановка полифонического генератора...")
        generator.stop()
        if 'plotter' in locals():
            plotter.close()
        print("Генератор остановлен.")

if __name__ == "__main__":
    main()