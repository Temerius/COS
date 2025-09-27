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

class NoteSystem:
    """Система именованных музыкальных нот"""
    
    def __init__(self):
        # Базовая частота A4 = 440 Гц
        self.A4_FREQ = 440.0
        
        # Названия нот в хроматической гамме
        self.NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Генерируем все ноты от C0 до C8
        self.notes = {}
        self.generate_notes()
    
    def generate_notes(self):
        """Генерирует все ноты с их частотами"""
        # A4 находится на позиции 57 (считая C0 как 0)
        # Формула: f = f0 * 2^(n/12), где n - полутон относительно A4
        
        for octave in range(9):  # C0 - C8
            for i, note_name in enumerate(self.NOTE_NAMES):
                # Вычисляем номер полутона относительно A4
                midi_number = octave * 12 + i
                semitones_from_A4 = midi_number - 57  # A4 = MIDI 69, но мы считаем от C0
                
                # Вычисляем частоту
                frequency = self.A4_FREQ * (2 ** (semitones_from_A4 / 12))
                
                # Создаем имя ноты
                note_key = f"{note_name}{octave}"
                self.notes[note_key] = frequency
        
        # Добавляем специальные ноты
        self.notes['REST'] = 0.0  # Пауза
        self.notes['SILENCE'] = 0.0  # Тишина
    
    def get_frequency(self, note_name):
        """Получить частоту ноты по имени"""
        return self.notes.get(note_name, 0.0)
    
    def list_notes_in_octave(self, octave):
        """Список всех нот в октаве"""
        return [f"{note}{octave}" for note in self.NOTE_NAMES]
    
    def print_note_info(self, note_name):
        """Показать информацию о ноте"""
        freq = self.get_frequency(note_name)
        if freq > 0:
            print(f"Нота {note_name}: {freq:.2f} Гц")
        else:
            print(f"Нота {note_name}: пауза")

class MelodyPlayer:
    def __init__(self, sample_rate=44100, buffer_size=1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.dt = 1.0 / sample_rate
        
        self.running = True
        self.playing = False
        self.current_note_index = 0
        self.note_start_time = 0.0
        self.global_time = 0.0
        
        self.current_melody = []
        self.melody_name = ""
        self.tempo = 120
        
        # Система нот
        self.note_system = NoteSystem()
        
        self.plot_buffer_size = 4096
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
        
        self.load_melodies()
    
    def load_melodies(self):
        """Загружает мелодии с именованными нотами"""
        self.melodies = {
            'mario': {
                'name': 'Super Mario Bros Theme',
                'tempo': 140,
                'notes': [
                    {'note': 'E5', 'duration': 0.12, 'type': 'pulse', 'amp': 0.6, 'duty': 3.0},
                    {'note': 'E5', 'duration': 0.12, 'type': 'pulse', 'amp': 0.6, 'duty': 3.0},
                    {'note': 'REST', 'duration': 0.12, 'type': 'sine', 'amp': 0, 'duty': 2.0},
                    {'note': 'E5', 'duration': 0.12, 'type': 'pulse', 'amp': 0.6, 'duty': 3.0},
                    {'note': 'REST', 'duration': 0.12, 'type': 'sine', 'amp': 0, 'duty': 2.0},
                    {'note': 'C5', 'duration': 0.12, 'type': 'pulse', 'amp': 0.6, 'duty': 3.0},
                    {'note': 'E5', 'duration': 0.12, 'type': 'pulse', 'amp': 0.6, 'duty': 3.0},
                    {'note': 'REST', 'duration': 0.12, 'type': 'sine', 'amp': 0, 'duty': 2.0},
                    {'note': 'G5', 'duration': 0.24, 'type': 'pulse', 'amp': 0.7, 'duty': 3.0},
                    {'note': 'REST', 'duration': 0.24, 'type': 'sine', 'amp': 0, 'duty': 2.0},
                    {'note': 'G4', 'duration': 0.24, 'type': 'pulse', 'amp': 0.5, 'duty': 3.0},
                    {'note': 'REST', 'duration': 0.24, 'type': 'sine', 'amp': 0, 'duty': 2.0},
                ]
            },
            'tetris': {
                'name': 'Tetris Theme (Korobeiniki)',
                'tempo': 120,
                'notes': [
                    {'note': 'E5', 'duration': 0.4, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'B4', 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'C5', 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'D5', 'duration': 0.4, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'C5', 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'B4', 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'A4', 'duration': 0.4, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'A4', 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'C5', 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'E5', 'duration': 0.4, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'D5', 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'C5', 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'B4', 'duration': 0.6, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'B4', 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'C5', 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'note': 'D5', 'duration': 0.8, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                ]
            },
            'nokia': {
                'name': 'Nokia Ringtone',
                'tempo': 180,
                'notes': [
                    {'note': 'E5', 'duration': 0.125, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'D5', 'duration': 0.125, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'F#4', 'duration': 0.25, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'G#4', 'duration': 0.25, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'C#5', 'duration': 0.125, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'B4', 'duration': 0.125, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'D4', 'duration': 0.25, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'E4', 'duration': 0.25, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'B4', 'duration': 0.125, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'A4', 'duration': 0.125, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'C#4', 'duration': 0.25, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'E4', 'duration': 0.25, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                    {'note': 'A4', 'duration': 0.5, 'type': 'sine', 'amp': 0.7, 'duty': 2.0},
                ]
            },
            'imperial': {
                'name': 'Imperial March (Star Wars)',
                'tempo': 100,
                'notes': [
                    {'note': 'G4', 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'G4', 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'G4', 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'D#4', 'duration': 0.375, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'A#4', 'duration': 0.125, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'G4', 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'D#4', 'duration': 0.375, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'A#4', 'duration': 0.125, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'G4', 'duration': 1.0, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'D5', 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'D5', 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'D5', 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'D#5', 'duration': 0.375, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'A#4', 'duration': 0.125, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'F#4', 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'D#4', 'duration': 0.375, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'A#4', 'duration': 0.125, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'note': 'G4', 'duration': 1.0, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                ]
            }
        }
    
    def generate_sine(self, t_array, frequency, amplitude, phase):
        return amplitude * np.sin(2 * np.pi * frequency * t_array + phase)
    
    def generate_pulse(self, t_array, frequency, amplitude, phase, duty_cycle):
        if frequency == 0:
            return np.zeros(len(t_array))
        period = 1.0 / frequency
        mod_t = np.mod(t_array + phase / (2 * np.pi * frequency), period)
        return np.where(mod_t < period / duty_cycle, amplitude, -amplitude)
    
    def generate_triangle(self, t_array, frequency, amplitude, phase):
        if frequency == 0:
            return np.zeros(len(t_array))
        period = 1.0 / frequency
        phase_offset = phase / (2 * np.pi * frequency)
        mod_t = np.mod(t_array - period/4 + phase_offset, period)
        return 4 * amplitude * frequency * np.abs(mod_t - period/2) - amplitude
    
    def generate_sawtooth(self, t_array, frequency, amplitude, phase):
        if frequency == 0:
            return np.zeros(len(t_array))
        period = 1.0 / frequency
        phase_offset = phase / (2 * np.pi * frequency)
        mod_t = np.mod(t_array + phase_offset, period)
        return 2 * amplitude * frequency * mod_t - amplitude
    
    def generate_signal(self, t_array, signal_type, frequency, amplitude, phase, duty_cycle):
        if frequency == 0:
            return np.zeros(len(t_array))
            
        if signal_type == 'sine':
            return self.generate_sine(t_array, frequency, amplitude, phase)
        elif signal_type == 'pulse':
            return self.generate_pulse(t_array, frequency, amplitude, phase, duty_cycle)
        elif signal_type == 'triangle':
            return self.generate_triangle(t_array, frequency, amplitude, phase)
        elif signal_type == 'sawtooth':
            return self.generate_sawtooth(t_array, frequency, amplitude, phase)
        else:
            return np.zeros(len(t_array))
    
    def load_melody(self, melody_key):
        if melody_key in self.melodies:
            melody_data = self.melodies[melody_key]
            self.current_melody = melody_data['notes']
            self.melody_name = melody_data['name']
            self.tempo = melody_data['tempo']
            self.current_note_index = 0
            self.note_start_time = 0.0
            return True
        return False
    
    def start_melody(self):
        if self.current_melody:
            self.playing = True
            self.current_note_index = 0
            self.note_start_time = 0.0
            self.global_time = 0.0
    
    def stop_melody(self):
        self.playing = False
    
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        
        t_array = np.arange(frames) * self.dt + self.global_time
        signal = np.zeros(frames)
        
        if self.playing and self.current_melody:
            note_time = self.global_time - self.note_start_time
            
            if self.current_note_index < len(self.current_melody):
                current_note = self.current_melody[self.current_note_index]
                
                if note_time >= current_note['duration']:
                    self.current_note_index += 1
                    self.note_start_time = self.global_time
                    if self.current_note_index >= len(self.current_melody):
                        self.current_note_index = 0
                        self.note_start_time = self.global_time
                
                if self.current_note_index < len(self.current_melody):
                    current_note = self.current_melody[self.current_note_index]
                    
                    frequency = self.note_system.get_frequency(current_note['note'])
                    
                    signal = self.generate_signal(
                        t_array - self.note_start_time,
                        current_note['type'],
                        frequency,
                        current_note['amp'],
                        0.0,
                        current_note['duty']
                    )
                    
                    envelope = np.ones(len(signal))
                    if frequency > 0:
                        attack_samples = int(0.01 * self.sample_rate)
                        release_samples = int(0.05 * self.sample_rate)
                        
                        for i in range(min(attack_samples, len(envelope))):
                            envelope[i] = i / attack_samples
                        
                        for i in range(max(0, len(envelope) - release_samples), len(envelope)):
                            envelope[i] = (len(envelope) - i) / release_samples
                    
                    signal *= envelope
        
        outdata[:, 0] = signal.astype(np.float32)
        
        for i in range(len(signal)):
            self.plot_buffer.append(signal[i])
            self.time_buffer.append(self.global_time + i * self.dt)
        
        self.global_time += frames * self.dt
    
    def start(self):
        self.stream.start()
    
    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
    
    def get_plot_data(self):
        return np.array(self.time_buffer), np.array(self.plot_buffer)
    
    def print_status(self):
        status = "ИГРАЕТ" if self.playing else "ПАУЗА"
        progress = f"{self.current_note_index + 1}/{len(self.current_melody)}" if self.current_melody else "0/0"
        current_note_info = ""
        
        if self.playing and self.current_melody and self.current_note_index < len(self.current_melody):
            note_data = self.current_melody[self.current_note_index]
            note_name = note_data['note']
            freq = self.note_system.get_frequency(note_name)
            freq_str = f"{freq:.1f}Гц" if freq > 0 else "пауза"
            current_note_info = f" | Нота: {note_name} ({note_data['type']}) {freq_str}"
        
        print(f"\r{status} | {self.melody_name} | Прогресс: {progress}{current_note_info}",
              end='', flush=True)

class MelodyPlotter:
    def __init__(self, melody_player):
        self.player = melody_player
        
        plt.ion()
        self.fig, (self.ax_wave, self.ax_spectrum) = plt.subplots(2, 1, figsize=(14, 10))
        
        self.wave_line, = self.ax_wave.plot([], [], 'b-', linewidth=1.5)
        self.spectrum_line, = self.ax_spectrum.plot([], [], 'r-', linewidth=1)
        
        self.ax_wave.set_xlim(0, 0.1)
        self.ax_wave.set_ylim(-1.5, 1.5)
        self.ax_wave.set_xlabel('Время (с)')
        self.ax_wave.set_ylabel('Амплитуда')
        self.ax_wave.set_title('Форма волны мелодии', color='blue', fontweight='bold')
        self.ax_wave.grid(True, alpha=0.3)
        
        self.ax_spectrum.set_xlim(0, 2000)
        self.ax_spectrum.set_ylim(0, 1)
        self.ax_spectrum.set_xlabel('Частота (Гц)')
        self.ax_spectrum.set_ylabel('Спектральная плотность')
        self.ax_spectrum.set_title('Спектр сигнала', color='red', fontweight='bold')
        self.ax_spectrum.grid(True, alpha=0.3)
        
        try:
            self.fig.canvas.set_window_title('Проигрыватель мелодий - Анализ')
        except AttributeError:
            try:
                self.fig.canvas.manager.set_window_title('Проигрыватель мелодий - Анализ')
            except:
                pass
        
        plt.tight_layout()
        plt.show(block=False)

    def close(self):
        plt.close(self.fig)



def print_help():
    print("=" * 80)
    print("ПРОИГРЫВАТЕЛЬ МЕЛОДИЙ НА ОСНОВЕ СГЕНЕРИРОВАННЫХ СИГНАЛОВ")
    print("=" * 80)
    print("Управление воспроизведением:")
    print("  Space - Старт/Стоп мелодии")
    print("  r - Рестарт (начать сначала)")
    print()
    print("Выбор мелодии:")
    print("  1 - Super Mario Bros Theme (импульсные сигналы)")
    print("  2 - Tetris Theme (пилообразные сигналы)")
    print("  3 - Nokia Ringtone (синусоидальные сигналы)")
    print("  4 - Imperial March (треугольные сигналы)")
    print()
    print("Информация:")
    print("  i - Показать информацию о текущей мелодии")
    print("  l - Список всех доступных мелодий")
    print()
    print("=" * 80)

def print_melody_info(player):
    if player.current_melody:
        print(f"\nИнформация о мелодии: {player.melody_name}")
        print(f"Темп: {player.tempo} BPM")
        print(f"Количество нот: {len(player.current_melody)}")
        
        signal_types = {}
        for note in player.current_melody:
            signal_types[note['type']] = signal_types.get(note['type'], 0) + 1
        
        print("Используемые типы сигналов:")
        for sig_type, count in signal_types.items():
            print(f"  - {sig_type}: {count} нот")
        
        freq_range = [note['freq'] for note in player.current_melody if note['freq'] > 0]
        if freq_range:
            print(f"Диапазон частот: {min(freq_range):.1f} - {max(freq_range):.1f} Гц")
    else:
        print("\nМелодия не загружена")

def print_melody_list(player):
    print("\nДоступные мелодии:")
    for i, (key, data) in enumerate(player.melodies.items(), 1):
        print(f"  {i}. {data['name']} (темп: {data['tempo']} BPM, нот: {len(data['notes'])})")

def main():
    print_help()
    
    player = MelodyPlayer()
    keyboard = KeyboardController()
    
    try:
        player.start()
        plotter = MelodyPlotter(player)
        
        player.load_melody('mario')
        
        print(f"\nПроигрыватель мелодий запущен!")
        print(f"Загружена мелодия: {player.melody_name}")
        print("Нажмите Space для воспроизведения")
        
        while player.running:
            key = keyboard.get_key()
            
            if key:
                if key == ' ':
                    if player.playing:
                        player.stop_melody()
                    else:
                        player.start_melody()
                elif key == 'r':
                    player.current_note_index = 0
                    player.note_start_time = 0.0
                    if not player.playing:
                        player.start_melody()
                
                elif key == '1':
                    player.load_melody('mario')
                    print(f"\nЗагружена: {player.melody_name}")
                elif key == '2':
                    player.load_melody('tetris')
                    print(f"\nЗагружена: {player.melody_name}")
                elif key == '3':
                    player.load_melody('nokia')
                    print(f"\nЗагружена: {player.melody_name}")
                elif key == '4':
                    player.load_melody('imperial')
                    print(f"\nЗагружена: {player.melody_name}")
                
                elif key == 'i':
                    print_melody_info(player)
                elif key == 'l':
                    print_melody_list(player)
                
                elif key == '\x1b' or key == '\x03':
                    break
            
            player.print_status()
            plt.pause(0.001)
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nОстановка проигрывателя мелодий...")
        player.stop()
        if 'plotter' in locals():
            plotter.close()
        print("Проигрыватель остановлен.")

if __name__ == "__main__":
    main()