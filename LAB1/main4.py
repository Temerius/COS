import numpy as np
import sounddevice as sd
import time
import sys
import msvcrt
from main import KeyboardController

class MelodyPlayer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        self.running = True
        self.playing = False
        self.current_audio_data = None
        self.current_position = 0
        
        self.current_melody = []
        self.melody_name = ""
        self.tempo = 120
        
        self.stream = sd.OutputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=sample_rate,
            blocksize=4096 
        )
        
        self.load_melodies()
    
    def load_melodies(self):
        self.melodies = {
            'mario': {
                'name': 'Super Mario Bros Theme',
                'tempo': 140,
                'notes': [
                    {'freq': 659.25, 'duration': 0.12, 'type': 'pulse', 'amp': 0.6, 'duty': 3.0},
                    {'freq': 659.25, 'duration': 0.12, 'type': 'pulse', 'amp': 0.6, 'duty': 3.0},
                    {'freq': 0, 'duration': 0.12, 'type': 'sine', 'amp': 0, 'duty': 2.0},
                    {'freq': 659.25, 'duration': 0.12, 'type': 'pulse', 'amp': 0.6, 'duty': 3.0},
                    {'freq': 0, 'duration': 0.12, 'type': 'sine', 'amp': 0, 'duty': 2.0},
                    {'freq': 523.25, 'duration': 0.12, 'type': 'pulse', 'amp': 0.6, 'duty': 3.0},
                    {'freq': 659.25, 'duration': 0.12, 'type': 'pulse', 'amp': 0.6, 'duty': 3.0},
                    {'freq': 0, 'duration': 0.12, 'type': 'sine', 'amp': 0, 'duty': 2.0},
                    {'freq': 783.99, 'duration': 0.24, 'type': 'pulse', 'amp': 0.7, 'duty': 3.0},
                    {'freq': 0, 'duration': 0.24, 'type': 'sine', 'amp': 0, 'duty': 2.0},
                    {'freq': 392.00, 'duration': 0.24, 'type': 'pulse', 'amp': 0.5, 'duty': 3.0},
                    {'freq': 0, 'duration': 0.24, 'type': 'sine', 'amp': 0, 'duty': 2.0},
                ]
            },
            'tetris': {
                'name': 'Tetris Theme',
                'tempo': 120,
                'notes': [
                    {'freq': 659.25, 'duration': 0.4, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 493.88, 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 523.25, 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 587.33, 'duration': 0.4, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 523.25, 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 493.88, 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 440.00, 'duration': 0.4, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 440.00, 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 523.25, 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 659.25, 'duration': 0.4, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 587.33, 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 523.25, 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 493.88, 'duration': 0.6, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 493.88, 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 523.25, 'duration': 0.2, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                    {'freq': 587.33, 'duration': 0.8, 'type': 'sawtooth', 'amp': 0.5, 'duty': 2.0},
                ]
            },
            'imperial': {
                'name': 'Imperial March',
                'tempo': 100,
                'notes': [
                    {'freq': 392.00, 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 392.00, 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 392.00, 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 311.13, 'duration': 0.375, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 466.16, 'duration': 0.125, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 392.00, 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 311.13, 'duration': 0.375, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 466.16, 'duration': 0.125, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 392.00, 'duration': 1.0, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 587.33, 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 587.33, 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 587.33, 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 622.25, 'duration': 0.375, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 466.16, 'duration': 0.125, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 369.99, 'duration': 0.5, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 311.13, 'duration': 0.375, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 466.16, 'duration': 0.125, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
                    {'freq': 392.00, 'duration': 1.0, 'type': 'triangle', 'amp': 0.8, 'duty': 2.0},
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
    
    def apply_envelope(self, signal, attack_time=0.01, release_time=0.05):
        if len(signal) == 0:
            return signal
        
        envelope = np.ones(len(signal))
        attack_samples = int(attack_time * self.sample_rate)
        release_samples = int(release_time * self.sample_rate)
        
        for i in range(min(attack_samples, len(envelope))):
            envelope[i] = i / attack_samples
        
        for i in range(max(0, len(envelope) - release_samples), len(envelope)):
            envelope[i] = (len(envelope) - i) / release_samples
        
        return signal * envelope
    
    def generate_melody_audio(self):
        if not self.current_melody:
            return np.array([])
        
        print(f"Генерация аудио для мелодии: {self.melody_name}")
        
        total_samples = 0
        for note in self.current_melody:
            note_samples = int(note['duration'] * self.sample_rate)
            total_samples += note_samples
        
        total_samples *= 1
        
        audio_data = np.zeros(total_samples)
        current_pos = 0
        
        for repeat in range(1): 
            for note in self.current_melody:
                note_samples = int(note['duration'] * self.sample_rate)
                
                if current_pos + note_samples > len(audio_data):
                    break
                
                t_note = np.arange(note_samples) * self.dt
                
                note_signal = self.generate_signal(
                    t_note,
                    note['type'],
                    note['freq'],
                    note['amp'],
                    0.0,
                    note['duty']
                )
                

                if note['freq'] > 0:
                    note_signal = self.apply_envelope(note_signal)
                

                audio_data[current_pos:current_pos + note_samples] = note_signal
                current_pos += note_samples
            
            if repeat < 9:
                pause_samples = int(0.5 * self.sample_rate)
                current_pos += pause_samples
        
        print(f"Аудио сгенерировано: {len(audio_data)} сэмплов, {len(audio_data)/self.sample_rate:.1f} секунд")
        return audio_data
    
    def load_melody(self, melody_key):
        if melody_key in self.melodies:
            melody_data = self.melodies[melody_key]
            self.current_melody = melody_data['notes']
            self.melody_name = melody_data['name']
            self.tempo = melody_data['tempo']
            
            self.current_audio_data = self.generate_melody_audio()
            self.current_position = 0
            
            return True
        return False
    
    def start_melody(self):
        if self.current_audio_data is not None:
            self.playing = True
            self.current_position = 0
    
    def stop_melody(self):
        self.playing = False
    
    def restart_melody(self):
        self.current_position = 0
        if not self.playing:
            self.start_melody()
    
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        

        outdata.fill(0)
        
        if self.playing and self.current_audio_data is not None:

            available_samples = len(self.current_audio_data) - self.current_position
            samples_to_copy = min(frames, available_samples)
            
            if samples_to_copy > 0:

                outdata[:samples_to_copy, 0] = self.current_audio_data[
                    self.current_position:self.current_position + samples_to_copy
                ]
                self.current_position += samples_to_copy
            
            if self.current_position >= len(self.current_audio_data):
                self.current_position = 0
    
    def start(self):
        self.stream.start()
    
    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
    
    def get_playback_info(self):
        """Возвращает информацию о воспроизведении"""
        if self.current_audio_data is None:
            return "Нет данных", "0/0", 0.0
        
        total_duration = len(self.current_audio_data) / self.sample_rate
        current_time = self.current_position / self.sample_rate
        progress_percent = (self.current_position / len(self.current_audio_data)) * 100
        
        status = "ИГРАЕТ" if self.playing else "ПАУЗА"
        time_info = f"{current_time:.1f}/{total_duration:.1f}с"
        
        return status, time_info, progress_percent
    
    def print_status(self):
        status, time_info, progress = self.get_playback_info()
        print(f"\r{status} | {self.melody_name} | Время: {time_info} | Прогресс: {progress:.1f}%",
              end='', flush=True)

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
    print("  h - Показать справку")
    print("  q или Esc - Выход")
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
        
        if player.current_audio_data is not None:
            duration = len(player.current_audio_data) / player.sample_rate
            print(f"Общая длительность (с повторами): {duration:.1f} секунд")
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
        
        # Загружаем мелодию по умолчанию
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
                    player.restart_melody()
                
                elif key == '1':
                    player.load_melody('mario')
                    print(f"\nЗагружена: {player.melody_name}")
                elif key == '2':
                    player.load_melody('tetris')
                    print(f"\nЗагружена: {player.melody_name}")
                elif key == '3':
                    player.load_melody('imperial')
                    print(f"\nЗагружена: {player.melody_name}")
                
                elif key == 'i':
                    print_melody_info(player)
                elif key == 'l':
                    print_melody_list(player)
                elif key == 'h':
                    print_help()
                
                elif key == 'q' or key == '\x1b' or key == '\x03':
                    break
            
            player.print_status()
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nОстановка проигрывателя мелодий...")
        player.stop()
        print("Проигрыватель остановлен.")

if __name__ == "__main__":
    main()