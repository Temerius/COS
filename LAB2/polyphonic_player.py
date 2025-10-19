"""
Polyphonic Player Module
Полифонический проигрыватель MIDI с поддержкой одновременных нот
"""
import numpy as np
import sounddevice as sd
import json
import os
from midi_parser import MIDIParser


class PolyphonicPlayer:
    """Полифонический проигрыватель MIDI с поддержкой одновременных нот"""
    
    def __init__(self, sample_rate=44100, buffer_size=2048*2):
        """
        Инициализация проигрывателя
        
        Args:
            sample_rate: частота дискретизации (Гц)
            buffer_size: размер аудио буфера (сэмплы)
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.dt = 1.0 / sample_rate
        
        # Состояние проигрывателя
        self.running = True
        self.playing = False
        self.global_time = 0.0
        self.playback_start_time = 0.0
        
        # Данные нот из JSON
        self.notes_data = []
        self.song_name = "No song loaded"
        
        # Активные ноты: {note_id: {'frequency': float, 'amplitude': float, 'phase': float}}
        self.active_notes = {}
        self.next_note_index = 0
        
        # Парсер MIDI
        self.midi_parser = MIDIParser()
        
        # Аудио поток
        self.stream = sd.OutputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=sample_rate,
            blocksize=buffer_size
        )
    
    def load_from_json(self, json_path):
        """
        Загрузка нот из JSON файла
        
        Args:
            json_path: путь к JSON файлу с нотами
            
        Returns:
            bool: успешность загрузки
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.notes_data = json.load(f)
            
            # Сортируем по времени начала
            self.notes_data.sort(key=lambda x: x['start_time'])
            
            self.song_name = os.path.basename(json_path)
            self.next_note_index = 0
            self.active_notes = {}
            
            print(f"\n✅ Loaded: {self.song_name}")
            print(f"   Total notes: {len(self.notes_data)}")
            
            # Статистика
            if self.notes_data:
                total_duration = max([n['start_time'] + n['duration'] for n in self.notes_data])
                print(f"   Duration: {total_duration:.2f}s")
            
            return True
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            return False
    
    def load_from_midi(self, midi_path):
        """
        Загрузка и парсинг MIDI файла
        
        Args:
            midi_path: путь к MIDI файлу
            
        Returns:
            bool: успешность загрузки
        """
        try:
            # Создаём временный JSON файл
            json_path = "temp_midi_notes.json"
            print(f"📂 Parsing MIDI: {midi_path}")
            self.midi_parser.parse_and_save(midi_path, json_path)
            
            # Загружаем из JSON
            result = self.load_from_json(json_path)
            
            if result:
                self.song_name = os.path.basename(midi_path)
            
            return result
        except Exception as e:
            print(f"❌ Error loading MIDI: {e}")
            return False
    
    def generate_sine_wave(self, n_samples, frequency, amplitude, phase):
        """
        Генерация синусоиды с сохранением фазы
        
        Args:
            n_samples: количество сэмплов
            frequency: частота (Гц)
            amplitude: амплитуда (0.0 - 1.0)
            phase: начальная фаза (радианы)
            
        Returns:
            tuple: (signal, new_phase)
        """
        if frequency == 0:
            return np.zeros(n_samples), 0.0
        
        t = np.arange(n_samples) * self.dt
        new_phase = phase + 2 * np.pi * frequency * t
        signal = amplitude * np.sin(new_phase)
        
        # Возвращаем сигнал и последнюю фазу
        return signal, new_phase[-1]
    
    def audio_callback(self, outdata, frames, time_info, status):
        """
        Аудио callback с полифонией
        
        Args:
            outdata: выходной буфер
            frames: количество фреймов
            time_info: информация о времени
            status: статус потока
        """
        if status:
            print(f"Audio status: {status}")
        
        signal = np.zeros(frames)
        
        if self.playing and self.notes_data:
            # Текущее время воспроизведения
            playback_time = self.global_time - self.playback_start_time
            
            # Активируем новые ноты
            while self.next_note_index < len(self.notes_data):
                note = self.notes_data[self.next_note_index]
                
                if note['start_time'] <= playback_time:
                    # Добавляем ноту в активные
                    note_id = f"{self.next_note_index}_{note['midi_number']}"
                    self.active_notes[note_id] = {
                        'frequency': note['frequency'],
                        'amplitude': note['amplitude_norm'] * 0.3,  # Снижаем громкость
                        'phase': 0.0,
                        'end_time': note['start_time'] + note['duration'],
                        'note_name': note['note']
                    }
                    self.next_note_index += 1
                else:
                    break
            
            # Генерируем сигнал от всех активных нот
            notes_to_remove = []
            
            for note_id, note_info in self.active_notes.items():
                # Проверяем, не закончилась ли нота
                if playback_time >= note_info['end_time']:
                    notes_to_remove.append(note_id)
                    continue
                
                # Генерируем сигнал для этой ноты
                note_signal, new_phase = self.generate_sine_wave(
                    frames,
                    note_info['frequency'],
                    note_info['amplitude'],
                    note_info['phase']
                )
                
                # Применяем envelope для сглаживания начала и конца
                note_duration = note_info['end_time'] - (playback_time - frames * self.dt)
                envelope = np.ones(frames)
                
                # Attack (первые 5ms)
                attack_samples = int(0.005 * self.sample_rate)
                if note_info['phase'] < 2 * np.pi:  # Только в начале ноты
                    for i in range(min(attack_samples, frames)):
                        envelope[i] = i / attack_samples
                
                # Release (последние 10ms)
                release_samples = int(0.01 * self.sample_rate)
                samples_to_end = int(note_duration * self.sample_rate)
                if samples_to_end < release_samples:
                    for i in range(frames):
                        if samples_to_end - i < release_samples and samples_to_end - i >= 0:
                            envelope[i] = (samples_to_end - i) / release_samples
                
                note_signal *= envelope
                
                # Обновляем фазу
                note_info['phase'] = new_phase
                
                # Добавляем к общему сигналу (ПОЛИФОНИЯ!)
                signal += note_signal
            
            # Удаляем завершённые ноты
            for note_id in notes_to_remove:
                del self.active_notes[note_id]
            
            # Проверка на завершение композиции
            if self.next_note_index >= len(self.notes_data) and len(self.active_notes) == 0:
                self.playing = False
                print("\n\n🎵 Playback finished!")
        
        # Нормализация для предотвращения клиппинга
        max_val = np.max(np.abs(signal))
        if max_val > 1.0:
            signal = signal / max_val
        
        outdata[:, 0] = signal.astype(np.float32)
        self.global_time += frames * self.dt
    
    def start(self):
        """Запуск аудио потока"""
        self.stream.start()
    
    def stop(self):
        """Остановка аудио потока"""
        self.running = False
        self.stream.stop()
        self.stream.close()
    
    def start_playback(self):
        """Начать воспроизведение"""
        if self.notes_data:
            self.playing = True
            self.next_note_index = 0
            self.active_notes = {}
            self.playback_start_time = self.global_time
            print("\n▶ Playing...")
    
    def stop_playback(self):
        """Остановить воспроизведение"""
        self.playing = False
        self.active_notes = {}
        print("\n⏸ Paused")
    
    def restart(self):
        """Перезапустить воспроизведение"""
        self.next_note_index = 0
        self.active_notes = {}
        self.playback_start_time = self.global_time
        if not self.playing:
            self.start_playback()
    
    def print_status(self):
        """Вывод текущего статуса проигрывателя"""
        status = "▶ PLAYING" if self.playing else "⏸ PAUSED"
        
        if self.notes_data:
            playback_time = self.global_time - self.playback_start_time if self.playing else 0
            total_duration = max([n['start_time'] + n['duration'] for n in self.notes_data]) if self.notes_data else 0
            progress = f"{playback_time:.1f}s / {total_duration:.1f}s"
            
            active_count = len(self.active_notes)
            active_info = f" | 🎵 Active: {active_count}"
            
            # Показываем названия активных нот
            if self.active_notes:
                note_names = list(set([info['note_name'] for info in self.active_notes.values()]))[:5]
                notes_str = ", ".join(note_names)
                if len(note_names) > 5:
                    notes_str += "..."
                active_info += f" [{notes_str}]"
        else:
            progress = "No song"
            active_info = ""
        
        print(f"\r{status} | {self.song_name} | {progress}{active_info}          ",
              end='', flush=True)
    
    def get_stats(self):
        """
        Получить статистику о текущем состоянии
        
        Returns:
            dict: словарь со статистикой
        """
        return {
            'playing': self.playing,
            'song_name': self.song_name,
            'total_notes': len(self.notes_data),
            'active_notes': len(self.active_notes),
            'current_note_index': self.next_note_index,
            'playback_time': self.global_time - self.playback_start_time if self.playing else 0
        }


if __name__ == "__main__":
    # Тест проигрывателя
    print("Testing Polyphonic Player...")
    player = PolyphonicPlayer()
    player.start()
    
    import time
    print("Player started. Press Ctrl+C to stop.")
    
    try:
        while True:
            player.print_status()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
        print("\nPlayer stopped.")