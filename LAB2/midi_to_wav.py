"""
MIDI to WAV Converter
Конвертация MIDI файлов в WAV без реального времени
"""
import numpy as np
import soundfile as sf
import json
import os
from midi_parser import MIDIParser


class MIDIToWavConverter:
    """Конвертер MIDI в WAV с полифонией"""
    
    def __init__(self, sample_rate=44100):
        """
        Инициализация конвертера
        
        Args:
            sample_rate: частота дискретизации выходного WAV
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.midi_parser = MIDIParser()
    
    def generate_sine_wave(self, duration, frequency, amplitude, phase=0.0):
        """
        Генерация синусоиды
        
        Args:
            duration: длительность (секунды)
            frequency: частота (Гц)
            amplitude: амплитуда (0.0 - 1.0)
            phase: начальная фаза (радианы)
            
        Returns:
            tuple: (signal, end_phase)
        """
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) * self.dt
        
        signal_phase = phase + 2 * np.pi * frequency * t
        signal = amplitude * np.sin(signal_phase)
        
        # Применяем envelope для сглаживания
        envelope = np.ones(n_samples)
        
        # Attack (первые 5ms)
        attack_samples = int(0.005 * self.sample_rate)
        if attack_samples > 0 and attack_samples < n_samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Release (последние 10ms)
        release_samples = int(0.01 * self.sample_rate)
        if release_samples > 0 and release_samples < n_samples:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        signal *= envelope
        
        return signal, signal_phase[-1] if n_samples > 0 else phase
    
    def convert_from_json(self, json_path, output_wav):
        """
        Конвертация из JSON (распарсенного MIDI) в WAV
        
        Args:
            json_path: путь к JSON файлу с нотами
            output_wav: путь к выходному WAV файлу
            
        Returns:
            bool: успешность конвертации
        """
        try:
            # Загружаем ноты
            with open(json_path, 'r', encoding='utf-8') as f:
                notes_data = json.load(f)
            
            if not notes_data:
                print("❌ No notes found in JSON")
                return False
            
            # Сортируем по времени начала
            notes_data.sort(key=lambda x: x['start_time'])
            
            # Определяем общую длительность
            total_duration = max([n['start_time'] + n['duration'] for n in notes_data])
            total_samples = int(total_duration * self.sample_rate)
            
            print(f"\n📊 Converting:")
            print(f"   Notes: {len(notes_data)}")
            print(f"   Duration: {total_duration:.2f}s")
            print(f"   Samples: {total_samples}")
            
            # Создаём пустой аудио буфер
            audio_buffer = np.zeros(total_samples)
            
            # Рендерим каждую ноту
            print(f"\n🎵 Rendering notes...")
            for i, note in enumerate(notes_data):
                if (i + 1) % 100 == 0:
                    progress = (i + 1) / len(notes_data) * 100
                    print(f"   Progress: {progress:.1f}% ({i + 1}/{len(notes_data)})")
                
                # Вычисляем позицию в буфере
                start_sample = int(note['start_time'] * self.sample_rate)
                duration = note['duration']
                frequency = note['frequency']
                amplitude = note['amplitude_norm'] * 0.3  # Снижаем громкость
                
                # Генерируем сигнал ноты
                signal, _ = self.generate_sine_wave(duration, frequency, amplitude)
                
                # Добавляем в буфер (полифония!)
                end_sample = start_sample + len(signal)
                if end_sample <= total_samples:
                    audio_buffer[start_sample:end_sample] += signal
                else:
                    # Обрезаем если нота выходит за границы
                    available = total_samples - start_sample
                    audio_buffer[start_sample:] += signal[:available]
            
            # Нормализация для предотвращения клиппинга
            max_val = np.max(np.abs(audio_buffer))
            if max_val > 1.0:
                print(f"   Normalizing: {max_val:.2f} → 1.0")
                audio_buffer = audio_buffer / max_val
            
            # Сохраняем в WAV
            print(f"\n💾 Saving to: {output_wav}")
            sf.write(output_wav, audio_buffer, self.sample_rate)
            
            file_size = os.path.getsize(output_wav) / (1024 * 1024)
            print(f"✅ Success! File size: {file_size:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during conversion: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def convert_from_midi(self, midi_path, output_wav):
        """
        Конвертация MIDI → JSON → WAV
        
        Args:
            midi_path: путь к MIDI файлу
            output_wav: путь к выходному WAV файлу
            
        Returns:
            bool: успешность конвертации
        """
        try:
            # Парсим MIDI
            print(f"📂 Parsing MIDI: {midi_path}")
            temp_json = "temp_conversion.json"
            self.midi_parser.parse_and_save(midi_path, temp_json)
            
            # Конвертируем в WAV
            result = self.convert_from_json(temp_json, output_wav)
            
            # Удаляем временный JSON
            if os.path.exists(temp_json):
                os.remove(temp_json)
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def print_banner():
    """Баннер программы"""
    print("=" * 80)
    print("🎵 MIDI to WAV Converter - Полифонический рендеринг")
    print("=" * 80)


def main():
    """Главная функция"""
    print_banner()
    
    print("\n📋 Выберите режим:")
    print("  1 - Конвертировать MIDI файл")
    print("  2 - Конвертировать JSON (распарсенный MIDI)")
    print("  3 - Пакетная конвертация (все MIDI в папке)")
    
    try:
        choice = input("\nВыбор (1/2/3): ").strip()
        
        converter = MIDIToWavConverter(sample_rate=44100)
        
        if choice == '1':
            # Одиночный MIDI файл
            midi_path = input("\n📂 Путь к MIDI файлу: ").strip().strip('"').strip("'")
            
            if not os.path.exists(midi_path):
                print("❌ Файл не найден!")
                return
            
            # Генерируем имя выходного файла
            output_wav = os.path.splitext(midi_path)[0] + ".wav"
            
            custom = input(f"💾 Сохранить как [{output_wav}] или введите другое имя: ").strip()
            if custom:
                output_wav = custom
            
            converter.convert_from_midi(midi_path, output_wav)
        
        elif choice == '2':
            # JSON файл
            json_path = input("\n📂 Путь к JSON файлу: ").strip().strip('"').strip("'")
            
            if not os.path.exists(json_path):
                print("❌ Файл не найден!")
                return
            
            output_wav = os.path.splitext(json_path)[0] + ".wav"
            
            custom = input(f"💾 Сохранить как [{output_wav}] или введите другое имя: ").strip()
            if custom:
                output_wav = custom
            
            converter.convert_from_json(json_path, output_wav)
        
        elif choice == '3':
            # Пакетная конвертация
            folder = input("\n📂 Путь к папке с MIDI файлами: ").strip().strip('"').strip("'")
            
            if not os.path.isdir(folder):
                print("❌ Папка не найдена!")
                return
            
            # Находим все MIDI файлы
            midi_files = [f for f in os.listdir(folder) if f.lower().endswith(('.mid', '.midi'))]
            
            if not midi_files:
                print("❌ MIDI файлы не найдены в папке!")
                return
            
            print(f"\n🎵 Найдено {len(midi_files)} MIDI файлов")
            confirm = input("Начать конвертацию? (y/n): ").strip().lower()
            
            if confirm != 'y':
                print("Отменено")
                return
            
            success_count = 0
            for i, midi_file in enumerate(midi_files, 1):
                print(f"\n{'='*80}")
                print(f"[{i}/{len(midi_files)}] {midi_file}")
                print(f"{'='*80}")
                
                midi_path = os.path.join(folder, midi_file)
                output_wav = os.path.splitext(midi_path)[0] + ".wav"
                
                if converter.convert_from_midi(midi_path, output_wav):
                    success_count += 1
            
            print(f"\n{'='*80}")
            print(f"✅ Готово! Успешно: {success_count}/{len(midi_files)}")
            print(f"{'='*80}")
        
        else:
            print("❌ Неверный выбор!")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()