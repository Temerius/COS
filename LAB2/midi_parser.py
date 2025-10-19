"""
MIDI Parser Module
Парсинг MIDI файлов в JSON формат
"""
import mido
import json


class MIDIParser:
    """Класс для парсинга MIDI файлов"""
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def __init__(self):
        pass
    
    @staticmethod
    def midi_to_freq(midi_note: int) -> float:
        """Конвертация MIDI-номера в частоту (Гц)."""
        return 440.0 * (2 ** ((midi_note - 69) / 12.0))
    
    @classmethod
    def midi_to_note_name(cls, midi_note: int) -> str:
        """Конвертация MIDI-номера в название ноты (например, C#4)."""
        octave = (midi_note // 12) - 1
        note = cls.NOTE_NAMES[midi_note % 12]
        return f"{note}{octave}"
    
    def parse_midi_file(self, file_path: str) -> list:
        """
        Парсинг MIDI файла в список нот
        
        Args:
            file_path: путь к MIDI файлу
            
        Returns:
            list: список словарей с информацией о нотах
        """
        mid = mido.MidiFile(file_path)
        ticks_per_beat = mid.ticks_per_beat
        tempo = 500000
        current_time = 0
        
        active_notes = {}
        parsed_notes = []
        
        for track in mid.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    
                elif msg.type == 'note_on' and msg.velocity > 0:
                    start_time = mido.tick2second(current_time, ticks_per_beat, tempo)
                    active_notes[(msg.note, msg.channel)] = (start_time, msg.velocity)
                    
                elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                    if (msg.note, msg.channel) in active_notes:
                        start_time, velocity = active_notes.pop((msg.note, msg.channel))
                        end_time = mido.tick2second(current_time, ticks_per_beat, tempo)
                        duration = end_time - start_time
                        
                        parsed_notes.append({
                            "note": self.midi_to_note_name(msg.note),
                            "midi_number": msg.note,
                            "frequency": round(self.midi_to_freq(msg.note), 2),
                            "velocity": velocity,
                            "amplitude_norm": round(velocity / 127.0 * 4, 3),
                            "duration": round(duration, 4),
                            "channel": msg.channel,
                            "start_time": round(start_time, 4)
                        })
        
        return parsed_notes
    
    def parse_and_save(self, file_path: str, output_json: str) -> int:
        """
        Парсинг MIDI файла и сохранение в JSON
        
        Args:
            file_path: путь к MIDI файлу
            output_json: путь к выходному JSON файлу
            
        Returns:
            int: количество распознанных нот
        """
        parsed_notes = self.parse_midi_file(file_path)
        
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(parsed_notes, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Сохранено {len(parsed_notes)} нот в {output_json}")
        return len(parsed_notes)


def parse_midi(file_path: str, output_json: str):
    """Обёртка для обратной совместимости со старым кодом"""
    parser = MIDIParser()
    return parser.parse_and_save(file_path, output_json)


if __name__ == "__main__":
    parser = MIDIParser()
    parser.parse_and_save("example.mid", "midi_notes.json")