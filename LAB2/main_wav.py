"""
Spectral Analyzer
Красивый эквалайзер с полосками в режиме реального времени
"""
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import time
import os
from keyboard_controller import KeyboardController


class WavPlayer:
    """Проигрыватель WAV файлов с FFT анализом"""
    
    def __init__(self, sample_rate=44100, buffer_size=2048):
        """
        Инициализация проигрывателя
        
        Args:
            sample_rate: частота дискретизации
            buffer_size: размер буфера для FFT
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Состояние
        self.running = True
        self.playing = False
        self.audio_data = None
        self.current_position = 0
        self.file_sample_rate = sample_rate
        self.filename = "No file loaded"
        
        # Буфер для FFT анализа
        self.fft_buffer = np.zeros(buffer_size)
        fft_size = buffer_size // 2 + 1
        self.fft_spectrum = np.zeros(fft_size)
        self.frequencies = np.fft.rfftfreq(buffer_size, 1.0 / sample_rate)
        
        # Аудио поток
        self.stream = sd.OutputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=sample_rate,
            blocksize=buffer_size
        )
    
    def load_wav(self, filepath):
        """Загрузка WAV файла"""
        try:
            data, samplerate = sf.read(filepath)
            
            # Конвертируем в моно если стерео
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            self.audio_data = data
            self.file_sample_rate = samplerate
            self.filename = os.path.basename(filepath)
            self.current_position = 0
            
            duration = len(data) / samplerate
            print(f"\n✅ Loaded: {self.filename}")
            print(f"   Sample rate: {samplerate} Hz")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Samples: {len(data)}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading WAV: {e}")
            return False
    
    def audio_callback(self, outdata, frames, time_info, status):
        """Аудио callback для воспроизведения и анализа"""
        if status:
            print(f"Audio status: {status}")
        
        signal = np.zeros(frames)
        
        if self.playing and self.audio_data is not None:
            # Получаем данные из аудио файла
            end_pos = min(self.current_position + frames, len(self.audio_data))
            chunk_size = end_pos - self.current_position
            
            if chunk_size > 0:
                signal[:chunk_size] = self.audio_data[self.current_position:end_pos]
                self.current_position = end_pos
                
                # Обновляем буфер для FFT
                self.fft_buffer = signal.copy()
                
                # Вычисляем FFT
                self.compute_fft()
            
            # Зацикливание
            if self.current_position >= len(self.audio_data):
                self.current_position = 0
        
        outdata[:, 0] = signal.astype(np.float32)
    
    def compute_fft(self):
        """Вычисление FFT спектра"""
        # Применяем оконную функцию
        windowed = self.fft_buffer * np.hanning(len(self.fft_buffer))
        
        # Вычисляем FFT
        fft_data = np.fft.rfft(windowed)
        magnitude = np.abs(fft_data)
        
        # Конвертируем в децибелы
        spectrum = 20 * np.log10(magnitude + 1e-10)
        self.fft_spectrum = spectrum[:len(self.fft_spectrum)]
    
    def start(self):
        self.stream.start()
    
    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
    
    def start_playback(self):
        if self.audio_data is not None:
            self.playing = True
            print("\n▶ Playing...")
    
    def stop_playback(self):
        self.playing = False
        print("\n⏸ Paused")
    
    def restart(self):
        self.current_position = 0
        if not self.playing:
            self.start_playback()
    
    def get_progress(self):
        if self.audio_data is None:
            return 0.0, 0.0
        current_time = self.current_position / self.file_sample_rate
        total_time = len(self.audio_data) / self.file_sample_rate
        return current_time, total_time


class EqualizerVisualizer:
    """Красивый визуализатор-эквалайзер"""
    
    def __init__(self, player, num_bars=32, update_interval=50):
        """
        Инициализация визуализатора
        
        Args:
            player: экземпляр WavPlayer
            num_bars: количество полосок эквалайзера
            update_interval: интервал обновления (мс)
        """
        self.player = player
        self.num_bars = num_bars
        self.update_interval = update_interval
        
        # Группируем частоты в полоски (логарифмически)
        self.freq_ranges = self.create_frequency_ranges()
        
        # Текущие высоты полосок и пики
        self.bar_heights = np.zeros(num_bars)
        self.bar_peaks = np.zeros(num_bars)
        self.peak_hold_times = np.zeros(num_bars)
        
        # Сглаживание (инерция)
        self.smoothing = 0.7
        
        # Настройка matplotlib
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.patch.set_facecolor('#000000')
        self.ax.set_facecolor('#000000')
        
        # Создаём полоски
        self.bars = []
        self.peak_lines = []
        
        bar_width = 1.0 / num_bars * 0.8
        for i in range(num_bars):
            x = i / num_bars
            
            # Основная полоска
            bar = Rectangle((x, 0), bar_width, 0, 
                          facecolor=self.get_bar_color(0),
                          edgecolor='none')
            self.ax.add_patch(bar)
            self.bars.append(bar)
            
            # Линия пика
            peak_line = self.ax.plot([x, x + bar_width], [0, 0],
                                    color='#ff0000', linewidth=3,
                                    solid_capstyle='round')[0]
            self.peak_lines.append(peak_line)
        
        # Настройка осей
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        
        # Текст с информацией
        self.info_text = self.fig.text(
            0.5, 0.95, '', 
            transform=self.fig.transFigure,
            verticalalignment='top',
            horizontalalignment='center',
            fontsize=14,
            color='#ffffff',
            family='monospace',
            weight='bold'
        )
        
        # Анимация
        self.animation = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=update_interval,
            blit=False
        )
    
    def create_frequency_ranges(self):
        freq_min = 100
        freq_max = min(20000, self.player.sample_rate // 2)
        
        
        freq_edges = np.logspace(np.log10(freq_min), np.log10(freq_max), self.num_bars + 1)
        
        print(freq_edges)
        print(len(freq_edges))
        ranges = []
        for i in range(self.num_bars):
            ranges.append((freq_edges[i], freq_edges[i + 1]))
        
        return ranges
    
    def get_bar_color(self, height):
        if height < 0.3:
            r, g, b = 1.0, 0.9, 0.0
        elif height < 0.6:
            r = 1.0
            g = 0.9 - (height - 0.3) / 0.3 * 0.5
            b = 0.0
        else:
            r = 1.0
            g = 0.4 - (height - 0.6) / 0.4 * 0.4
            b = 0.0
        
        return (r, g, b)
    
    def update_plot(self, frame):
        """Обновление визуализации"""
        new_heights = np.zeros(self.num_bars)
        
        for i, (freq_min, freq_max) in enumerate(self.freq_ranges):
            mask = (self.player.frequencies >= freq_min) & (self.player.frequencies < freq_max)
            
            if np.any(mask):
                spectrum_slice = self.player.fft_spectrum[mask]
                
                height = np.max(spectrum_slice)
                height = (height + 80) / 80
                height = np.clip(height, 0, 1)
                
                new_heights[i] = height
        
        self.bar_heights = (self.smoothing * self.bar_heights + 
                           (1 - self.smoothing) * new_heights)
        
        for i in range(self.num_bars):
            if self.bar_heights[i] > self.bar_peaks[i]:
                self.bar_peaks[i] = self.bar_heights[i]
                self.peak_hold_times[i] = 30
            else:
                if self.peak_hold_times[i] > 0:
                    self.peak_hold_times[i] -= 1
                else:
                    self.bar_peaks[i] *= 0.95
        
        bar_width = 1.0 / self.num_bars * 0.8
        
        for i in range(self.num_bars):
            x = i / self.num_bars
            height = self.bar_heights[i]
            
            self.bars[i].set_height(height)
            self.bars[i].set_facecolor(self.get_bar_color(height))
            
            peak_y = self.bar_peaks[i]
            self.peak_lines[i].set_data([x, x + bar_width], [peak_y, peak_y])
        
        status = "▶ PLAYING" if self.player.playing else "⏸ PAUSED"
        current_time, total_time = self.player.get_progress()
        
        info = f"{status}  |  {self.player.filename}  |  {current_time:.1f}s / {total_time:.1f}s"
        self.info_text.set_text(info)
        
        return self.bars + self.peak_lines + [self.info_text]
    
    def show(self):
        """Показать окно визуализации"""
        plt.show()


def print_help():
    """Вывод справки"""
    print("=" * 80)
    print("🎵 AUDIO EQUALIZER - Real-time FFT Visualization 🎵")
    print("=" * 80)
    print("\n📋 Controls:")
    print("  SPACE  - Play/Pause")
    print("  R      - Restart")
    print("  L      - Load WAV file")
    print("  ESC    - Exit")
    print()
    print("✨ Features:")
    print("   • Beautiful bar equalizer with gradient colors")
    print("   • Peak hold indicators")
    print("   • Logarithmic frequency scale (20Hz - 20kHz)")
    print("   • Smooth animations with inertia")
    print("=" * 80)


def main():
    """Главная функция"""
    print_help()
    

    player = WavPlayer(sample_rate=44100, buffer_size=2048)
    keyboard = KeyboardController()
    

    print("\n📂 Enter WAV file path (or drag & drop):")
    try:
        filepath = input("> ").strip().strip('"').strip("'")
        
        if os.path.exists(filepath):
            if not player.load_wav(filepath):
                print("❌ Failed to load WAV file")
                return
        else:
            print(f"❌ File not found: {filepath}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
   
    player.start()

    print("\n🎨 Starting visualizer...")
    print("   Close the window or press ESC to exit")
    visualizer = EqualizerVisualizer(player, num_bars=32, update_interval=50)
    

    player.start_playback()
    

    import threading
    
    def keyboard_handler():
        """Обработчик клавиатуры в отдельном потоке"""
        while player.running:
            key = keyboard.get_key()
            
            if key:
                if key == ' ':
                    if player.playing:
                        player.stop_playback()
                    else:
                        player.start_playback()
                elif key in ['r', 'R']:
                    player.restart()
                elif key in ['l', 'L']:
                    print("\n\n📂 Enter WAV file path:")
                    try:
                        new_file = input("> ").strip().strip('"').strip("'")
                        if os.path.exists(new_file):
                            player.load_wav(new_file)
                        else:
                            print("❌ File not found")
                    except:
                        pass
                elif key in ['\x1b', '\x03']:
                    print("\n👋 Exiting...")
                    player.running = False
                    plt.close('all')
                    break
            
            time.sleep(0.02)
    

    kb_thread = threading.Thread(target=keyboard_handler, daemon=True)
    kb_thread.start()
    
    try:
      
        visualizer.show()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    finally:
        player.stop()
        keyboard.restore_terminal()
        print("\n✅ Analyzer stopped")


if __name__ == "__main__":
    main()