"""
Main Application
Главное приложение для запуска MIDI проигрывателя
"""
import os
import time
from polyphonic_player import PolyphonicPlayer
from keyboard_controller import KeyboardController


def print_banner():
    """Вывод баннера приложения"""
    print("=" * 80)
    print("🎵 POLYPHONIC MIDI PLAYER - Real-time Playback with Multiple Voices 🎵")
    print("=" * 80)


def print_help():
    """Вывод справки по управлению"""
    print("\n📋 Controls:")
    print("  SPACE  - Play/Pause")
    print("  R      - Restart from beginning")
    print("  M      - Load MIDI file")
    print("  J      - Load JSON file (parsed notes)")
    print("  H      - Show this help")
    print("  ESC    - Exit application")
    print()
    print("✨ Features:")
    print("   • Polyphonic playback (multiple notes simultaneously)")
    print("   • Continuous phase tracking (no clicks between notes)")
    print("   • Automatic MIDI parsing")
    print("   • Real-time status display")
    print("=" * 80)


def handle_load_midi(player):
    """
    Обработка загрузки MIDI файла
    
    Args:
        player: экземпляр PolyphonicPlayer
    """
    print("\n\n📂 Enter MIDI file path (or drag & drop):")
    try:
        filepath = input("> ").strip().strip('"').strip("'")
        
        if os.path.exists(filepath):
            if player.load_from_midi(filepath):
                print("✅ MIDI loaded successfully! Press SPACE to play")
            else:
                print("❌ Failed to load MIDI file")
        else:
            print(f"❌ File not found: {filepath}")
    except Exception as e:
        print(f"❌ Error: {e}")


def handle_load_json(player):
    """
    Обработка загрузки JSON файла
    
    Args:
        player: экземпляр PolyphonicPlayer
    """
    print("\n\n📂 Enter JSON file path (parsed notes):")
    try:
        filepath = input("> ").strip().strip('"').strip("'")
        
        if os.path.exists(filepath):
            if player.load_from_json(filepath):
                print("✅ JSON loaded successfully! Press SPACE to play")
            else:
                print("❌ Failed to load JSON file")
        else:
            print(f"❌ File not found: {filepath}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Главная функция приложения"""
    print_banner()
    print_help()
    
    # Инициализация компонентов
    player = PolyphonicPlayer(sample_rate=44100, buffer_size=2048)
    keyboard = KeyboardController()
    
    try:
        # Запуск аудио потока
        player.start()
        print("\n🎵 Player ready! Press M to load a MIDI file")
        print("   Press H for help\n")
        
        # Главный цикл
        while player.running:
            # Получаем нажатие клавиши
            key = keyboard.get_key()
            
            if key:
                # Обработка команд
                if key == ' ':
                    # Play/Pause
                    if player.playing:
                        player.stop_playback()
                    else:
                        player.start_playback()
                
                elif key in ['r', 'R']:
                    # Restart
                    player.restart()
                
                elif key in ['m', 'M']:
                    # Load MIDI
                    handle_load_midi(player)
                
                elif key in ['j', 'J']:
                    # Load JSON
                    handle_load_json(player)
                
                elif key in ['h', 'H']:
                    # Help
                    print("\n")
                    print_help()
                
                elif key in ['\x1b', '\x03']:
                    # ESC or Ctrl+C - Exit
                    print("\n\n👋 Exiting...")
                    break
            
            # Обновление статуса
            player.print_status()
            
            # Небольшая задержка для снижения нагрузки на CPU
            time.sleep(0.02)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Очистка ресурсов
        print("\n🎵 Stopping player...")
        player.stop()
        keyboard.restore_terminal()
        print("✅ Player stopped. Goodbye!")


if __name__ == "__main__":
    main()