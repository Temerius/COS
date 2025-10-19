"""
Keyboard Controller Module
Кроссплатформенное управление вводом с клавиатуры
"""
import sys

if sys.platform == "win32":
    import msvcrt
else:
    import termios
    import tty


class KeyboardController:
    """Контроллер для неблокирующего ввода с клавиатуры"""
    
    def __init__(self):
        """Инициализация контроллера"""
        self.old_settings = None
        
        if sys.platform != "win32":
            try:
                # Сохраняем текущие настройки терминала
                self.old_settings = termios.tcgetattr(sys.stdin)
                # Переключаем в raw режим для посимвольного чтения
                tty.setraw(sys.stdin.fileno())
            except Exception as e:
                print(f"⚠️  Warning: Cannot set terminal to raw mode: {e}")
    
    def __del__(self):
        """Восстановление настроек терминала при удалении объекта"""
        self.restore_terminal()
    
    def restore_terminal(self):
        """Восстановление оригинальных настроек терминала"""
        if sys.platform != "win32" and self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass
    
    def get_key(self) -> str | None:
        """
        Неблокирующее чтение клавиши
        
        Returns:
            str | None: нажатая клавиша или None если ничего не нажато
        """
        if sys.platform == "win32":
            return self._get_key_windows()
        else:
            return self._get_key_unix()
    
    def _get_key_windows(self) -> str | None:
        """Чтение клавиши в Windows"""
        if msvcrt.kbhit():
            key = msvcrt.getch()
            
            # Обработка специальных клавиш (стрелки и т.д.)
            if key == b'\xe0':
                key = msvcrt.getch()
                return None
            
            # ESC и Ctrl+C
            elif key in [b'\x1b', b'\x03']:
                return key.decode('utf-8', errors='ignore')
            
            # Обычные клавиши
            else:
                return key.decode('utf-8', errors='ignore')
        
        return None
    
    def _get_key_unix(self) -> str | None:
        """Чтение клавиши в Unix/Linux/Mac"""
        import select
        
        # Проверяем, есть ли данные для чтения
        if select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            
            # Обработка escape-последовательностей (стрелки и т.д.)
            if char == '\x1b':
                # Читаем следующие символы для определения специальной клавиши
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    next_char = sys.stdin.read(1)
                    if next_char == '[':
                        # Стрелки и другие специальные клавиши
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            arrow = sys.stdin.read(1)
                            return None  # Игнорируем стрелки
                return '\x1b'  # Возвращаем ESC
            
            return char
        
        return None
    
    def wait_for_key(self) -> str:
        """
        Блокирующее ожидание нажатия клавиши
        
        Returns:
            str: нажатая клавиша
        """
        while True:
            key = self.get_key()
            if key is not None:
                return key
            import time
            time.sleep(0.01)


if __name__ == "__main__":
    # Тест контроллера
    print("Keyboard Controller Test")
    print("Press keys (ESC to exit):")
    
    controller = KeyboardController()
    
    try:
        while True:
            key = controller.get_key()
            if key:
                if key == '\x1b':
                    print("\nESC pressed, exiting...")
                    break
                print(f"Key pressed: {repr(key)}")
            
            import time
            time.sleep(0.01)
    
    finally:
        controller.restore_terminal()
        print("Terminal restored")