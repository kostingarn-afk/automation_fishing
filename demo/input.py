"""Инъекция ввода: перемещение курсора и right down/up через `SendInput`.

Основной путь — `SendInput` с абсолютными координатами (design.md §7): клиент читает
реальную позицию курсора, поэтому координаты в `WM_*` сообщениях для мира не работают.
Ветка `mouse_event` оставлена только для probe-скрипта (задача 3.3).

Любая отправка требует подтверждённого foreground: инъекция в чужое окно — необратимое
действие с непредсказуемыми последствиями (задача 3.4).
"""

from __future__ import annotations

import ctypes
import random
import time
from ctypes import wintypes

import win

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

KEYEVENTF_KEYUP = 0x0002

# Клавиша прыжка — именованный ключ наравне с символами (`send_key`).
VK_SPACE = 0x20
KEY_NAMES = {"space": VK_SPACE}
JUMP_KEY = "space"
# Удержание клавиши прыжка: `down` и `up` одним пакетом клиент на 60 FPS не видит
# (как `CLICK_HOLD` у мыши). Для каста диапазон не используется — там путь проверен.
KEY_HOLD = (0.06, 0.12)

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# Пауза курсор -> клик (план §4.3). Диапазон задан спекой задачи 3.2.
STRIKE_DELAY = (0.15, 0.25)
# Удержание правой кнопки: down и up в одном пакете SendInput клиент на 60 FPS не видит.
# Диапазон «плавный», значение выбирается случайно — как у живой руки.
CLICK_HOLD = (0.06, 0.20)
# Перевод курсора на поплавок: мгновенный скачок SetCursorPos — признак инъекции,
# поэтому движение разбивается на шаги за случайные 0.10–0.30 с.
MOVE_DURATION = (0.10, 0.30)
MOVE_STEP_SECONDS = 0.012


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _InputUnion(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _InputUnion))


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
user32.SetCursorPos.restype = wintypes.BOOL
user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
user32.GetCursorPos.restype = wintypes.BOOL
user32.mouse_event.argtypes = (
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.ULONG),
)
user32.mouse_event.restype = None
user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
user32.GetSystemMetrics.restype = ctypes.c_int


class InputBlocked(RuntimeError):
    """Ввод не отправлен: окно клиента не является активным.

    `frame` — последний кадр ROI, если отказ случился уже в цикле: по нему сохраняется
    кадр отказа, как и для `CaptureFailed`.
    """

    def __init__(self, message: str, frame=None) -> None:
        super().__init__(message)
        self.frame = frame


def require_foreground(window: win.GameWindow) -> None:
    """Отказ от инъекции, если клиент не в foreground (задача 3.4)."""
    if win.is_foreground(window):
        return
    hwnd, title = win.foreground_window()
    message = (
        f"ввод заблокирован: активно окно {hwnd:#x} «{title}», "
        f"а не клиент {window.hwnd:#x}"
    )
    print(message)
    raise InputBlocked(message)


def _virtual_screen() -> tuple[int, int, int, int]:
    metrics = (SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN)
    left, top, width, height = (user32.GetSystemMetrics(metric) for metric in metrics)
    return left, top, width, height


def normalize(x: int, y: int) -> tuple[int, int]:
    """Экранные пиксели -> нормализованные 0..65535 для `MOUSEEVENTF_ABSOLUTE`."""
    left, top, width, height = _virtual_screen()
    nx = int(round((int(x) - left) * 65535 / max(width - 1, 1)))
    ny = int(round((int(y) - top) * 65535 / max(height - 1, 1)))
    return max(0, min(nx, 65535)), max(0, min(ny, 65535))


def _send(inputs: list[INPUT]) -> bool:
    if not inputs:
        return False
    array = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), array, ctypes.sizeof(INPUT))
    return sent == len(inputs)


def move_cursor(x: int, y: int) -> bool:
    return bool(user32.SetCursorPos(int(x), int(y)))


def cursor_position() -> tuple[int, int]:
    point = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        return 0, 0
    return int(point.x), int(point.y)


def move_cursor_smooth(
    x: int, y: int, duration: tuple[float, float] = MOVE_DURATION
) -> float:
    """Переводит курсор в точку за случайные 0.10–0.30 с, с замедлением к концу.

    Мгновенный скачок `SetCursorPos` — заметный признак инъекции, поэтому движение
    разбивается на шаги. Возвращает фактическую длительность, с.
    """
    start_x, start_y = cursor_position()
    span = random.uniform(duration[0], duration[1])
    steps = max(2, int(span / MOVE_STEP_SECONDS))
    for step in range(1, steps + 1):
        progress = step / steps
        eased = 1.0 - (1.0 - progress) ** 2
        user32.SetCursorPos(
            int(round(start_x + (x - start_x) * eased)),
            int(round(start_y + (y - start_y) * eased)),
        )
        time.sleep(span / steps)
    return span


def _hold_button() -> float:
    """Случайное удержание кнопки из диапазона `CLICK_HOLD`."""
    pause = random.uniform(CLICK_HOLD[0], CLICK_HOLD[1])
    time.sleep(pause)
    return pause


def send_right_click(x: int, y: int) -> bool:
    """Ветка (a): `SendInput` с абсолютными координатами в виртуальном рабочем столе.

    `down` и `up` разнесены по времени: в одном пакете клиент их не различает.
    """
    nx, ny = normalize(x, y)
    move_flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    if not _send(
        [
            INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(nx, ny, 0, move_flags, 0, None)),
            INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_RIGHTDOWN, 0, None)),
        ]
    ):
        return False
    _hold_button()
    return _send([INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_RIGHTUP, 0, None))])


def send_right_click_legacy(x: int, y: int) -> bool:
    """Ветка (b): `SetCursorPos` + `mouse_event` — устаревший путь, только для probe."""
    move_cursor_smooth(x, y)
    user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, None)
    _hold_button()
    user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, None)
    return True


def send_key(key: str, hold: float = 0.0) -> bool:
    """Нажатие клавиши: символ («1»..«9», «a»..«z») или именованный ключ `KEY_NAMES`.

    `hold > 0` разносит `down` и `up` по времени двумя вызовами `SendInput`:
    одномоментные события клиент на 60 FPS не различает. Каст (`hold=0`)
    по-прежнему уходит одним пакетом — его тайминги подобраны на клиенте.
    """
    if not key:
        raise ValueError("пустое имя клавиши")
    vk = KEY_NAMES.get(key.lower())
    if vk is None:
        if len(key) != 1:
            raise ValueError(f"ожидается один символ или именованный ключ, получено {key!r}")
        vk = ord(key.upper())
        if not 0x30 <= vk <= 0x5A:
            raise ValueError(f"неподдерживаемая клавиша {key!r}")
    down = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(vk, 0, 0, 0, None))
    up = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP, 0, None))
    if hold <= 0:
        return _send([down, up])
    if not _send([down]):
        return False
    time.sleep(hold)
    if _send([up]):
        return True
    # `up` не прошёл: повтор дешевле, чем оставить клавишу логически нажатой
    return _send([up])


def cast(window: win.GameWindow, key: str) -> None:
    """Каст заклинания Fishing клавишей `cast_key`."""
    require_foreground(window)
    if not send_key(key):
        raise OSError(ctypes.get_last_error(), "SendInput: каст не отправлен")


def jump(window: win.GameWindow) -> None:
    """Прыжок: `JUMP_KEY` с удержанием `KEY_HOLD`.

    Фокус проверяется до отправки: инъекция в чужое окно — необратимое действие.
    `InputBlocked` — окно не в фокусе, `OSError` — `SendInput` не принял нажатие;
    оба отказа обрабатывает вызывающий код, цикл из-за них не проваливается.
    """
    require_foreground(window)
    hold = random.uniform(KEY_HOLD[0], KEY_HOLD[1])
    if not send_key(JUMP_KEY, hold=hold):
        raise OSError(ctypes.get_last_error(), "SendInput: прыжок не отправлен")


def strike(
    window: win.GameWindow,
    point: tuple[int, int],
    delay: tuple[float, float] = STRIKE_DELAY,
    sender=None,
) -> float:
    """Наводит курсор, выдерживает паузу и кликает правой кнопкой.

    `sender` — способ отправки клика: по умолчанию `SendInput` с абсолютными координатами,
    `send_right_click_legacy` подставляет probe-скрипт (задача 3.3).
    Возвращает фактическую паузу в секундах: её значение попадает в трассу,
    поэтому задержка `strike - burst` проверяема (задача 3.2).
    """
    send = send_right_click if sender is None else sender
    require_foreground(window)
    x, y = point
    moved = move_cursor_smooth(x, y)
    pause = random.uniform(delay[0], delay[1])
    time.sleep(pause)
    # фокус мог уйти за паузу — проверяем непосредственно перед кликом (design §9)
    require_foreground(window)
    if not send(x, y):
        raise OSError(ctypes.get_last_error(), "SendInput: подсечка не отправлена")
    print(f"  курсор переведён за {moved:.2f} с, пауза перед кликом {pause:.2f} с")
    return pause
