"""Окно клиента WoW: поиск, геометрия клиентской области, foreground и стоп.

Модуль знает только про окно и координаты: захвата кадров и инъекции ввода здесь нет
(design.md §3 — три тонких адаптера). DPI-awareness объявляется при импорте, то есть
до первого обращения к окну (задача 2.2): иначе Windows отдаёт отмасштабированную
геометрию и клик уезжает от кадра.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass

GAME_WINDOW_TITLE = "World of Warcraft"
GAME_WINDOW_CLASSES = ("GxWindowClass", "GxWindowClassD3d")
# Доли ROI по осям раздельно (design.md §10): по вертикали окно сдвинуто в воду,
# иначе поплавок (y = 0.287 клиента) и всплеск оказываются вне кадра.
ROI_FRACTION_X = (0.35, 0.65)
ROI_FRACTION_Y = (0.20, 0.50)
VK_F12 = 0x7B
START_COUNTDOWN = 5

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

user32.EnumWindows.argtypes = (WNDENUMPROC, wintypes.LPARAM)
user32.EnumWindows.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.GetClassNameW.restype = ctypes.c_int
user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
user32.GetClientRect.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
user32.ClientToScreen.restype = wintypes.BOOL
user32.GetForegroundWindow.argtypes = ()
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
user32.GetAsyncKeyState.restype = ctypes.c_short

kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = (
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.LPWSTR,
    ctypes.POINTER(wintypes.DWORD),
)
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
kernel32.CloseHandle.restype = wintypes.BOOL


class WindowNotFound(RuntimeError):
    """Top-level окно клиента не найдено: захват и ввод недопустимы."""


@dataclass(frozen=True)
class GameWindow:
    """Найденное окно клиента: дескриптор, класс и владелец процесса."""

    hwnd: int
    title: str
    class_name: str
    pid: int
    exe: str


@dataclass(frozen=True)
class Region:
    """Прямоугольник в экранных координатах; формат совместим с `mss`."""

    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict[str, int]:
        """Регион в виде словаря, который принимает `mss.MSS.grab`."""
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Точка в пикселях региона -> экранные координаты (единый хелпер, design §5)."""
        return int(round(self.left + x)), int(round(self.top + y))


def _declare_dpi_awareness() -> str:
    """Объявляет DPI-awareness процесса. Возвращает принятый режим."""
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2; доступно с Windows 8.1
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return "per-monitor"
    except (AttributeError, OSError):
        pass
    try:
        if user32.SetProcessDPIAware():
            return "system"
    except (AttributeError, OSError):
        pass
    return "none"


DPI_AWARENESS = _declare_dpi_awareness()


def _window_text(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def _class_name(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, 256)
    return buffer.value


def process_exe(pid: int) -> str:
    """Полный путь к exe процесса; пустая строка, если доступ закрыт."""
    if pid <= 0:
        return ""
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return buffer.value
        return ""
    finally:
        kernel32.CloseHandle(handle)


def iter_top_level_windows() -> list[GameWindow]:
    """Все top-level окна с непустым заголовком."""
    windows: list[GameWindow] = []

    @WNDENUMPROC
    def collect(hwnd, _lparam):
        title = _window_text(hwnd)
        if not title:
            return True
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        windows.append(
            GameWindow(
                hwnd=int(hwnd),
                title=title,
                class_name=_class_name(hwnd),
                pid=int(pid.value),
                exe=process_exe(int(pid.value)),
            )
        )
        return True

    user32.EnumWindows(collect, 0)
    return windows


def find_game_window(
    title: str = GAME_WINDOW_TITLE,
    classes: tuple[str, ...] = GAME_WINDOW_CLASSES,
) -> GameWindow | None:
    """Top-level окно клиента по заголовку с проверкой одного из известных классов."""
    expected = title.strip().casefold()
    for window in iter_top_level_windows():
        if window.class_name not in classes:
            continue
        if window.title.strip().casefold() == expected:
            return window
    return None


def require_game_window() -> GameWindow:
    window = find_game_window()
    if window is None:
        raise WindowNotFound(
            f"top-level окно «{GAME_WINDOW_TITLE}» с классом {'/'.join(GAME_WINDOW_CLASSES)} "
            "не найдено: клиент не запущен либо заголовок/класс отличаются"
        )
    return window


def describe(window: GameWindow) -> str:
    return (
        f"hwnd={window.hwnd:#x} class={window.class_name} "
        f"pid={window.pid} exe={window.exe or '<недоступен>'}"
    )


def client_size(window: GameWindow) -> tuple[int, int]:
    """Размер клиентской области в пикселях (без рамки и заголовка)."""
    rect = wintypes.RECT()
    if not user32.GetClientRect(window.hwnd, ctypes.byref(rect)):
        raise OSError(ctypes.get_last_error(), "GetClientRect")
    return rect.right - rect.left, rect.bottom - rect.top


def client_origin(window: GameWindow) -> tuple[int, int]:
    """Экранная позиция левого верхнего пикселя клиентской области."""
    point = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(window.hwnd, ctypes.byref(point)):
        raise OSError(ctypes.get_last_error(), "ClientToScreen")
    return point.x, point.y


def _check_axis(size: int, fraction: tuple[float, float]) -> None:
    low, high = fraction
    if size <= 0:
        raise ValueError(f"нулевой размер клиентской области: {size}")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"доли ROI должны возрастать в пределах 0..1: {fraction}")


def roi_box(
    width: int,
    height: int,
    fraction_x: tuple[float, float] = ROI_FRACTION_X,
    fraction_y: tuple[float, float] = ROI_FRACTION_Y,
) -> tuple[int, int, int, int]:
    """Клиентский прямоугольник ROI по долям: (left, top, width, height).

    Чистая функция: считается от размера клиентской области, поэтому смена позиции
    окна и расхождение `gxResolution` с реальным размером не ломают расчёт (design §5).
    """
    _check_axis(width, fraction_x)
    _check_axis(height, fraction_y)
    left = int(width * fraction_x[0])
    right = int(width * fraction_x[1])
    top = int(height * fraction_y[0])
    bottom = int(height * fraction_y[1])
    return left, top, right - left, bottom - top


def roi_region(
    window: GameWindow,
    fraction_x: tuple[float, float] = ROI_FRACTION_X,
    fraction_y: tuple[float, float] = ROI_FRACTION_Y,
) -> Region:
    """ROI в экранных координатах: прямоугольник из `roi_box`, сдвинутый на позицию клиента."""
    width, height = client_size(window)
    left, top, roi_width, roi_height = roi_box(width, height, fraction_x, fraction_y)
    origin_x, origin_y = client_origin(window)
    return Region(origin_x + left, origin_y + top, roi_width, roi_height)


def foreground_window() -> tuple[int, str]:
    """Дескриптор и заголовок активного окна — для диагностики отказов."""
    hwnd = int(user32.GetForegroundWindow() or 0)
    return hwnd, _window_text(hwnd) if hwnd else ""


def is_foreground(window: GameWindow) -> bool:
    return int(user32.GetForegroundWindow() or 0) == window.hwnd


def stop_requested(vk: int = VK_F12) -> bool:
    """Аварийный стоп: F12 читается глобально, независимо от фокуса приложения."""
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def start_countdown(window: GameWindow, seconds: int = START_COUNTDOWN) -> bool:
    """Стартовый контракт: отсчёт и проверка фокуса. True — прогон можно начинать."""
    for remaining in range(seconds, 0, -1):
        if stop_requested():
            print("F12 до старта: прогон отменён пользователем")
            return False
        print(f"старт через {remaining} с — переключитесь в окно «{GAME_WINDOW_TITLE}»")
        time.sleep(1.0)
    if not is_foreground(window):
        hwnd, title = foreground_window()
        print(
            f"фокус не в клиенте: активно окно {hwnd:#x} «{title}», "
            f"а не {window.hwnd:#x}; прогон не начинается"
        )
        return False
    return True
