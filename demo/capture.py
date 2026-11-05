"""Захват клиентской области и ROI через `mss`; каталоги рантайма и запись кадров.

Модуль не анализирует изображение: только получает кадр, проверяет его пригодность
и умеет сохранять PNG. Любой сбой окна или бэкенда возвращает `None` и причину
в `Grabber.last_state` — исключение наружу не выпускается (задача 2.5).
"""

from __future__ import annotations

import ctypes
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np

import win

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE_ROOT = PROJECT_ROOT / "probe"
FAIL_ROOT = PROJECT_ROOT / "fail"
TRACE_ROOT = PROJECT_ROOT / "trace"

CAPTURE_HZ = 20.0

# Порог «однородно чёрного» кадра: клиент в fullscreen или потеря контекста D3D
# дают почти нулевую дисперсию, тогда как вода с поплавком — нет.
BLACK_MEAN_MAX = 8.0
BLACK_STD_MAX = 4.0

FRAME_OK = "ok"
FRAME_EMPTY = "empty"
FRAME_ZERO = "zero"
FRAME_BLACK = "black"

CONSOLE_CODEPAGE = 65001


def configure_console() -> None:
    """UTF-8 в консоли до первого вывода.

    Кодовая страница cp866 не содержит части символов, которые встречаются в заголовках
    чужих окон, и print падает с UnicodeEncodeError вместо диагностики. Вызывается
    при импорте: все точки входа импортируют этот модуль раньше первого вывода.
    """
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(CONSOLE_CODEPAGE)
    except (AttributeError, OSError):
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


configure_console()


def is_uniform_black(frame: np.ndarray) -> bool:
    """Однородно чёрный кадр: низкое среднее и низкий разброс — клиент не отдаёт картинку."""
    return float(frame.mean()) <= BLACK_MEAN_MAX and float(frame.std()) <= BLACK_STD_MAX


def frame_state(frame: np.ndarray | None) -> str:
    """Пригодность кадра: ok / empty / zero / black."""
    if frame is None:
        return FRAME_EMPTY
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim < 2:
        return FRAME_ZERO
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        return FRAME_ZERO
    if is_uniform_black(frame):
        return FRAME_BLACK
    return FRAME_OK


class Grabber:
    """Обёртка `mss`: кадр клиента и кадр ROI, каждый — BGR `ndarray` или `None`."""

    def __init__(self) -> None:
        # mss.mss() помечен deprecated в 10.2.0 (design.md §6)
        self._sct = mss.MSS()
        self.last_state = FRAME_EMPTY

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> "Grabber":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def grab_region(self, region: win.Region) -> np.ndarray | None:
        try:
            raw = self._sct.grab(region.as_mss())
        except Exception as exc:  # mss бросает разные типы: потеря окна, смена разрешения
            self.last_state = f"{FRAME_EMPTY}: {exc}"
            return None
        frame = np.asarray(raw, dtype=np.uint8)[:, :, :3]
        state = frame_state(frame)
        self.last_state = state
        if state != FRAME_OK:
            return None
        return np.ascontiguousarray(frame)

    def grab_client(self, window: win.GameWindow) -> np.ndarray | None:
        try:
            width, height = win.client_size(window)
            origin_x, origin_y = win.client_origin(window)
        except (OSError, ValueError) as exc:
            self.last_state = f"{FRAME_EMPTY}: {exc}"
            return None
        if width <= 0 or height <= 0:
            self.last_state = FRAME_ZERO
            return None
        return self.grab_region(win.Region(origin_x, origin_y, width, height))

    def grab_roi(
        self,
        window: win.GameWindow,
        fraction_x: tuple[float, float] = win.ROI_FRACTION_X,
        fraction_y: tuple[float, float] = win.ROI_FRACTION_Y,
    ) -> np.ndarray | None:
        try:
            region = win.roi_region(window, fraction_x, fraction_y)
        except (OSError, ValueError) as exc:
            self.last_state = f"{FRAME_EMPTY}: {exc}"
            return None
        return self.grab_region(region)


class Pacer:
    """Держит период захвата, чтобы цикл не выходил за 20 Гц в обе стороны."""

    def __init__(self, hz: float = CAPTURE_HZ) -> None:
        if hz <= 0:
            raise ValueError(f"частота должна быть положительной: {hz}")
        self._period = 1.0 / hz
        self._next = time.monotonic()

    def wait(self) -> None:
        self._next += self._period
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next = time.monotonic()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(root: Path = PROBE_ROOT) -> Path:
    """Каталог одной записи: `probe/<дата-время>/`."""
    return ensure_dir(root / timestamp())


def save_frame(frame: np.ndarray, path: Path) -> bool:
    """PNG по ASCII-пути; False, если записать не удалось."""
    ensure_dir(path.parent)
    return bool(cv2.imwrite(str(path), frame))


def save_fail_frame(
    frame: np.ndarray | None,
    cycle: int,
    moment_ms: int,
    marker: str,
    block: int | None = None,
) -> Path | None:
    """Кадр отказа: `fail/[b<NN>_]cycle<NN>_<мс>_<маркер>.png` (задача 5.5).

    Номер блока входит в имя: без него кадры второго блока затрут кадры первого.
    """
    if frame is None:
        return None
    prefix = f"b{block:02d}_" if block is not None else ""
    path = FAIL_ROOT / f"{prefix}cycle{cycle:02d}_{moment_ms:06d}_{marker}.png"
    return path if save_frame(frame, path) else None
