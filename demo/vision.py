"""Чистые функции детекции: яркие кластеры, всплеск поклёвки, присутствие поплавка.

Ни `mss`, ни `ctypes`, ни файловой системы, ни состояния между вызовами (design.md §3):
на вход — `ndarray`, на выход — координаты и bool. Поэтому функции можно вызывать
на сохранённых кадрах при офлайн-подборе порога (задача 4.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Значения по умолчанию; рабочие подбираются офлайн по записи канала (задача 4.5).
BRIGHT_THRESHOLD = 150
MIN_AREA = 10

# Всплеск — локальный рост яркости у поплавка (design.md §10): глобальный порог
# всплеск не отделяет, пена занимает ~20 px и живёт ~100 мс при шуме воды 9–15 px.
# Абсолютные порог и прирост подбираются офлайн — они живут в `loop10` (`BURST_THRESHOLD`,
# `SPLASH_RISE`), здесь остаётся только форма окна и относительный критерий роста.
SPLASH_WINDOW_HALF = 15
SPLASH_RISE_RATIO = 1.15

# Поплавок красно-белый; на ROI 576x324 занимает единицы-десятки пикселей.
# Пороги измерены на реальных кадрах (design.md §10): поплавок area 51–66, Vmean 145,
# вода V ≈ 84, красный берег V ≈ 77.
# Оттенок красного гуляет по сцене: 0–10 в «береговой» сцене и 11–13 в водной
# (замер: BGR (26,63,121) → H 11–13, S 200, V 121), поэтому верхняя граница — 15.
RED_HUE_MAX = 15
BOBBER_MIN_AREA = 8
BOBBER_MAX_AREA = 400
BOBBER_SATURATION_MIN = 90
BOBBER_VALUE_MIN = 100
BOBBER_RADIUS = 40

# Захват поплавка (не верификация!) требует строгого признака: красный поплавок в воде
# насыщеннее любого блика (S 188…204, V 105…206, area 15…66) и держится в своей полосе
# оттенка, тогда как мягкий фильтр даёт в кадре 17–35 кластеров-бликов, и трекер уходил
# на воду (циклы 5, 9, 10 прогона 22:03). На 12 кадрах водной сцены строгий пресет даёт
# ровно один кластер — поплавок, и ни одного на кадрах, где поплавка нет.
# Для `bobber_present` пороги остаются мягкими: строгость там давала бы ложные `OK`.
TRACK_HUE_MAX = 12
TRACK_MIN_AREA = 12
TRACK_MAX_AREA = 200
TRACK_SATURATION_MIN = 170
TRACK_VALUE_MIN = 105


@dataclass(frozen=True)
class Blob:
    """Кластер: центр описывающего прямоугольника в пикселях кадра и площадь в пикселях.

    Для компактных кластеров (поплавок, всплеск) центр bbox совпадает с центроидом маски;
    для вытянутых областей они расходятся, поэтому в отчётах уточняется, что имеется в виду.
    """

    cx: float
    cy: float
    area: int


def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR/BGRA/gray -> gray без копирования, если кадр уже одноканальный."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def find_bright_blobs(
    img: np.ndarray,
    threshold: int = BRIGHT_THRESHOLD,
    min_area: int = MIN_AREA,
) -> list[Blob]:
    """Связные области ярче `threshold` площадью не меньше `min_area`.

    Результат отсортирован по убыванию площади; кластеры меньше `min_area`
    отбрасываются как шум.
    """
    gray = to_gray(img)
    _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    blobs: list[Blob] = []
    for index in range(1, count):  # 0 — фон
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        left = int(stats[index, cv2.CC_STAT_LEFT])
        top = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        blobs.append(Blob(cx=left + width / 2.0, cy=top + height / 2.0, area=area))
    blobs.sort(key=lambda blob: blob.area, reverse=True)
    return blobs


def _red_mask(
    roi: np.ndarray,
    saturation_min: int,
    value_min: int,
    hue_max: int = RED_HUE_MAX,
) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_a = np.array([0, saturation_min, value_min], np.uint8)
    upper_a = np.array([hue_max, 255, 255], np.uint8)
    lower_b = np.array([170, saturation_min, value_min], np.uint8)
    upper_b = np.array([180, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower_a, upper_a) | cv2.inRange(hsv, lower_b, upper_b)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def find_red_candidates(
    roi: np.ndarray,
    min_area: int = BOBBER_MIN_AREA,
    max_area: int = BOBBER_MAX_AREA,
    saturation_min: int = BOBBER_SATURATION_MIN,
    value_min: int = BOBBER_VALUE_MIN,
    hue_max: int = RED_HUE_MAX,
) -> list[Blob]:
    """Красные кластеры подходящей площади и яркости, крупные первыми.

    Ограничение площади сверху отсекает рельеф и доспех: поплавок — мелкий объект.
    """
    if roi is None or roi.size == 0 or roi.ndim != 3 or roi.shape[2] < 3:
        return []
    mask = _red_mask(roi, saturation_min, value_min, hue_max)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    value = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]

    candidates: list[Blob] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        if float(value[labels == index].mean()) < value_min:
            continue
        left = int(stats[index, cv2.CC_STAT_LEFT])
        top = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        candidates.append(Blob(cx=left + width / 2.0, cy=top + height / 2.0, area=area))
    candidates.sort(key=lambda blob: blob.area, reverse=True)
    return candidates


def find_tracker_candidates(roi: np.ndarray) -> list[Blob]:
    """Кандидаты на поплавок для захвата: строгий насыщенный красный малой площади.

    Отличается от `find_red_candidates` порогами (`TRACK_*`): блики воды отсекаются,
    поэтому захват не уходит на воду (design.md §11).
    """
    return find_red_candidates(
        roi,
        min_area=TRACK_MIN_AREA,
        max_area=TRACK_MAX_AREA,
        saturation_min=TRACK_SATURATION_MIN,
        value_min=TRACK_VALUE_MIN,
        hue_max=TRACK_HUE_MAX,
    )


def find_bobber(
    roi: np.ndarray,
    near: tuple[float, float] | None = None,
    radius: float = BOBBER_RADIUS,
    min_area: int = BOBBER_MIN_AREA,
    max_area: int = BOBBER_MAX_AREA,
) -> Blob | None:
    """Самый крупный красный кластер поплавка в окне проверки.

    `near` — точка в пикселях кадра (обычно центроид всплеска): кластеры дальше `radius`
    игнорируются, поэтому берег, костёр и доспех персонажа вне окна не влияют (design §10).
    """
    for blob in find_red_candidates(roi, min_area, max_area):
        if near is None or math.hypot(blob.cx - near[0], blob.cy - near[1]) <= radius:
            return blob
    return None


def bobber_present(
    roi: np.ndarray,
    near: tuple[float, float] | None = None,
    radius: float = BOBBER_RADIUS,
    min_area: int = BOBBER_MIN_AREA,
    max_area: int = BOBBER_MAX_AREA,
) -> bool:
    """Есть ли поплавок в окне проверки.

    Без `near` ищется по всему кадру: в сцене с красным рельефом это даёт ложные
    срабатывания, поэтому в сессии всегда передаётся точка подсечки.
    """
    return find_bobber(roi, near=near, radius=radius, min_area=min_area, max_area=max_area) is not None


def to_value(img: np.ndarray) -> np.ndarray:
    """HSV-компонента V (max по каналам): для красного поплавка она выше серой яркости."""
    if img.ndim == 2:
        return img
    return img.max(axis=2)


def _window(img: np.ndarray, center: tuple[float, float], half: int):
    height, width = img.shape[:2]
    left = max(int(round(center[0])) - half, 0)
    top = max(int(round(center[1])) - half, 0)
    right = min(int(round(center[0])) + half + 1, width)
    bottom = min(int(round(center[1])) + half + 1, height)
    if right <= left or bottom <= top:
        return None
    return img[top:bottom, left:right], left, top


def bright_area(
    img: np.ndarray,
    center: tuple[float, float],
    half: int = SPLASH_WINDOW_HALF,
    threshold: int = BRIGHT_THRESHOLD,
) -> int:
    """Число пикселей ярче `threshold` в окне вокруг точки — метрика всплеска.

    Локальное окно убирает шум воды: глобальный порог всплеск не отделяет (design §10).
    """
    window = _window(img, center, half)
    if window is None:
        return 0
    patch, _, _ = window
    return int((to_value(patch) >= int(threshold)).sum())


def bright_centroid(
    img: np.ndarray,
    center: tuple[float, float],
    half: int = SPLASH_WINDOW_HALF,
    threshold: int = BRIGHT_THRESHOLD,
) -> tuple[float, float] | None:
    """Центроид ярких пикселей в окрестности точки (пиксели исходного кадра) или None."""
    window = _window(img, center, half * 2)
    if window is None:
        return None
    patch, left, top = window
    mask = to_value(patch) >= int(threshold)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(left + xs.mean()), float(top + ys.mean())


def frame_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Средняя абсолютная разница кадров; `inf`, если кадры несравнимы."""
    if a is None or b is None or a.shape != b.shape:
        return float("inf")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(diff.mean())
