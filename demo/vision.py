"""Чистые функции детекции: яркие кластеры, всплеск поклёвки, присутствие поплавка.

Ни `mss`, ни `ctypes`, ни файловой системы, ни состояния между вызовами (design.md §3):
на вход — `ndarray`, на выход — координаты и bool. Поэтому функции можно вызывать
на сохранённых кадрах при офлайн-подборе порога (задача 4.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

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

# Вода на кадре клиента. Пределы широкие: измеренная тёмная вода (ночь, глубокая тень) —
# H 86–90, S 133–160, V 59–73; дневная — H 90–100, S 60–120, V 120–200. Небо, берег,
# интерфейс и доспех в эти пределы не попадают: у них другой тон либо низкая насыщенность.
WATER_HUE_RANGE = (70, 115)
WATER_SATURATION_MIN = 50
WATER_VALUE_RANGE = (25, 215)
# Мутная и илистая вода уходит в жёлто-коричневый тон (замер живой сцены: H 20–28,
# S 103–129, V 80–102), поэтому второй путь берёт преобладающий оттенок кадра: вода —
# самое большое однородное по тону поле. Доля пикселей отсекает кадры без воды вовсе.
WATER_DOMINANT_HUE_BIN = 16
WATER_DOMINANT_SATURATION_MIN = 40
WATER_DOMINANT_VALUE_MIN = 20
WATER_DOMINANT_SHARE = 0.25
# Открытие снимает пиксельный шум, закрытие заливает блики и пену внутри воды.
WATER_OPEN_KERNEL = 3
WATER_CLOSE_KERNEL = 15
# Санитарные проверки: без них «водой» станет любой голубоватый уголок интерфейса.
WATER_MIN_FRACTION = 0.10
WATER_MIN_SIDE_FRACTION = 0.20
# Отступ области поиска от границ воды: точка заброса не должна упираться в кромку.
WATER_MARGIN = 8
# Кандидат в поплавок обязан стоять в воде: центр кластера — сам объект (у поплавка он
# «не вода»), поэтому проверяется кольцо вокруг него. Замер по записи probe/20261106-092059
# на 340 кадрах: у поплавка доля «водных» пикселей в кольце 0.77–1.00 (медиана 0.93),
# у шипов наплечника персонажа — медиана 0.50. Близость к воде считается по насыщенности
# и яркости самой воды сцены: доспех и берег насыщеннее и ярче (design D10).
WATER_SIMILARITY_SAT = 1.35
WATER_SIMILARITY_VAL = 1.45
WATER_AROUND_SHARE = 0.75
WATER_AROUND_RING = 10
# Порог «ярких» пикселей всплеска относительно яркости воды: вода в кадре поклёвки
# остаётся собой, а пена всплеска в 2–3 раза ярче (замер живой сцены 06.11.2026:
# вода V≈84, пена V≈200). Порог ниже яркости воды обращает счётчик в насыщение —
# база 5545 из 5625, всплеск не детектируется никогда (design D4).
BURST_VALUE_RATIO = 1.5
# Порог «ярких» пикселей всплеска относительно яркости воды: вода в кадре поклёвки
# остаётся собой, а пена всплеска в 2–3 раза ярче (замер живой сцены 06.11.2026:
# вода V≈84, пена V≈200). Порог ниже яркости воды обращает счётчик в насыщение —
# база 5545 из 5625, всплеск не детектируется никогда (design D4).
BURST_VALUE_RATIO = 1.5

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
class BobberPreset:
    """Пределы поплавка: по умолчанию — строгий пресет захвата (`TRACK_*`).

    Калибровка подменяет пределы на измеренные в текущей сцене (design D4), поэтому
    детекция не зависит от освещения и точки мира.
    """

    hue_max: int = TRACK_HUE_MAX
    saturation_min: int = TRACK_SATURATION_MIN
    value_min: int = TRACK_VALUE_MIN
    area_min: int = TRACK_MIN_AREA
    area_max: int = TRACK_MAX_AREA


TRACK_PRESET = BobberPreset()
# Верификация исхода остаётся на мягких пределах: строгий пресет там давал ложные `OK`.
VERIFY_PRESET = BobberPreset(
    hue_max=RED_HUE_MAX,
    saturation_min=BOBBER_SATURATION_MIN,
    value_min=BOBBER_VALUE_MIN,
    area_min=BOBBER_MIN_AREA,
    area_max=BOBBER_MAX_AREA,
)


@dataclass(frozen=True)
class Blob:
    """Кластер: центр описывающего прямоугольника в пикселях кадра и площадь в пикселях.

    Для компактных кластеров (поплавок, всплеск) центр bbox совпадает с центроидом маски;
    для вытянутых областей они расходятся, поэтому в отчётах уточняется, что имеется в виду.
    """

    cx: float
    cy: float
    area: int


@dataclass(frozen=True)
class WaterLimits:
    """Пределы «водности» воды: насыщенность, яркость и порог ярких пикселей всплеска.

    Считаются по самой воде сцены, поэтому подходят и для тёмной, и для светлой воды:
    берег, трава и доспех всегда насыщеннее и ярче воды (design D10). `burst_value` —
    яркость, выше которой начинается пена всплеска: ниже яркости воды счётчик ярких
    пикселей насыщается, и поклёвка перестаёт отличаться от покоя (design D4).
    """

    saturation_max: float
    value_max: float
    burst_value: float


@dataclass(frozen=True)
class WaterMask:
    """Маска воды, её границы и пределы «водности» в координатах кадра клиента."""

    mask: np.ndarray
    left: int
    top: int
    width: int
    height: int
    limits: WaterLimits | None = None


def detect_water(frame: np.ndarray | None) -> WaterMask | None:
    """Крупнейшая связная область воды на кадре клиента или `None`, если воды нет.

    Два пути, потому что цвет воды в игре разный: сначала пределы сине-зелёной воды
    (`WATER_*`), затем преобладающий оттенок кадра — он ловит мутную и илистую воду,
    которая в сине-зелёные пределы не попадает (design D1). В обоих случаях маска
    чистится морфологией (открытие — шум, закрытие — блики и пена) и проверяется доля
    в кадре: иначе «водой» окажется уголок интерфейса.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3 or frame.size == 0:
        return None
    for raw in (_colour_water_mask(frame), _dominant_hue_mask(frame)):
        if raw is None:
            continue
        water = _largest_water_region(_clean_water_mask(raw), frame.shape)
        if water is not None:
            return replace(water, limits=_water_limits(frame, raw))
    return None


def _water_limits(frame: np.ndarray, raw: np.ndarray) -> WaterLimits | None:
    """Пределы «водности» по самой воде: медианы насыщенности и яркости с запасом.

    Медианы считаются по исходной, ещё не сомкнутой маске: закрытие 15×15 заливает доспех
    «водой», и пределы перестают что-либо отсекать (design D10).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    inside = raw > 0
    if not inside.any():
        return None
    median_s = float(np.median(hsv[:, :, 1][inside]))
    median_v = float(np.median(hsv[:, :, 2][inside]))
    return WaterLimits(
        saturation_max=WATER_SIMILARITY_SAT * median_s,
        value_max=WATER_SIMILARITY_VAL * median_v,
        burst_value=BURST_VALUE_RATIO * median_v,
    )


def _colour_water_mask(frame: np.ndarray) -> np.ndarray | None:
    """Маска воды по пределам сине-зелёного тона (до морфологии)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([WATER_HUE_RANGE[0], WATER_SATURATION_MIN, WATER_VALUE_RANGE[0]], np.uint8)
    upper = np.array([WATER_HUE_RANGE[1], 255, WATER_VALUE_RANGE[1]], np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _dominant_hue_mask(frame: np.ndarray) -> np.ndarray | None:
    """Маска воды по преобладающему оттенку кадра (до морфологии): мутная вода, ил."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int32) // WATER_DOMINANT_HUE_BIN
    counts = np.bincount(hue.ravel(), minlength=180 // WATER_DOMINANT_HUE_BIN + 1)
    dominant = int(np.argmax(counts))
    if counts[dominant] < WATER_DOMINANT_SHARE * hue.size:
        return None
    suitable = (
        (hue == dominant)
        & (hsv[:, :, 1] >= WATER_DOMINANT_SATURATION_MIN)
        & (hsv[:, :, 2] >= WATER_DOMINANT_VALUE_MIN)
    )
    return suitable.astype(np.uint8) * 255


def _clean_water_mask(mask: np.ndarray) -> np.ndarray:
    """Морфология маски воды: открытие убирает шум, закрытие заливает блики и пену."""
    open_kernel = np.ones((WATER_OPEN_KERNEL, WATER_OPEN_KERNEL), np.uint8)
    close_kernel = np.ones((WATER_CLOSE_KERNEL, WATER_CLOSE_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)


def _largest_water_region(mask: np.ndarray, shape: tuple[int, ...]) -> WaterMask | None:
    """Крупнейшая компонента маски, прошедшая санитарные проверки площади и сторон."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:  # только фон
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[index, cv2.CC_STAT_AREA])
    if area < WATER_MIN_FRACTION * mask.size:
        return None
    left = int(stats[index, cv2.CC_STAT_LEFT])
    top = int(stats[index, cv2.CC_STAT_TOP])
    width = int(stats[index, cv2.CC_STAT_WIDTH])
    height = int(stats[index, cv2.CC_STAT_HEIGHT])
    if width < WATER_MIN_SIDE_FRACTION * shape[1] or height < WATER_MIN_SIDE_FRACTION * shape[0]:
        return None
    component = labels == index
    return WaterMask(
        mask=component,
        left=left,
        top=top,
        width=width,
        height=height,
    )


def search_box(water: WaterMask, margin: int = WATER_MARGIN) -> tuple[int, int, int, int]:
    """Прямоугольник области поиска: bbox воды, сжатый на отступ.

    Узкую полосу воды отступ не съедает целиком: если сжатие делает область пустой,
    возвращается исходный bbox.
    """
    if water.width <= 2 * margin or water.height <= 2 * margin:
        return water.left, water.top, water.width, water.height
    return (
        water.left + margin,
        water.top + margin,
        water.width - 2 * margin,
        water.height - 2 * margin,
    )


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


def find_tracker_candidates(
    roi: np.ndarray,
    preset: BobberPreset | None = None,
    water: WaterLimits | None = None,
) -> list[Blob]:
    """Кандидаты на поплавок для захвата: насыщенный красный кластер в воде.

    Отличается от `find_red_candidates` порогами по умолчанию (`TRACK_*`): блики воды
    отсекаются, поэтому захват не уходит на воду (design.md §11). `water` — пределы
    «водности» сцены: кластер, вокруг которого не вода (берег, трава, доспех персонажа),
    кандидатом не считается (design D10).
    """
    limits = TRACK_PRESET if preset is None else preset
    blobs = find_red_candidates(
        roi,
        min_area=limits.area_min,
        max_area=limits.area_max,
        saturation_min=limits.saturation_min,
        value_min=limits.value_min,
        hue_max=limits.hue_max,
    )
    if water is None:
        return blobs
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return [blob for blob in blobs if water_around(hsv, blob, water)]


def water_around(
    hsv: np.ndarray,
    blob: Blob,
    water: WaterLimits,
    ring: int = WATER_AROUND_RING,
    threshold: float = WATER_AROUND_SHARE,
) -> bool:
    """Стоит ли кластер в воде: доля «водных» пикселей в кольце вокруг него.

    Проверять центр кластера бессмысленно: центр — сам объект, у поплавка он всегда
    «не вода». Кольцо же у поплавка водное, а у шипа наплечника, травы на берегу и деталей
    доспеха — нет (design D10). Замеряется по кадру, а не по маске сцены: персонаж
    в анимации постоянно смещается, и маска, снятая до каста, быстро устаревает.
    Внутренний радиус берётся по площади кластера, чтобы сам объект не попадал в замер.
    """
    inner = max(10, int(math.sqrt(max(blob.area, 1) / math.pi)) + 6)
    outer = inner + ring
    height, width = hsv.shape[:2]
    x, y = int(round(blob.cx)), int(round(blob.cy))
    y0, y1 = max(0, y - outer), min(height, y + outer + 1)
    x0, x1 = max(0, x - outer), min(width, x + outer + 1)
    window = hsv[y0:y1, x0:x1]
    if window.size == 0:
        return False
    core = np.zeros(window.shape[:2], dtype=bool)
    core_y, core_x = y - y0, x - x0
    core[
        max(0, core_y - inner) : core_y + inner + 1,
        max(0, core_x - inner) : core_x + inner + 1,
    ] = True
    shell = ~core
    if not shell.any():
        return False
    ring_pixels = window[shell]
    similar = (ring_pixels[:, 1] <= water.saturation_max) & (
        ring_pixels[:, 2] <= water.value_max
    )
    return float(similar.mean()) >= threshold


def inside_mask(mask: np.ndarray, x: float, y: float) -> bool:
    """Лежит ли точка внутри маски (координаты пикселей кадра)."""
    height, width = mask.shape[:2]
    column = int(round(x))
    row = int(round(y))
    if not (0 <= row < height and 0 <= column < width):
        return False
    return bool(mask[row, column])


def find_bobber(
    roi: np.ndarray,
    near: tuple[float, float] | None = None,
    radius: float = BOBBER_RADIUS,
    min_area: int = BOBBER_MIN_AREA,
    max_area: int = BOBBER_MAX_AREA,
    mask: np.ndarray | None = None,
    preset: BobberPreset | None = None,
) -> Blob | None:
    """Самый крупный красный кластер поплавка в окне проверки.

    `near` — точка в пикселях кадра (обычно центроид всплеска): кластеры дальше `radius`
    игнорируются, поэтому берег, костёр и доспех персонажа вне окна не влияют (design §10).
    `mask` — маска воды: кластеры вне воды игнорируются. `preset` — пределы кластера;
    по умолчанию мягкий `VERIFY_PRESET`: строгий пресет захвата здесь давал ложные `OK`.
    """
    limits = (
        preset
        if preset is not None
        else BobberPreset(
            hue_max=RED_HUE_MAX,
            saturation_min=BOBBER_SATURATION_MIN,
            value_min=BOBBER_VALUE_MIN,
            area_min=min_area,
            area_max=max_area,
        )
    )
    for blob in find_red_candidates(
        roi,
        limits.area_min,
        limits.area_max,
        limits.saturation_min,
        limits.value_min,
        limits.hue_max,
    ):
        if mask is not None and not inside_mask(mask, blob.cx, blob.cy):
            continue
        if near is None or math.hypot(blob.cx - near[0], blob.cy - near[1]) <= radius:
            return blob
    return None


def bobber_present(
    roi: np.ndarray,
    near: tuple[float, float] | None = None,
    radius: float = BOBBER_RADIUS,
    min_area: int = BOBBER_MIN_AREA,
    max_area: int = BOBBER_MAX_AREA,
    mask: np.ndarray | None = None,
    preset: BobberPreset | None = None,
) -> bool:
    """Есть ли поплавок в окне проверки.

    Без `near` ищется по всему кадру: в сцене с красным рельефом это даёт ложные
    срабатывания, поэтому в сессии всегда передаётся точка подсечки.
    """
    blob = find_bobber(
        roi,
        near=near,
        radius=radius,
        min_area=min_area,
        max_area=max_area,
        mask=mask,
        preset=preset,
    )
    return blob is not None


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


def colour_at(
    img: np.ndarray,
    center: tuple[float, float],
    preset: BobberPreset | None = None,
    radius: int = 8,
) -> tuple[float, float, float] | None:
    """Средние H/S/V пикселей поплавка в окрестности точки (по пределам пресета).

    Нужны калибровке: по подтверждённым `OK` она учит настоящий цвет поплавка в сцене.
    Средний оттенок корректен для красного (H около 0): кластер не пересекает границу
    шкалы 0/180, поэтому усреднение по кругу не требуется.
    """
    window = _window(img, center, radius)
    if window is None:
        return None
    patch, _, _ = window
    limits = TRACK_PRESET if preset is None else preset
    mask = _red_mask(patch, limits.saturation_min, limits.value_min, limits.hue_max)
    if not mask.any():
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    values = hsv[mask > 0]
    return (
        float(values[:, 0].mean()),
        float(values[:, 1].mean()),
        float(values[:, 2].mean()),
    )


def frame_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Средняя абсолютная разница кадров; `inf`, если кадры несравнимы."""
    if a is None or b is None or a.shape != b.shape:
        return float("inf")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(diff.mean())
