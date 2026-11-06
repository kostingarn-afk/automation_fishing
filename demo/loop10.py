"""Сессия рыбалки блоками по 10 циклов: каст -> всплеск -> подсечка -> верификация.

Линейный цикл без автомата состояний (design.md §4): состояние — локальные переменные,
единственный возврат на рекаст — завершение цикла маркером отказа. Блок из 10 циклов
закрывается вердиктом и сменяется следующим без участия пользователя (design.md §12).
Запуск: `python loop10.py`

Коды выхода: 0 — нет блока «не принято» (в том числе при остановке пользователем);
1 — был блок «не принято» или аварийное завершение; 2 — окно, фокус, аргументы
или отмена до старта.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import capture
import input as input_layer
import vision
import win

# Работа делится на блоки по 10 циклов: блок закрывается вердиктом, затем сразу начинается
# следующий — сессия идёт до отмены пользователем (F12). Блок принят при <= 2 неудачных циклов.
BLOCK_CYCLES = 10
CAST_KEY = "1"
OK_REQUIRED = 8
FAILURES_ALLOWED = BLOCK_CYCLES - OK_REQUIRED

BITE_TIMEOUT = 22.0
# Захват поплавка: каст отправлен, поплавок появляется за ~0.3 с (кадр 6 из записей).
# Если за 4 с строгий кластер не появился — каст не прошёл или поплавок вне ROI: `MISS`,
# и сразу следующий каст вместо 22 с ожидания.
ACQUIRE_TIMEOUT = 4.0
# Первый захват — не раньше, чем каст долетел и камера успокоилась: иначе окна всплеска
# настраиваются на прибрежную траву, которая в большом ROI тоже проходит по красному.
ACQUIRE_DELAY = 0.9
# После старта захвата набор кандидатов пополняется каждый кадр: поплавок, найденный позже
# первых кадров, должен попасть в окна, иначе трекер до конца цикла смотрит на траву (D3).
# Две точки ближе этого радиуса — один и тот же кандидат, второе окно не нужно.
MONITOR_MATCH_RADIUS = 10.0
# Сколько окон всплеска держать одновременно: каждое — одна проверка яркости на кадр.
MONITOR_LIMIT = 6
# Поплавок исчез до обнаруженного всплеска (рыба сошла) — выходим сразу, без 22 с.
GONE_WITHOUT_BURST_FRAMES = 30
# Пауза курсор -> клик: единственный источник в `input.py` (план §4.3, спека задачи 3.2).
STRIKE_DELAY = input_layer.STRIKE_DELAY
STRIKE_RETRIES = 3
STRIKE_INTERVAL = 0.4
# Автолут собирает добычу и закрывает окно за 1.2–2.5 с; до этого поплавок ещё виден,
# поэтому окно верификации случайное из этого диапазона (уточнено на клиенте).
VERIFY_WINDOW = (1.2, 2.5)
GONE_CONFIRM_FRAMES = 3

# Прыжок: один цикл блока выбирается случайно из 3..BLOCK_CYCLES. После его закрытия —
# пауза 1–1.5 с, нажатие `space`, ещё пауза 1–1.5 с, и только затем следующий каст.
# Прыжок не входит в контракт поимки: маркер цикла и вердикт блока от него не зависят.
JUMP_CYCLES = (3, BLOCK_CYCLES)
JUMP_PAUSE_BEFORE = (1.0, 1.5)
JUMP_PAUSE_AFTER = (1.0, 1.5)
# Шаг опроса F12 в паузах прыжка: остановка срабатывает не позже одного шага.
JUMP_SLEEP_STEP = 0.05

# Калибровка: цель — 8 подтверждённых OK, после чего параметры детекции применяются.
# Считаются успешные циклы, а не блоки и не рыбы: 8 OK можно набрать за 8–25 циклов.
CALIBRATION_TARGET_OK = 8
CALIBRATION_MAX_CYCLES = 25
# Область сессии не должна содержать интерфейс и персонажа: маска воды накрывает и панели
# действий, и доспех, а их красные детали перехватывают захват поплавка (живой прогон
# 06.11.2026: трекер уходил на доспех внизу экрана). Доли клиента, как в спеке.
SCENE_SAFE_X = (0.02, 0.98)
SCENE_SAFE_Y = (0.02, 0.74)
# Разрешающий пресет на время калибровки: строгий пресет захвата не находит тёмный
# поплавок (V≈54 при требовании V≥105), а калибровать без поимок нечего (design D3).
SEARCH_PRESET = vision.BobberPreset(
    hue_max=15,
    saturation_min=140,
    value_min=25,
    # Поплавок занимает 160–195 px в этой геометрии (замер по записи probe/20261106-092059),
    # а шипы наплечника персонажа дают 8–40 px: нижняя граница отсекает доспех (design D10).
    area_min=40,
    area_max=400,
)

# Подобраны офлайн по 10 записям водной сцены (задача 4.5): 8 всплесков, 0 одиночных
# срабатываний. Отчёт воспроизводится командой
# `python probe_splash.py --tune <каталоги записей>` (пишется в каталог первой записи).
# Прирост 10 отвергнут: в живом прогоне качание поплавка давало +10 при базе 54
# и ложную подсечку, поэтому минимальный прирост — 18.
BURST_THRESHOLD = 130
SPLASH_RISE = 18
# Окно всплеска: диаметр поплавка плюс запас на пену вокруг него. Порог ярких пикселей
# калибруется по яркости воды сцены (`vision.WaterLimits.burst_value`), а не по пикам,
# измеренным при пороге по умолчанию.
WINDOW_PADDING = 6
WIN_FAIL = "WIN_FAIL"
LOST = "LOST"
MISS = "MISS"
NO_STRIKE = "NO_STRIKE"
OK = "OK"
MARKERS = (WIN_FAIL, LOST, MISS, NO_STRIKE, OK)


class RunInterrupted(RuntimeError):
    """Аварийная остановка по F12."""


class CaptureFailed(RuntimeError):
    """Слой захвата не отдал кадр: цикл помечается WIN_FAIL."""

    def __init__(self, message: str, frame=None) -> None:
        super().__init__(message)
        self.frame = frame


class Trace:
    """Построчная трасса: одна строка на цикл, ровно один маркер, UTF-8."""

    def __init__(self, path) -> None:
        capture.ensure_dir(path.parent)
        self.path = path
        self._handle = path.open("w", encoding="utf-8", newline="\n")

    def write(
        self,
        cycle: int,
        cast_ms: int,
        burst: str,
        strike: str,
        gone: str,
        marker: str,
        reason: str = "-",
        jump: str = "-",
    ) -> None:
        """Строка трассы: моменты фаз в мс от каста, ровно один маркер и причина отказа.

        Причина — отдельная колонка: моменты содержат только числа или `-`, поэтому строку
        можно разбирать позиционно, а диагностическая причина (`NO_BOBBER`, `NO_BURST`, ...)
        не подменяет момент. Колонка `jump` — момент нажатия клавиши прыжка или `-`;
        она стоит перед маркером, поэтому маркер и причина остаются последними.
        """
        if marker not in MARKERS:
            raise ValueError(f"неизвестный маркер: {marker}")
        line = (
            f"#{cycle:02d} | cast {cast_ms} | burst {burst} | strike {strike} | "
            f"gone {gone} | jump {jump} | {marker} | {reason}"
        )
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def compute_verdict(markers: list[str], crashed: bool = False, complete: bool = True) -> str:
    """Вердикт блока: «принято» требует полного блока, >= OK_REQUIRED циклов OK и без WIN_FAIL.

    `complete=False` — блок прерван (F12) до 10-го цикла: вердикта «принято» быть не может.
    `WIN_FAIL` — сломался слой окна или захвата: даже 9 `OK` в блоке не делают его принятым.
    """
    if crashed or not complete or WIN_FAIL in markers:
        return "не принято"
    return "принято" if markers.count(OK) >= OK_REQUIRED else "не принято"


def marker_distribution(markers: list[str]) -> str:
    parts = [f"{marker}={markers.count(marker)}" for marker in MARKERS if markers.count(marker)]
    return ", ".join(parts) if parts else "нет циклов"


@dataclass
class Scene:
    """Область поиска поплавка, маска воды и пределы «водности» этой сцены.

    `mask is None` — вода не найдена, область взята из долей по умолчанию: поиск идёт
    по всему кадру области, как до появления автокалибровки. `water` — пределы насыщенности
    и яркости воды: по ним захват отбрасывает кластеры на берегу и на доспехе (design D10).
    """

    region: win.Region
    mask: np.ndarray | None = None
    source: str = "доли по умолчанию"
    water: vision.WaterLimits | None = None

    def where(self) -> str:
        """Короткое описание области для консоли и отчёта (экранные координаты)."""
        return (
            f"{self.region.width}x{self.region.height} "
            f"в экранных ({self.region.left},{self.region.top}), {self.source}"
        )


def clip_to_safe(
    rect: tuple[int, int, int, int],
    client_width: int,
    client_height: int,
) -> tuple[int, int, int, int]:
    """Обрезает прямоугольник до безопасной зоны клиента: без интерфейса и персонажа.

    Маска воды может накрыть панели действий и доспех: их красные детали перехватывают
    захват поплавка. Зона задана долями клиента (`SCENE_SAFE_*`); если после обрезки
    прямоугольник пуст, возвращается исходный — лучшее из доступного.
    """
    left = max(rect[0], int(client_width * SCENE_SAFE_X[0]))
    top = max(rect[1], int(client_height * SCENE_SAFE_Y[0]))
    right = min(rect[0] + rect[2], int(client_width * SCENE_SAFE_X[1]))
    bottom = min(rect[1] + rect[3], int(client_height * SCENE_SAFE_Y[1]))
    if right - left < 2 or bottom - top < 2:
        return rect
    return left, top, right - left, bottom - top


def detect_scene(window: win.GameWindow, grabber: capture.Grabber) -> Scene:
    """Область поиска поплавка по маске воды на свежем кадре клиента.

    Кадр берётся один раз на старте сессии: камера между циклами зафиксирована, поэтому
    повторный расчёт стоил бы кадра без пользы (предложение §What Changes). Из области
    вычитаются интерфейс и персонаж — иначе их красные детали перехватывают захват.
    """
    frame = grabber.grab_client(window)
    if frame is None:
        raise CaptureFailed(f"кадр клиента на старте не получен: {grabber.last_state}")
    water = vision.detect_water(frame)
    if water is None:
        print(
            f"вода не найдена: область поиска — доли по умолчанию "
            f"X {win.ROI_FRACTION_X}, Y {win.ROI_FRACTION_Y}"
        )
        return Scene(region=win.roi_region(window), mask=None)
    left, top, width, height = vision.search_box(water)
    client_width, client_height = win.client_size(window)
    left, top, width, height = clip_to_safe((left, top, width, height), client_width, client_height)
    origin_x, origin_y = win.client_origin(window)
    mask = np.ascontiguousarray(water.mask[top : top + height, left : left + width])
    print(
        f"вода найдена: область {width}x{height} в клиенте ({left},{top}), "
        f"зона без интерфейса и персонажа"
    )
    return Scene(
        region=win.Region(origin_x + left, origin_y + top, width, height),
        mask=mask,
        source="маска воды",
        water=water.limits,
    )


# Границы параметров детекции: значение вне диапазона не применяется, вместо него берётся
# значение по умолчанию. Границы шире измеренных, но не позволяют «отключить» детекцию.
PARAM_BOUNDS = {
    "hue_max": (5, 25),
    "saturation_min": (80, 220),
    "value_min": (25, 160),
    "area_min": (4, 80),
    "area_max": (40, 600),
    "burst_threshold": (40, 200),
    "splash_rise": (8, 30),
    "window_radius": (10, 90),
    "confirm_frames": (2, 4),
}


@dataclass
class CycleStats:
    """Наблюдение успешного цикла: чем на самом деле оказался пойманный поплавок.

    Метка `OK` — сильный признак: клик по блику поплавок не убирает, цикл дал бы
    `NO_STRIKE`, поэтому в статистику попадают только настоящие поимки (design D3).
    """

    hue: float
    saturation: float
    value: float
    area: int
    baseline: float
    peak: int

    @property
    def rise(self) -> float:
        """Прирост яркости в окне всплеска над базой."""
        return self.peak - self.baseline


@dataclass
class DetectionParams:
    """Параметры детекции на сессию: пресет поплавка и метрики всплеска."""

    preset: vision.BobberPreset = vision.TRACK_PRESET
    burst_threshold: int = BURST_THRESHOLD
    splash_rise: int = SPLASH_RISE
    window_radius: int = vision.SPLASH_WINDOW_HALF
    confirm_frames: int = 2

    @classmethod
    def defaults(cls) -> "DetectionParams":
        """Значения по умолчанию — текущие константы проекта."""
        return cls()

    def normalized(self) -> "DetectionParams":
        """Заменяет значения вне допустимых границ значениями по умолчанию."""
        defaults = DetectionParams()

        def pick(name: str, value: float, fallback: float) -> int:
            low, high = PARAM_BOUNDS[name]
            if low <= value <= high:
                return int(value)
            print(f"параметр {name}={value} вне границ {low}..{high}: беру {fallback}")
            return int(fallback)

        preset = vision.BobberPreset(
            hue_max=pick("hue_max", self.preset.hue_max, defaults.preset.hue_max),
            saturation_min=pick(
                "saturation_min", self.preset.saturation_min, defaults.preset.saturation_min
            ),
            value_min=pick("value_min", self.preset.value_min, defaults.preset.value_min),
            area_min=pick("area_min", self.preset.area_min, defaults.preset.area_min),
            area_max=pick("area_max", self.preset.area_max, defaults.preset.area_max),
        )
        return DetectionParams(
            preset=preset,
            burst_threshold=pick(
                "burst_threshold", self.burst_threshold, defaults.burst_threshold
            ),
            splash_rise=pick("splash_rise", self.splash_rise, defaults.splash_rise),
            window_radius=pick("window_radius", self.window_radius, defaults.window_radius),
            confirm_frames=pick(
                "confirm_frames", self.confirm_frames, defaults.confirm_frames
            ),
        )

    def summary(self) -> str:
        """Строка параметров для консоли и отчёта калибровки."""
        preset = self.preset
        return (
            f"поплавок H<={preset.hue_max}, S>={preset.saturation_min}, "
            f"V>={preset.value_min}, area {preset.area_min}-{preset.area_max}; "
            f"всплеск порог {self.burst_threshold}, прирост {self.splash_rise}, "
            f"окно ±{self.window_radius}, кадров {self.confirm_frames}"
        )

    @classmethod
    def from_stats(
        cls,
        stats: list["CycleStats"],
        water_burst_value: float | None = None,
    ) -> "DetectionParams":
        """Параметры по наблюдениям успешных циклов (design D4).

        Квантили, а не минимум и максимум: восемь наблюдений мало, крайние значения шумят.
        Все результаты зажимаются в допустимые границы `normalized()`.

        `water_burst_value` — порог яркости «ярких» пикселей, выведенный из яркости воды
        сцены. От измеренных пиков порог не берётся: пики измерены при пороге по умолчанию,
        и порог ниже яркости воды обращает счётчик в насыщение (живой прогон 06.11.2026:
        база 5545 из 5625 ярких пикселей, всплеск не детектируется никогда).
        """
        if not stats:
            return cls.defaults()
        hues = [item.hue for item in stats]
        saturations = [item.saturation for item in stats]
        values = [item.value for item in stats]
        areas = [item.area for item in stats]
        rises = [item.rise for item in stats]

        diameter = 2.0 * math.sqrt(max(percentile(areas, 0.9), 1.0) / math.pi)
        threshold = cls.defaults().burst_threshold
        if water_burst_value is not None:
            threshold = int(round(water_burst_value))
        params = cls(
            preset=vision.BobberPreset(
                hue_max=int(round(percentile(hues, 0.9))) + 4,
                saturation_min=int(round(percentile(saturations, 0.1))) - 20,
                value_min=int(round(percentile(values, 0.1))) - 15,
                area_min=int(percentile(areas, 0.1)) - 4,
                area_max=int(percentile(areas, 0.9)) + 8,
            ),
            burst_threshold=threshold,
            splash_rise=int(round(percentile(rises, 0.1))),
            # окно должно накрывать пену вокруг поплавка, а не весь кадр: три диаметра
            # давали ±37 при пятне пены ~20 px, и прирост размывался
            window_radius=int(round(diameter)) + WINDOW_PADDING,
            confirm_frames=2 if percentile(rises, 0.1) >= 12 else 3,
        )
        return params.normalized()


def percentile(values: list[float], fraction: float) -> float:
    """Квантиль числового ряда без интерполяции: ближайший индекс по дроби `fraction`."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


class BobberTracker:
    """Поплавок как строгий красный кластер, которого не было до каста и который держится.

    Мягкий красный фильтр по всему ROI не различает поплавок и блики воды (design.md §10:
    в кадре 17–35 кластеров-бликов), поэтому кандидат берётся из строгого пресета
    `vision.find_tracker_candidates` и дополнительно должен голосовать в нескольких кадрах.
    Нужен для различения `LOST` (поплавок виден, всплеск не найден) и `MISS` (каст не прошёл
    либо поплавок вне ROI).
    """

    MIN_VOTES = 5
    MATCH_RADIUS = 6.0
    # Сколько кандидатов отслеживать одновременно. В большой области воды «новых» красных
    # кластеров много, и выбор «самый голосистый» ошибается; надёжный признак — настоящий
    # всплеск, поэтому опрашиваются несколько окон сразу (design D3).
    MAX_CANDIDATES = 6
    # Кандидат считается живым, если голосовал не дольше этого числа кадров назад: прибрежная
    # трава и блики мерцают и не должны занимать окно всплеска вечно (design D3).
    RECENT_FRAMES = 10

    def __init__(
        self,
        baseline,
        preset: vision.BobberPreset | None = None,
        water: vision.WaterLimits | None = None,
    ) -> None:
        self._preset = preset
        self._water = water
        known = (
            vision.find_tracker_candidates(baseline, preset, water) if baseline is not None else []
        )
        self._known = [(blob.cx, blob.cy) for blob in known]
        # запись голосования: [cx, cy, голоса, номер кадра последнего попадания]
        self._votes: list[list[float]] = []
        self._frame = 0
        # Кеш кадрового поиска: за кадр трекер опрашивает до MAX_CANDIDATES точек, и повторный
        # поиск кластеров на каждую — это полный разбор кадра 20 раз в секунду (design D3).
        self._cache: list[tuple[object, list[vision.Blob]]] = []

    def _is_new(self, blob: vision.Blob) -> bool:
        return all(
            math.hypot(blob.cx - cx, blob.cy - cy) > self.MATCH_RADIUS for cx, cy in self._known
        )

    def _blobs(self, frame) -> list[vision.Blob]:
        """Кластеры кадра: поиск один раз на кадр, повторный вызов берёт кеш."""
        for cached_frame, blobs in self._cache:
            if cached_frame is frame:
                return blobs
        blobs = vision.find_tracker_candidates(frame, self._preset, self._water)
        self._cache = [(frame, blobs)] + self._cache[:1]
        return blobs

    def update(self, frame) -> None:
        self._frame += 1
        for blob in self._blobs(frame):
            if not self._is_new(blob):
                continue
            for entry in self._votes:
                if math.hypot(blob.cx - entry[0], blob.cy - entry[1]) <= self.MATCH_RADIUS:
                    entry[0], entry[1] = blob.cx, blob.cy
                    entry[2] += 1
                    entry[3] = self._frame
                    break
            else:
                self._votes.append([blob.cx, blob.cy, 1.0, self._frame])

    def seen(self) -> bool:
        return any(entry[2] >= self.MIN_VOTES for entry in self._votes)

    def point(self) -> tuple[float, float] | None:
        best = max(self._votes, key=lambda entry: entry[2], default=None)
        return (best[0], best[1]) if best else None

    def points(self, limit: int = MAX_CANDIDATES) -> list[tuple[float, float]]:
        """До `limit` подтверждённых и недавно видимых кандидатов, самые голосистые первыми.

        Недавность отсекает мерцающие кластеры (трава, блики): держится в кадре только
        настоящий поплавок, поэтому набор окон обновляется по ходу ожидания (design D3).
        """
        confirmed = [
            entry
            for entry in self._votes
            if entry[2] >= self.MIN_VOTES and self._frame - entry[3] <= self.RECENT_FRAMES
        ]
        confirmed.sort(key=lambda entry: entry[2], reverse=True)
        return [(entry[0], entry[1]) for entry in confirmed[:limit]]

    def confirmed(self, frame, radius: float = 12.0) -> vision.Blob | None:
        """Кластер у самой голосистой точки — проверка, что трекер не ушёл на блик.

        `radius` шире радиуса голосования: поплавок качается на воде на единицы пикселей,
        а пропуск присутствия даёт ложный `LOST`.
        """
        point = self.point()
        return None if point is None else self.confirmed_at(frame, point, radius)

    def confirmed_at(
        self,
        frame,
        point: tuple[float, float],
        radius: float = 12.0,
    ) -> vision.Blob | None:
        """Кластер у заданной точки кадра (кандидат может быть не самым голосистым)."""
        for blob in self._blobs(frame):
            if math.hypot(blob.cx - point[0], blob.cy - point[1]) <= radius:
                return blob
        return None

    def any_confirmed(self, frame, radius: float = 12.0) -> bool:
        """Хотя бы один недавний кандидат всё ещё виден: поплавок не исчез."""
        return any(
            self.confirmed_at(frame, point, radius) is not None for point in self.points()
        )


class BurstMonitor:
    """Всплеск поклёвки как локальный рост яркости у поплавка.

    Глобальный порог яркости всплеск не отделяет (design.md §10): пена занимает ~20 px
    и живёт ~100 мс, а шум воды даёт 9–15 px. В окне вокруг поплавка шум ±8 px,
    поклёвка даёт +26 px, поэтому сравниваем с медианой установившейся базы.
    """

    WARMUP_SAMPLES = 20
    BASELINE_SAMPLES = 20
    RISE = SPLASH_RISE
    RISE_RATIO = vision.SPLASH_RISE_RATIO
    # Подтверждение всплеска: порог подобран правилом «держится не меньше 2 кадров»
    # (probe_splash.MIN_EVENT_FRAMES), и спека требует «несколько кадров подряд».
    CONFIRM_FRAMES = 2

    def __init__(
        self,
        point: tuple[float, float],
        threshold: int = BURST_THRESHOLD,
        rise: int = SPLASH_RISE,
        radius: int = vision.SPLASH_WINDOW_HALF,
        confirm_frames: int = 2,
    ) -> None:
        self.point = point
        self._threshold = threshold
        self.rise = rise
        self.radius = radius
        self.confirm_frames = confirm_frames
        self._warmup = 0
        self._samples: list[int] = []
        self._above = 0
        self.baseline: float | None = None
        self.last = 0
        self.peak = 0

    def area(self, frame) -> int:
        self.last = vision.bright_area(
            frame, self.point, half=self.radius, threshold=self._threshold
        )
        return self.last

    def observe(self, frame) -> bool:
        """True — яркость у поплавка держится выше базы `confirm_frames` кадров подряд."""
        value = self.area(frame)
        if self.baseline is not None:
            self.peak = max(self.peak, value)
            above = value >= self.baseline + self.rise and value >= self.baseline * self.RISE_RATIO
            self._above = self._above + 1 if above else 0
            return self._above >= self.confirm_frames
        if self._warmup < self.WARMUP_SAMPLES:
            self._warmup += 1
            return False
        self._samples.append(value)
        if len(self._samples) >= self.BASELINE_SAMPLES:
            self.baseline = float(statistics.median(self._samples))
            print(f"  база ярких пикселей у поплавка: {self.baseline:.0f}")
        return False

    def quiet(self, value: int | None = None) -> bool:
        """Всплеск погас: яркость у поплавка вернулась к базе."""
        if self.baseline is None:
            return False
        current = self.last if value is None else value
        return current <= self.baseline + self.rise * 0.5


def wait_bobber_gone(
    grabber: capture.Grabber,
    window: win.GameWindow,
    region: win.Region,
    monitor: BurstMonitor,
    seconds: float,
    mask: np.ndarray | None = None,
    preset: vision.BobberPreset = vision.VERIFY_PRESET,
) -> bool:
    """Исход подтверждён, если всплеск погас, а поплавок в окне проверки не появился.

    В момент всплеска поплавок закрыт пеной, поэтому «поплавок отсутствует» само по себе
    ничего не значит: сначала ждём, пока яркость у точки вернётся к базе, и только потом
    проверяем красный кластер. Окно задаёт вызывающий код: автолут закрывает окно добычи
    за 1.5–3 с, и до этого поплавок ещё виден. `region` — область поиска сессии,
    `mask` — маска воды в её координатах.
    """
    pacer = capture.Pacer()
    cleared = 0
    gone_seen = False
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if win.stop_requested():
            raise RunInterrupted("F12 на фазе верификации")
        frame = grabber.grab_region(region)
        if frame is None:
            raise CaptureFailed(f"ROI не получен на фазе верификации: {grabber.last_state}")
        if monitor.quiet(monitor.area(frame)):
            cleared += 1
            if cleared >= GONE_CONFIRM_FRAMES and not vision.bobber_present(
                frame, near=monitor.point, mask=mask, preset=preset
            ):
                gone_seen = True
        else:
            cleared = 0
        pacer.wait()
    # окно выдерживается целиком: до закрытия окна добычи новый каст не пройдёт
    print(
        f"  поплавок {'исчез' if gone_seen else 'на месте'} — "
        f"{'исход подтверждён' if gone_seen else 'нужна повторная подсечка'}"
    )
    return gone_seen


def sleep_checked(seconds: float) -> None:
    """Пауза с опросом F12 шагами `JUMP_SLEEP_STEP`.

    Обычный `time.sleep` не даёт остановиться до 1.5 с, поэтому шаг ограничен:
    `F12` во время паузы прыжка поднимает `RunInterrupted` не позже одного шага.
    """
    deadline = time.monotonic() + max(seconds, 0.0)
    while True:
        if win.stop_requested():
            raise RunInterrupted("F12 в паузе прыжка")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(JUMP_SLEEP_STEP, remaining))


def perform_jump(window: win.GameWindow, number: int, moment_ms: Callable[[], int]) -> int | None:
    """Прыжок в закрытом цикле: пауза перед, `space`, пауза после.

    Возвращает момент нажатия в мс от каста или `None`, если прыжок не отправлен.
    Отказ ввода (`InputBlocked`) не превращает цикл в `WIN_FAIL`: прыжок не входит
    в контракт поимки, а если фокус не вернулся, следующий каст сам поднимет отказ.
    """
    before = random.uniform(*JUMP_PAUSE_BEFORE)
    sleep_checked(before)
    moment = moment_ms()
    try:
        input_layer.jump(window)
    except input_layer.InputBlocked:
        # подробности (какое окно активно) уже напечатал `require_foreground`
        print(f"#{number:02d}: прыжок пропущен — фокус не в клиенте")
        return None
    except OSError as exc:
        print(f"#{number:02d}: прыжок не отправлен — {exc}")
        return None
    after = random.uniform(*JUMP_PAUSE_AFTER)
    print(f"#{number:02d}: прыжок в {moment} мс, паузы {before:.2f} / {after:.2f} с")
    sleep_checked(after)
    return moment


def run_cycle(
    number: int,
    window: win.GameWindow,
    grabber: capture.Grabber,
    trace: Trace,
    click=None,
    cast_key: str | None = None,
    block: int | None = 1,
    jump: bool = False,
    scene: Scene | None = None,
    params: DetectionParams | None = None,
    stats: list[CycleStats] | None = None,
) -> str:
    """Один цикл «каст -> ожидание -> подсечка -> верификация».

    `click` — способ подсечки: по умолчанию `input_layer.strike` (ветка a),
    probe-скрипт подставляет ветку (b). `block` — номер блока, попадает в имена кадров
    отказов. `jump=True` — после закрытия цикла выполнить прыжок (один цикл на блок).
    `scene` — область поиска и маска воды (по умолчанию — доли клиента без маски),
    `params` — параметры детекции на сессию, `stats` — приёмник наблюдений успешных
    циклов для калибровки. Возвращает маркер цикла и пишет одну строку трассы.
    """
    clicker = input_layer.strike if click is None else click
    cast = CAST_KEY if cast_key is None else cast_key
    limits = DetectionParams.defaults() if params is None else params
    started = time.monotonic()

    def elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    if scene is None:
        # окно могло закрыться или уехать между циклами — это отказ слоя окна, не авария
        try:
            width, height = win.client_size(window)
            roi_left, roi_top, roi_width, roi_height = win.roi_box(width, height)
            origin_x, origin_y = win.client_origin(window)
        except (OSError, ValueError) as exc:
            raise CaptureFailed(f"геометрия окна недоступна: {exc}") from exc
        scene = Scene(
            region=win.Region(origin_x + roi_left, origin_y + roi_top, roi_width, roi_height)
        )
    region = scene.region
    baseline = grabber.grab_region(region)
    if baseline is None:
        raise CaptureFailed(f"ROI не получен до каста: {grabber.last_state}")

    try:
        input_layer.cast(window, cast)
    except input_layer.InputBlocked as exc:
        # фокус ушёл до каста: для разбора отказа прикладываем кадр до каста
        raise input_layer.InputBlocked(str(exc), baseline) from exc
    print(f"#{number:02d}: каст отправлен (ROI {region.width}x{region.height})")

    deadline = started + BITE_TIMEOUT
    acquire_deadline = started + ACQUIRE_TIMEOUT
    last_frame = baseline
    prev_frame = baseline
    tracker = BobberTracker(baseline, limits.preset, scene.water)
    monitor: BurstMonitor | None = None
    monitors: list[BurstMonitor] = []
    burst_ms: int | None = None
    burst_point: tuple[int, int] | None = None
    strike_ms: int | None = None
    gone_ms: int | None = None
    missing_frames = 0
    bobber_sample: tuple[vision.Blob, object] | None = None
    pacer = capture.Pacer()
    acquire_lock_ms = ACQUIRE_DELAY * 1000.0

    def add_monitors() -> list[tuple[float, float]]:
        """Добавляет окна для новых кандидатов трекера; возвращает добавленные точки.

        Набор окон пополняется, а не заменяется: ранний кандидат может оказаться травой,
        но настоящий поплавок появится позже и займёт своё окно (design D3).
        """
        free = MONITOR_LIMIT - len(monitors)
        if free <= 0:
            return []
        taken = [item.point for item in monitors]
        fresh = [
            point
            for point in tracker.points()
            if all(
                math.hypot(point[0] - x, point[1] - y) > MONITOR_MATCH_RADIUS for x, y in taken
            )
        ][:free]
        for point in fresh:
            monitors.append(
                BurstMonitor(
                    point,
                    limits.burst_threshold,
                    limits.splash_rise,
                    limits.window_radius,
                    limits.confirm_frames,
                )
            )
        return fresh

    def close(marker: str, reason: str = "-") -> str:
        """Общая точка закрытия цикла: прыжок, затем строка трассы.

        Все пять выходов цикла проходят здесь (design D2): пятая копия
        «пауза — прыжок — пауза» разъехалась бы с остальными. Момент и кадр отказа
        снимает вызывающий код до вызова — паузы в `reason@<мс>` не попадают.
        """
        jump_ms = perform_jump(window, number, elapsed_ms) if jump else None
        trace.write(
            number,
            0,
            "-" if burst_ms is None else str(burst_ms),
            "-" if strike_ms is None else str(strike_ms),
            "-" if gone_ms is None else str(gone_ms),
            marker,
            reason=reason,
            jump="-" if jump_ms is None else str(jump_ms),
        )
        return marker

    def finish_failure(marker: str, reason: str, message: str) -> str:
        moment = elapsed_ms()
        capture.save_fail_frame(last_frame, number, moment, marker, block=block)
        # моменты фаз — только числа или `-`; причина — токен, момент отказа — суффикс `@<мс>`
        print(f"#{number:02d}: {marker} — {message}")
        return close(marker, reason=f"{reason}@{moment}")

    def cycle_stats() -> CycleStats | None:
        """Наблюдение успешного цикла: каким оказался подтверждённый поплавок."""
        if bobber_sample is None or monitor is None or monitor.baseline is None:
            return None
        blob, sample_frame = bobber_sample
        colours = vision.colour_at(sample_frame, (blob.cx, blob.cy), limits.preset)
        if colours is None:
            return None
        hue, saturation, value = colours
        return CycleStats(
            hue=hue,
            saturation=saturation,
            value=value,
            area=blob.area,
            baseline=monitor.baseline,
            peak=monitor.peak,
        )

    while time.monotonic() < deadline:
        if win.stop_requested():
            raise RunInterrupted(f"F12 в цикле {number} на фазе ожидания поклёвки")
        frame = grabber.grab_region(region)
        if frame is None:
            raise CaptureFailed(
                f"ROI не получен на фазе ожидания: {grabber.last_state}", last_frame
            )
        prev_frame = last_frame
        last_frame = frame
        tracker.update(frame)
        if elapsed_ms() >= acquire_lock_ms:
            fresh = add_monitors()
            if fresh:
                # состав окон печатается, чтобы по трассе было видно, кто попал в захват
                print(
                    f"#{number:02d}: кандидатов в ROI: {len(monitors)} "
                    f"({', '.join(f'({x:.0f},{y:.0f})' for x, y in (item.point for item in monitors))})"
                )
        confirmed = tracker.any_confirmed(frame) if monitors else None
        if not monitors and time.monotonic() >= acquire_deadline:
            # каст не прошёл либо поплавок вне ROI: ждать 22 с бессмысленно
            return finish_failure(
                MISS,
                "NO_BOBBER",
                f"поплавок не появился за {ACQUIRE_TIMEOUT:.0f} с после каста",
            )
        if monitors:
            monitor = next((item for item in monitors if item.observe(frame)), None)
        if monitor is not None:
            burst_ms = elapsed_ms()
            burst_point = region.to_screen(monitor.point[0], monitor.point[1])
            # центроид пены — только диагностика: клик идёт по позиции поплавка
            splash = (
                vision.bright_centroid(frame, monitor.point, threshold=BURST_THRESHOLD)
                or monitor.point
            )
            print(
                f"#{number:02d}: всплеск на {burst_ms} мс, поплавок "
                f"({monitor.point[0]:.0f},{monitor.point[1]:.0f}) из {len(monitors)}, "
                f"ярких пикселей {monitor.last} при базе {monitor.baseline:.0f}, "
                f"центроид пены ({splash[0]:.0f},{splash[1]:.0f}), точка клика {burst_point}"
            )
            # цвет поплавка снимаем с кадра до всплеска: пена закрывает кластер
            sample_frame = prev_frame
            blob = tracker.confirmed_at(sample_frame, monitor.point)
            if blob is None:
                sample_frame = frame
                blob = tracker.confirmed_at(sample_frame, monitor.point)
            if blob is not None:
                bobber_sample = (blob, sample_frame)
            break
        if monitors:
            # проверка строго после всплеска: пена, закрывшая поплавок, уже дала бы burst
            if confirmed is None:
                missing_frames += 1
                if missing_frames >= GONE_WITHOUT_BURST_FRAMES:
                    # рыба сошла: поплавок исчез без обнаруженного всплеска — слой зрения
                    return finish_failure(
                        LOST, "NO_BURST", "поплавок исчез без обнаруженного всплеска"
                    )
            else:
                missing_frames = 0
        pacer.wait()

    if burst_ms is None or burst_point is None:
        # Поплавок виден (иначе вернулись бы по ACQUIRE_TIMEOUT), но всплеск не найден:
        # отказ слоя зрения — `LOST`, а не `MISS` (MISS = поплавка в ROI нет).
        return finish_failure(LOST, "TIMEOUT", f"всплеск не обнаружен за {BITE_TIMEOUT:.0f} с")

    for attempt in range(1, STRIKE_RETRIES + 1):
        if win.stop_requested():
            raise RunInterrupted(f"F12 в цикле {number} перед подсечкой")
        try:
            pause = clicker(window, burst_point, STRIKE_DELAY)
        except input_layer.InputBlocked as exc:
            # фокус мог уйти после наведения: кадр нужен для разбора отказа
            raise input_layer.InputBlocked(str(exc), last_frame) from exc
        strike_ms = elapsed_ms()
        verify_seconds = random.uniform(VERIFY_WINDOW[0], VERIFY_WINDOW[1])
        print(
            f"#{number:02d}: подсечка {attempt}/{STRIKE_RETRIES} в {burst_point}, "
            f"пауза после наведения {pause:.2f} с, окно подтверждения {verify_seconds:.1f} с"
        )
        if wait_bobber_gone(grabber, window, region, monitor, verify_seconds, scene.mask):
            gone_ms = elapsed_ms()
            break
        print(f"#{number:02d}: исход не подтверждён за {verify_seconds:.1f} с")
        if attempt < STRIKE_RETRIES:
            time.sleep(STRIKE_INTERVAL)

    if gone_ms is not None:
        print(f"#{number:02d}: OK — поплавок исчез на {gone_ms} мс")
        observation = cycle_stats()
        if stats is not None and observation is not None:
            stats.append(observation)
        return close(OK)

    frame = grabber.grab_region(region)
    if frame is None:
        frame = last_frame
    moment = elapsed_ms()
    capture.save_fail_frame(frame, number, moment, NO_STRIKE, block=block)
    print(f"#{number:02d}: NO_STRIKE — за {STRIKE_RETRIES} попытки исход не подтверждён")
    return close(NO_STRIKE, reason="NO_CONFIRM")


def write_calibration_report(
    trace_path: Path,
    scene: Scene,
    markers: list[str],
    params: DetectionParams | None,
    observations: int = 0,
    interrupted: bool = False,
) -> Path:
    """Отчёт калибровки рядом с её трассой: параметры, циклы, маркеры, область воды."""
    path = trace_path.with_name(trace_path.stem + "_report.md")
    applied = DetectionParams.defaults() if params is None else params
    if interrupted:
        source = "прервано, параметры не применялись"
    elif params is not None:
        source = "калибровка"
    else:
        source = "значения по умолчанию"
    lines = [
        "# Отчёт калибровки",
        "",
        f"- область поиска: {scene.where()}",
        f"- цель: {CALIBRATION_TARGET_OK} OK",
        f"- циклов: {len(markers)}",
        f"- маркеры: {marker_distribution(markers)}",
        f"- наблюдений для параметров: {observations}",
        f"- параметры: {applied.summary()}",
        f"- источник параметров: {source}",
        f"- трасса калибровки: {trace_path.name}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_calibration(
    window: win.GameWindow,
    grabber: capture.Grabber,
    scene: Scene,
) -> tuple[DetectionParams | None, list[str], Path]:
    """Калибровочная фаза: циклы до `CALIBRATION_TARGET_OK` подтверждённых `OK`.

    Параметры по умолчанию на время фазы заменяются разрешающим пресетом: строгий
    в тёмной сцене поплавок не находит, и калибровать было бы нечего (design D3).
    Прыжок и вердикт блока в фазе не выполняются. Возвращает `(параметры или None,
    маркеры, путь трассы)`; при остановке `F12` отчёт пишется и исключение повторяется.
    """
    trace_path = capture.TRACE_ROOT / f"calibration_{capture.timestamp()}.txt"
    trace = Trace(trace_path)
    markers: list[str] = []
    stats: list[CycleStats] = []
    limits = DetectionParams(preset=SEARCH_PRESET)
    print(
        f"калибровка: цель {CALIBRATION_TARGET_OK} OK, "
        f"не больше {CALIBRATION_MAX_CYCLES} циклов"
    )
    try:
        for cycle in range(1, CALIBRATION_MAX_CYCLES + 1):
            if win.stop_requested():
                raise RunInterrupted("F12 до цикла калибровки")
            print(
                f"--- калибровка, цикл {cycle}/{CALIBRATION_MAX_CYCLES}, "
                f"OK {markers.count(OK)}/{CALIBRATION_TARGET_OK}"
            )
            try:
                marker = run_cycle(
                    cycle,
                    window,
                    grabber,
                    trace,
                    scene=scene,
                    params=limits,
                    stats=stats,
                    block=None,
                )
            except (input_layer.InputBlocked, CaptureFailed) as exc:
                # отказ слоя окна или захвата — тот же WIN_FAIL, что и в блоке:
                # трейсбек в консоль не годится, отчёт калибровки должен сохраниться
                print(f"WIN_FAIL в калибровке: {exc}")
                reason = (
                    "INPUT_BLOCKED@0"
                    if isinstance(exc, input_layer.InputBlocked)
                    else "CAPTURE@0"
                )
                capture.save_fail_frame(
                    getattr(exc, "frame", None), cycle, 0, WIN_FAIL, block=None
                )
                trace.write(cycle, 0, "-", "-", "-", WIN_FAIL, reason=reason)
                markers.append(WIN_FAIL)
                break
            markers.append(marker)
            if markers.count(OK) >= CALIBRATION_TARGET_OK:
                break
    except RunInterrupted:
        trace.close()
        write_calibration_report(
            trace_path, scene, markers, None, observations=len(stats), interrupted=True
        )
        raise
    except BaseException:
        trace.close()
        raise
    trace.close()

    enough = (
        markers.count(OK) >= CALIBRATION_TARGET_OK
        and len(stats) >= CALIBRATION_TARGET_OK
    )
    params = (
        DetectionParams.from_stats(
            stats,
            scene.water.burst_value if scene.water is not None else None,
        )
        if enough
        else None
    )
    write_calibration_report(trace_path, scene, markers, params, observations=len(stats))
    return params, markers, trace_path


def run_block(
    number: int,
    window: win.GameWindow,
    grabber: capture.Grabber,
    scene: Scene | None = None,
    params: DetectionParams | None = None,
) -> tuple:
    """Один блок из `BLOCK_CYCLES` циклов: своя трасса, маркеры и причина остановки.

    Прыжок выполняется ровно в одном цикле блока (`jump_cycle`, случайно из `JUMP_CYCLES`);
    момент выбора печатается до первого цикла. `scene` и `params` — область поиска и
    параметры детекции, полученные на старте сессии. Возвращает `(trace_path, markers,
    crashed, stop_reason)`, где `stop_reason` — `None` (блок прошёл все циклы), `"F12"`
    (остановлен пользователем) или `"WIN_FAIL"` (отказ слоя окна или захвата). Различие
    важно: `F12` — это выбор пользователя, а не провал блока.
    """
    trace_path = capture.TRACE_ROOT / f"run_{capture.timestamp()}_b{number:02d}.txt"
    trace = Trace(trace_path)
    # слот прыжка фиксируется до первого цикла: решение принимается один раз на блок
    jump_cycle = random.randint(*JUMP_CYCLES)
    print(f"--- блок {number}: прыжок в цикле {jump_cycle} из {BLOCK_CYCLES}")
    markers: list[str] = []
    crashed = False
    stop_reason: str | None = None

    try:
        for cycle in range(1, BLOCK_CYCLES + 1):
            print(f"--- блок {number}, цикл {cycle}/{BLOCK_CYCLES}")
            started = time.monotonic()
            try:
                marker = run_cycle(
                    cycle,
                    window,
                    grabber,
                    trace,
                    block=number,
                    jump=cycle == jump_cycle,
                    scene=scene,
                    params=params,
                )
            except input_layer.InputBlocked as exc:
                moment = int((time.monotonic() - started) * 1000)
                print(f"WIN_FAIL: {exc}")
                capture.save_fail_frame(exc.frame, cycle, moment, WIN_FAIL, block=number)
                trace.write(cycle, 0, "-", "-", "-", WIN_FAIL, reason=f"INPUT_BLOCKED@{moment}")
                markers.append(WIN_FAIL)
                stop_reason = "WIN_FAIL"
                break
            except CaptureFailed as exc:
                moment = int((time.monotonic() - started) * 1000)
                print(f"WIN_FAIL: {exc}")
                capture.save_fail_frame(exc.frame, cycle, moment, WIN_FAIL, block=number)
                trace.write(cycle, 0, "-", "-", "-", WIN_FAIL, reason=f"CAPTURE@{moment}")
                markers.append(WIN_FAIL)
                stop_reason = "WIN_FAIL"
                break
            markers.append(marker)
    except RunInterrupted as exc:
        print(f"прерван: {exc}")
        stop_reason = "F12"
    except Exception as exc:  # аварийное завершение: трейсбек и вердикт блока «не принято»
        traceback.print_exc()
        print(f"аварийное завершение блока {number}: {exc}")
        crashed = True
    finally:
        trace.close()

    return trace_path, markers, crashed, stop_reason


def print_block_summary(number: int, trace_path, markers: list[str], verdict: str) -> None:
    print(f"\nитог блока {number}:")
    print(f"  трасса: {trace_path} ({len(markers)} строк)")
    print(f"  маркеры: {marker_distribution(markers)}")
    print(
        f"  циклов: {len(markers)} из {BLOCK_CYCLES}, "
        f"осечек: {len(markers) - markers.count(OK)} (допустимо {FAILURES_ALLOWED})"
    )
    print(f"  вердикт: {verdict}")


def parse_args(args: list[str]) -> tuple[int, str | None]:
    """Разбор аргументов: `(отсчёт, сообщение об ошибке)`.

    Явный `--countdown N` переопределяет дефолт (нужны короткие прогоны и тесты);
    неизвестные аргументы и отрицательный отсчёт — ошибка, а не молчаливое игнорирование.
    """
    countdown = win.START_COUNTDOWN
    rest = list(args)
    if "--countdown" in rest:
        index = rest.index("--countdown")
        if index + 1 >= len(rest):
            return countdown, "--countdown требует число секунд"
        raw = rest[index + 1]
        try:
            countdown = int(raw)
        except ValueError:
            return countdown, f"--countdown: не число секунд: {raw!r}"
        if countdown < 0:
            return countdown, f"--countdown: отрицательный отсчёт: {countdown}"
        del rest[index:index + 2]
    if rest:
        return countdown, f"неизвестные аргументы: {' '.join(rest)}"
    return countdown, None


def main(argv: list[str] | None = None) -> int:
    countdown, problem = parse_args(list(sys.argv[1:] if argv is None else argv))
    if problem is not None:
        print(problem)
        return 2

    print("ВНИМАНИЕ: прогон перехватит мышь — не трогайте манипулятор до конца сессии.")
    print("ВНИМАНИЕ: F12 — аварийная остановка в любой фазе; трасса блока сохраняется.")
    print(
        f"блоков по {BLOCK_CYCLES} циклов до отмены, таймаут поклёвки {BITE_TIMEOUT:.0f} с, "
        f"блок принят при >= {OK_REQUIRED} OK (допустимо {FAILURES_ALLOWED} осечки)"
    )
    print(
        f"предусловия: клиент в foreground, персонаж у воды, Auto Loot = ON, "
        f"Click-to-move = OFF, cast_key = {CAST_KEY!r}"
    )

    try:
        window = win.require_game_window()
    except win.WindowNotFound as exc:
        # это не маркер цикла: трасса ещё не создана, WIN_FAIL здесь не уместен
        print(f"окно клиента не найдено: {exc}")
        return 2
    print(f"клиент: {win.describe(window)}")

    if not win.start_countdown(window, countdown):
        return 2

    verdicts: list[str] = []
    stopped_early = False
    crashed = False
    block = 0

    with capture.Grabber() as grabber:
        # сцена и калибровка — один раз на сессию, до первого блока (design D5)
        try:
            scene = detect_scene(window, grabber)
        except CaptureFailed as exc:
            print(f"старт отменён: {exc}")
            return 1
        print(f"область поиска: {scene.where()}")

        try:
            calibrated, calibration_markers, calibration_trace = run_calibration(
                window, grabber, scene
            )
        except RunInterrupted as exc:
            print(f"прерван: {exc}")
            return 0
        except CaptureFailed as exc:
            print(f"WIN_FAIL в калибровке: {exc}")
            return 1

        params = DetectionParams.defaults() if calibrated is None else calibrated
        report_path = calibration_trace.with_name(calibration_trace.stem + "_report.md")
        print(f"отчёт калибровки: {report_path}")
        if WIN_FAIL in calibration_markers:
            print("сессия остановлена: отказ слоя окна или захвата в калибровке")
            return 1
        if calibrated is None:
            print(
                f"калибровка не набрала {CALIBRATION_TARGET_OK} OK за "
                f"{len(calibration_markers)} циклов: работаю на значениях по умолчанию"
            )
        print(f"параметры детекции: {params.summary()}")

        while True:
            if win.stop_requested():
                print("F12 до старта блока: сессия остановлена")
                stopped_early = True
                break
            block += 1
            print(f"\n=== блок {block}: {BLOCK_CYCLES} циклов ===")
            trace_path, markers, crashed, stop_reason = run_block(
                block, window, grabber, scene=scene, params=params
            )
            if stop_reason == "F12":
                # ручная остановка — не провал блока: вердикт в итог сессии не входит
                print_block_summary(block, trace_path, markers, "прерван")
                stopped_early = True
                break
            verdict = compute_verdict(
                markers, crashed=crashed, complete=len(markers) >= BLOCK_CYCLES
            )
            verdicts.append(verdict)
            print_block_summary(block, trace_path, markers, verdict)
            if crashed or stop_reason == "WIN_FAIL":
                stopped_early = True
                break
            print("--- следующий блок стартует сразу; F12 — остановка")

    print("\nитог сессии:")
    print(f"  блоков завершено: {len(verdicts)}")
    for index, verdict in enumerate(verdicts, start=1):
        print(f"    блок {index}: {verdict}")
    if stopped_early:
        print("  сессия остановлена досрочно")
    failed = crashed or any(verdict != "принято" for verdict in verdicts)
    if not verdicts:
        print("  сессия: прервана без завершённых блоков")
    else:
        print(f"  сессия: {'не принята' if failed else 'принята'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
