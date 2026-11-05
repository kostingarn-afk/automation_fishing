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

# Подобраны офлайн по 10 записям водной сцены (задача 4.5): 8 всплесков, 0 одиночных
# срабатываний. Отчёт воспроизводится командой
# `python probe_splash.py --tune <каталоги записей>` (пишется в каталог первой записи).
# Прирост 10 отвергнут: в живом прогоне качание поплавка давало +10 при базе 54
# и ложную подсечку, поэтому минимальный прирост — 18.
BURST_THRESHOLD = 130
SPLASH_RISE = 18
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
    ) -> None:
        """Строка трассы: моменты фаз в мс от каста, ровно один маркер и причина отказа.

        Причина — отдельная колонка: моменты содержат только числа или `-`, поэтому строку
        можно разбирать позиционно, а диагностическая причина (`NO_BOBBER`, `NO_BURST`, ...)
        не подменяет момент.
        """
        if marker not in MARKERS:
            raise ValueError(f"неизвестный маркер: {marker}")
        line = (
            f"#{cycle:02d} | cast {cast_ms} | burst {burst} | strike {strike} | "
            f"gone {gone} | {marker} | {reason}"
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

    def __init__(self, baseline) -> None:
        known = vision.find_tracker_candidates(baseline) if baseline is not None else []
        self._known = [(blob.cx, blob.cy) for blob in known]
        self._votes: list[list[float]] = []

    def _is_new(self, blob: vision.Blob) -> bool:
        return all(
            math.hypot(blob.cx - cx, blob.cy - cy) > self.MATCH_RADIUS for cx, cy in self._known
        )

    def update(self, frame) -> None:
        for blob in vision.find_tracker_candidates(frame):
            if not self._is_new(blob):
                continue
            for entry in self._votes:
                if math.hypot(blob.cx - entry[0], blob.cy - entry[1]) <= self.MATCH_RADIUS:
                    entry[0], entry[1] = blob.cx, blob.cy
                    entry[2] += 1
                    break
            else:
                self._votes.append([blob.cx, blob.cy, 1.0])

    def seen(self) -> bool:
        return any(entry[2] >= self.MIN_VOTES for entry in self._votes)

    def point(self) -> tuple[float, float] | None:
        best = max(self._votes, key=lambda entry: entry[2], default=None)
        return (best[0], best[1]) if best else None

    def confirmed(self, frame, radius: float = 12.0) -> vision.Blob | None:
        """Строгий кластер у выбранной точки — проверка, что трекер не ушёл на блик.

        `radius` шире радиуса голосования: поплавок качается на воде на единицы пикселей,
        а пропуск присутствия даёт ложный `LOST`.
        """
        point = self.point()
        if point is None:
            return None
        for blob in vision.find_tracker_candidates(frame):
            if math.hypot(blob.cx - point[0], blob.cy - point[1]) <= radius:
                return blob
        return None


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

    def __init__(self, point: tuple[float, float], threshold: int = BURST_THRESHOLD) -> None:
        self.point = point
        self._threshold = threshold
        self._warmup = 0
        self._samples: list[int] = []
        self._above = 0
        self.baseline: float | None = None
        self.last = 0

    def area(self, frame) -> int:
        self.last = vision.bright_area(frame, self.point, threshold=self._threshold)
        return self.last

    def observe(self, frame) -> bool:
        """True — яркость у поплавка держится выше базы `CONFIRM_FRAMES` кадров подряд."""
        value = self.area(frame)
        if self.baseline is not None:
            above = value >= self.baseline + self.RISE and value >= self.baseline * self.RISE_RATIO
            self._above = self._above + 1 if above else 0
            return self._above >= self.CONFIRM_FRAMES
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
        return current <= self.baseline + self.RISE * 0.5


def wait_bobber_gone(
    grabber: capture.Grabber,
    window: win.GameWindow,
    monitor: BurstMonitor,
    seconds: float,
) -> bool:
    """Исход подтверждён, если всплеск погас, а поплавок в окне проверки не появился.

    В момент всплеска поплавок закрыт пеной, поэтому «поплавок отсутствует» само по себе
    ничего не значит: сначала ждём, пока яркость у точки вернётся к базе, и только потом
    проверяем красный кластер. Окно задаёт вызывающий код: автолут закрывает окно добычи
    за 1.5–3 с, и до этого поплавок ещё виден.
    """
    pacer = capture.Pacer()
    cleared = 0
    gone_seen = False
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if win.stop_requested():
            raise RunInterrupted("F12 на фазе верификации")
        frame = grabber.grab_roi(window)
        if frame is None:
            raise CaptureFailed(f"ROI не получен на фазе верификации: {grabber.last_state}")
        if monitor.quiet(monitor.area(frame)):
            cleared += 1
            if cleared >= GONE_CONFIRM_FRAMES and not vision.bobber_present(
                frame, near=monitor.point
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


def run_cycle(
    number: int,
    window: win.GameWindow,
    grabber: capture.Grabber,
    trace: Trace,
    click=None,
    cast_key: str | None = None,
    block: int = 1,
) -> str:
    """Один цикл «каст -> ожидание -> подсечка -> верификация».

    `click` — способ подсечки: по умолчанию `input_layer.strike` (ветка a),
    probe-скрипт подставляет ветку (b). `block` — номер блока, попадает в имена кадров
    отказов. Возвращает маркер цикла и пишет одну строку трассы.
    """
    clicker = input_layer.strike if click is None else click
    cast = CAST_KEY if cast_key is None else cast_key
    started = time.monotonic()

    def elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    # окно могло закрыться или уехать между циклами — это отказ слоя окна, не авария
    try:
        width, height = win.client_size(window)
        roi_left, roi_top, roi_width, roi_height = win.roi_box(width, height)
        origin_x, origin_y = win.client_origin(window)
    except (OSError, ValueError) as exc:
        raise CaptureFailed(f"геометрия окна недоступна: {exc}") from exc
    region = win.Region(origin_x + roi_left, origin_y + roi_top, roi_width, roi_height)
    baseline = grabber.grab_roi(window)
    if baseline is None:
        raise CaptureFailed(f"ROI не получен до каста: {grabber.last_state}")

    try:
        input_layer.cast(window, cast)
    except input_layer.InputBlocked as exc:
        # фокус ушёл до каста: для разбора отказа прикладываем кадр до каста
        raise input_layer.InputBlocked(str(exc), baseline) from exc
    print(f"#{number:02d}: каст отправлен (ROI {roi_width}x{roi_height})")

    deadline = started + BITE_TIMEOUT
    acquire_deadline = started + ACQUIRE_TIMEOUT
    last_frame = baseline
    tracker = BobberTracker(baseline)
    monitor: BurstMonitor | None = None
    burst_ms: int | None = None
    burst_point: tuple[int, int] | None = None
    missing_frames = 0
    pacer = capture.Pacer()

    def finish_failure(marker: str, reason: str, message: str) -> str:
        moment = elapsed_ms()
        capture.save_fail_frame(last_frame, number, moment, marker, block=block)
        # моменты фаз — только числа или `-`; причина — токен, момент отказа — суффикс `@<мс>`
        trace.write(number, 0, "-", "-", "-", marker, reason=f"{reason}@{moment}")
        print(f"#{number:02d}: {marker} — {message}")
        return marker

    while time.monotonic() < deadline:
        if win.stop_requested():
            raise RunInterrupted(f"F12 в цикле {number} на фазе ожидания поклёвки")
        frame = grabber.grab_roi(window)
        if frame is None:
            raise CaptureFailed(
                f"ROI не получен на фазе ожидания: {grabber.last_state}", last_frame
            )
        last_frame = frame
        tracker.update(frame)
        if monitor is None:
            if tracker.seen() and tracker.confirmed(frame) is not None:
                monitor = BurstMonitor(tracker.point())
                print(f"#{number:02d}: поплавок найден в ROI ({monitor.point[0]:.0f},{monitor.point[1]:.0f})")
            elif time.monotonic() >= acquire_deadline:
                # каст не прошёл либо поплавок вне ROI: ждать 22 с бессмысленно
                return finish_failure(
                    MISS,
                    "NO_BOBBER",
                    f"поплавок не появился за {ACQUIRE_TIMEOUT:.0f} с после каста",
                )
        if monitor is not None and monitor.observe(frame):
            burst_ms = elapsed_ms()
            burst_point = region.to_screen(monitor.point[0], monitor.point[1])
            # центроид пены — только диагностика: клик идёт по позиции поплавка
            splash = (
                vision.bright_centroid(frame, monitor.point, threshold=BURST_THRESHOLD)
                or monitor.point
            )
            print(
                f"#{number:02d}: всплеск на {burst_ms} мс, поплавок "
                f"({monitor.point[0]:.0f},{monitor.point[1]:.0f}), "
                f"ярких пикселей {monitor.last} при базе {monitor.baseline:.0f}, "
                f"центроид пены ({splash[0]:.0f},{splash[1]:.0f}), точка клика {burst_point}"
            )
            break
        if monitor is not None:
            # проверка строго после всплеска: пена, закрывшая поплавок, уже дала бы burst
            if tracker.confirmed(frame) is None:
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

    strike_ms: int | None = None
    gone_ms: int | None = None
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
        if wait_bobber_gone(grabber, window, monitor, verify_seconds):
            gone_ms = elapsed_ms()
            break
        print(f"#{number:02d}: исход не подтверждён за {verify_seconds:.1f} с")
        if attempt < STRIKE_RETRIES:
            time.sleep(STRIKE_INTERVAL)

    if gone_ms is not None:
        trace.write(number, 0, str(burst_ms), str(strike_ms), str(gone_ms), OK)
        print(f"#{number:02d}: OK — поплавок исчез на {gone_ms} мс")
        return OK

    frame = grabber.grab_roi(window)
    if frame is None:
        frame = last_frame
    capture.save_fail_frame(frame, number, elapsed_ms(), NO_STRIKE, block=block)
    trace.write(
        number, 0, str(burst_ms), str(strike_ms), "-", NO_STRIKE, reason="NO_CONFIRM"
    )
    print(f"#{number:02d}: NO_STRIKE — за {STRIKE_RETRIES} попытки исход не подтверждён")
    return NO_STRIKE


def run_block(number: int, window: win.GameWindow, grabber: capture.Grabber) -> tuple:
    """Один блок из `BLOCK_CYCLES` циклов: своя трасса, маркеры и причина остановки.

    Возвращает `(trace_path, markers, crashed, stop_reason)`, где `stop_reason` — `None`
    (блок прошёл все циклы), `"F12"` (остановлен пользователем) или `"WIN_FAIL"` (отказ слоя
    окна или захвата). Различие важно: `F12` — это выбор пользователя, а не провал блока.
    """
    trace_path = capture.TRACE_ROOT / f"run_{capture.timestamp()}_b{number:02d}.txt"
    trace = Trace(trace_path)
    markers: list[str] = []
    crashed = False
    stop_reason: str | None = None

    try:
        for cycle in range(1, BLOCK_CYCLES + 1):
            print(f"--- блок {number}, цикл {cycle}/{BLOCK_CYCLES}")
            started = time.monotonic()
            try:
                marker = run_cycle(cycle, window, grabber, trace, block=number)
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
        while True:
            if win.stop_requested():
                print("F12 до старта блока: сессия остановлена")
                stopped_early = True
                break
            block += 1
            print(f"\n=== блок {block}: {BLOCK_CYCLES} циклов ===")
            trace_path, markers, crashed, stop_reason = run_block(block, window, grabber)
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
