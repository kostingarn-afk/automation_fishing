"""Спайк 3: запись канала поклёвки и офлайн-подбор порога (задачи 4.4, 4.5).

Запись (по умолчанию 20 с, 20 Гц, каст автоматически):
    python probe_splash.py [--seconds 20] [--key 1] [--countdown 5]
    # -> probe/<дата>/frames/frame_00000.png ... + metadata.json

Подбор порога по одной или нескольким записям:
    python probe_splash.py --tune probe/<дата> [probe/<дата2> ...]
    # -> <каталог первой записи>/tuning_report.md

Метрика всплеска — локальная (design.md §10): число пикселей ярче порога в окне вокруг
поплавка. Глобальный порог не годится: пена занимает ~20 px при шуме воды 9–15 px,
а в окне у поплавка шум ±8 px и поклёвка даёт +26 px.

Коды выхода: 0 — успех; 1 — окно не найдено; 2 — фокус не в клиенте;
3 — нет данных для подбора; 4 — каст не отправлен; 5 — кадр не записан;
6 — подходящий набор не найден.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2

import capture
import input as input_layer
import loop10
import vision
import win

RECORD_SECONDS = 20.0
TARGET_FRAMES = 400
CAST_LEAD_SECONDS = 0.3
DEFAULT_CAST_KEY = "1"
DEFAULT_COUNTDOWN = win.START_COUNTDOWN

TUNE_THRESHOLDS = (130, 140, 150, 160, 170)
TUNE_RISES = (10, 15, 18, 22, 26, 30)
TUNE_HALF = vision.SPLASH_WINDOW_HALF
TUNE_WARMUP_FRAMES = 30
TUNE_BASELINE_FRAMES = 20
MIN_BURSTS = 5
EVENT_GAP_FRAMES = 3
MIN_EVENT_FRAMES = 2


def record(seconds: float, cast_key: str, countdown: int, repeat: int = 1) -> int:
    window = win.find_game_window()
    if window is None:
        print(
            f"окно не найдено: заголовок «{win.GAME_WINDOW_TITLE}», "
            f"классы {'/'.join(win.GAME_WINDOW_CLASSES)}"
        )
        return 1

    print(f"клиент: {win.describe(window)}")
    print(
        f"запись канала: {seconds:.1f} с при {capture.CAPTURE_HZ:.0f} Гц, "
        f"каст клавишей {cast_key!r}, повторов {repeat}"
    )
    print("нужен персонаж у воды с удочкой; F12 — аварийная остановка")

    if not win.start_countdown(window, countdown):
        return 2

    for attempt in range(1, repeat + 1):
        print(f"\n--- запись {attempt}/{repeat}")
        code = _record_once(window, seconds, cast_key)
        if code != 0:
            return code
    return 0


def _record_once(window: win.GameWindow, seconds: float, cast_key: str) -> int:
    directory = capture.run_dir()
    frames_dir = capture.ensure_dir(directory / "frames")
    print(f"каталог записи: {directory}")

    cast_at_ms: int | None = None
    cast_frame: int | None = None
    saved = 0
    shape: list[int] = []
    interrupted = False

    with capture.Grabber() as grabber:
        pacer = capture.Pacer()
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            if win.stop_requested():
                print("F12: запись прервана пользователем")
                interrupted = True
                break

            if cast_at_ms is None and time.monotonic() - started >= CAST_LEAD_SECONDS:
                try:
                    input_layer.cast(window, cast_key)
                except (input_layer.InputBlocked, OSError, ValueError) as exc:
                    print(f"каст не отправлен: {exc}")
                    return 4
                cast_at_ms = int((time.monotonic() - started) * 1000)
                cast_frame = saved
                print(f"каст отправлен на {cast_at_ms} мс, кадр {cast_frame}")

            frame = grabber.grab_roi(window)
            if frame is None:
                print(f"кадр {saved} не получен: {grabber.last_state}")
                pacer.wait()
                continue

            path = frames_dir / f"frame_{saved:05d}.png"
            if not capture.save_frame(frame, path):
                print(f"кадр не записан: {path}")
                return 5
            shape = [int(frame.shape[0]), int(frame.shape[1])]
            saved += 1
            pacer.wait()

    metadata = {
        "started_at": capture.timestamp(),
        "seconds": round(seconds, 2),
        "hz": capture.CAPTURE_HZ,
        "frames": saved,
        "frame_shape": shape,
        "cast_key": cast_key,
        "cast_at_ms": cast_at_ms,
        "cast_frame_index": cast_frame,
        "roi": win.roi_region(window).as_mss(),
        "roi_fraction_x": list(win.ROI_FRACTION_X),
        "roi_fraction_y": list(win.ROI_FRACTION_Y),
        "dpi_awareness": win.DPI_AWARENESS,
        "interrupted": interrupted,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"кадров записано: {saved}")
    if saved < TARGET_FRAMES:
        print(
            f"предупреждение: меньше {TARGET_FRAMES} кадров — повторите запись "
            f"с --seconds {seconds + 1:.0f}"
        )
    print(f"дальше: python probe_splash.py --tune {directory}")
    return 0


def _bobber_point(paths: list[Path], cast_frame: int) -> tuple[float, float] | None:
    """Позиция поплавка в ROI: устойчивый красный кластер, которого не было до каста.

    Голосование идёт по всей записи, а не до первых пяти голосов: иначе побеждает первый
    случайный кандидат, а поплавок живёт в кадре до конца канала и набирает больше всех.
    """
    baseline = cv2.imread(str(paths[0])) if paths else None
    if baseline is None:
        return None
    tracker = loop10.BobberTracker(baseline)
    for path in paths[cast_frame::2]:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        tracker.update(frame)
    return tracker.point() if tracker.seen() else None


def _series_all(
    paths: list[Path],
    point: tuple[float, float],
    thresholds: tuple[int, ...],
) -> dict[int, list[int]]:
    """Ряды ярких пикселей у поплавка по всем порогам за один проход по кадрам."""
    series: dict[int, list[int]] = {threshold: [] for threshold in thresholds}
    for path in paths:
        frame = cv2.imread(str(path))
        for threshold in thresholds:
            series[threshold].append(
                0 if frame is None else vision.bright_area(frame, point, TUNE_HALF, threshold)
            )
    return series


def _events(values: list[int], start: int, baseline: float, rise: int) -> list[tuple[int, int, int]]:
    """События всплеска: (первый кадр, последний кадр, пиковый прирост).

    Окна, разнесённые не дальше `EVENT_GAP_FRAMES`, склеиваются в одно событие, поэтому
    «длительность» здесь — размах события, а не длина непрерывной серии кадров выше базы.
    Рантайм (`loop10.BurstMonitor`) требует именно непрерывной серии (`CONFIRM_FRAMES = 2`),
    то есть критерий подбора мягче рантайма: числа отчёта нельзя читать как «столько-то
    кадров подряд». При переподборе на новых записях длину серии нужно считать отдельно.
    """
    events: list[tuple[int, int, int]] = []
    for index in range(start, len(values)):
        value = values[index]
        if value < baseline + rise or value < baseline * vision.SPLASH_RISE_RATIO:
            continue
        peak = int(value - baseline)
        if events and index - events[-1][1] <= EVENT_GAP_FRAMES:
            events[-1] = (events[-1][0], index, max(events[-1][2], peak))
        else:
            events.append((index, index, peak))
    return events


def tune(directories: list[Path]) -> int:
    if not directories:
        print("--tune требует хотя бы один каталог записи")
        return 3

    recordings = []
    for directory in directories:
        metadata_path = directory / "metadata.json"
        frames_dir = directory / "frames"
        if not metadata_path.exists() or not frames_dir.is_dir():
            print(f"нет записи канала: {directory} (нужны metadata.json и frames/)")
            return 3
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths = sorted(frames_dir.glob("frame_*.png"))
        if not paths:
            print(f"кадры не найдены: {frames_dir}")
            return 3
        cast_frame = int(metadata.get("cast_frame_index") or 0)
        point = _bobber_point(paths, cast_frame)
        if point is None:
            print(f"{directory}: поплавок не найден — запись пропущена")
            continue
        recordings.append((directory, paths, cast_frame, point))
        print(f"{directory}: кадров {len(paths)}, каст на {cast_frame}, поплавок {point}")

    if not recordings:
        print("ни в одной записи поплавок не найден — подбирать нечего")
        return 3

    combos = [(threshold, rise) for threshold in TUNE_THRESHOLDS for rise in TUNE_RISES]
    rows = []
    prepared = []
    for directory, paths, cast_frame, point in recordings:
        series = _series_all(paths, point, TUNE_THRESHOLDS)
        start = min(cast_frame + TUNE_WARMUP_FRAMES, max(len(paths) - 1, 0))
        prepared.append((directory, series, start))

    for threshold, rise in combos:
        total_events = 0
        weak = 0
        details = []
        for directory, series, start in prepared:
            values = series[threshold]
            baseline_window = values[start:start + TUNE_BASELINE_FRAMES]
            if not baseline_window:
                continue
            baseline = float(statistics.median(baseline_window))
            events = _events(values, start, baseline, rise)
            for first, last, peak in events:
                if last - first + 1 < MIN_EVENT_FRAMES:
                    weak += 1
                else:
                    total_events += 1
                    details.append((directory.name, first, last, peak, round(baseline)))
        rows.append(
            {
                "threshold": threshold,
                "rise": rise,
                "events": total_events,
                "weak": weak,
                "details": details,
            }
        )
        print(f"  порог {threshold}, прирост {rise}: событий {total_events}, одиночных кадров {weak}")

    qualified = [row for row in rows if row["events"] >= MIN_BURSTS and row["weak"] == 0]
    # Приоритет — значениям, уже зафиксированным в константах: они проверены на живом клиенте,
    # а правило «больше всплесков» тянет порог к ложным срабатываниям (design.md §11).
    fixed = next(
        (
            row
            for row in qualified
            if row["threshold"] == loop10.BURST_THRESHOLD
            and row["rise"] == loop10.SPLASH_RISE
        ),
        None,
    )
    chosen = fixed if fixed is not None else (
        max(qualified, key=lambda row: (row["events"], row["threshold"])) if qualified else None
    )

    lines = [
        "# Отчёт о подборе порога всплеска (задача 4.5)",
        "",
        f"- записей: {len(recordings)}",
        f"- метрика: пиксели ярче порога в окне {TUNE_HALF * 2 + 1}x{TUNE_HALF * 2 + 1} вокруг поплавка",
        f"- база: медиана {TUNE_BASELINE_FRAMES} кадров после {TUNE_WARMUP_FRAMES} кадров прогрева",
        f"- критерий: не менее {MIN_BURSTS} всплесков длиной от {MIN_EVENT_FRAMES} кадров, без одиночных кадров",
        "",
        "| burst_threshold | splash_rise | всплесков | одиночных кадров |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['threshold']} | {row['rise']} | {row['events']} | {row['weak']} |")

    lines += ["", "## Выбор", ""]
    if chosen is None:
        best = max(rows, key=lambda row: (row["events"], -row["weak"]))
        lines.append(
            f"Ни один набор не дал {MIN_BURSTS} всплесков без одиночных срабатываний. Лучший: "
            f"burst_threshold={best['threshold']}, splash_rise={best['rise']} "
            f"(всплесков {best['events']}, одиночных {best['weak']}). Демо не запускать: "
            "всплеск не отделяется от воды."
        )
    else:
        source = (
            "совпадает с константами `loop10.py` (задача 4.6)"
            if chosen is fixed
            else "перенести в константы `loop10.py` (задача 4.6)"
        )
        lines.append(
            f"Выбрано: burst_threshold={chosen['threshold']}, splash_rise={chosen['rise']}; "
            f"всплесков {chosen['events']}, одиночных срабатываний 0. Набор {source}."
        )
        lines += ["", "| запись | кадр | длительность, кадров | пиковый прирост | база |", "|---|---|---|---|---|"]
        for name, first, last, peak, baseline in chosen["details"]:
            lines.append(f"| {name} | {first} | {last - first + 1} | {peak} | {baseline} |")

    report_path = recordings[0][0] / "tuning_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-6:]))
    print(f"\nотчёт: {report_path}")
    return 0 if chosen is not None else 6


def main(argv: list[str]) -> int:
    seconds = RECORD_SECONDS
    cast_key = DEFAULT_CAST_KEY
    countdown = DEFAULT_COUNTDOWN
    repeat = 1
    tune_dirs: list[Path] = []

    args = argv[1:]
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--tune":
            if index + 1 >= len(args):
                print("--tune требует каталог записи")
                return 3
            tune_dirs.append(Path(args[index + 1]))
            index += 2
        elif argument == "--seconds":
            if index + 1 >= len(args):
                print("--seconds требует число")
                return 3
            seconds = float(args[index + 1])
            index += 2
        elif argument == "--key":
            if index + 1 >= len(args):
                print("--key требует клавишу")
                return 4
            cast_key = args[index + 1]
            index += 2
        elif argument == "--countdown":
            if index + 1 >= len(args):
                print("--countdown требует число секунд")
                return 3
            countdown = int(args[index + 1])
            index += 2
        elif argument == "--repeat":
            if index + 1 >= len(args):
                print("--repeat требует число")
                return 3
            repeat = max(1, int(args[index + 1]))
            index += 2
        else:
            print(f"неизвестный аргумент: {argument}")
            return 3

    if tune_dirs:
        return tune(tune_dirs)
    return record(seconds, cast_key, countdown, repeat)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
