"""Спайк 1.5: стартовый контракт, один каст и контроль окна лута.

Закрывает §3.4 п.8: рабочий `cast_key`, `Auto Loot = ON`, `Click-to-move = OFF`,
а также стартовый контракт из §4.1 плана.
Запуск: `python probe_preflight.py [cast_key]` (по умолчанию «1»).

Коды выхода: 0 — pre-flight пройден; 1 — окно не найдено; 2 — фокус не в клиенте;
3 — ROI не получен; 4 — каст не отправлен; 5 — F12; 6 — открылось окно лута;
7 — сцена после каста не изменилась.

Замечание: в клиенте 3.3.5 окно лута рисуется внутри игрового окна, поэтому
«появление окна лута» проверяется по новым top-level окнам процесса — на репаке
их обычно нет. Изменение ROI после каста (появление поплавка) печатается как
косвенное подтверждение, что каст дошёл до клиента.
"""

from __future__ import annotations

import sys
import time

import capture
import input as input_layer
import vision
import win

DEFAULT_CAST_KEY = "1"
ROI_SAMPLE_SECONDS = 3.0
# Порог смены сцены после каста, а не детекции поклёвки: `vision.frame_delta` усредняет
# разницу по всему ROI 576x324, поэтому всплеск (~21 px) даёт вклад порядка 0.01 и норму
# не меняет. Шум анимации воды — 3.5–3.6 (design.md §10), смена камеры/позиции — заметно
# больше, поэтому порог выставлен на 8.0.
SCENE_CHANGE_THRESHOLD = 8.0


def main(argv: list[str]) -> int:
    cast_key = argv[1] if len(argv) > 1 else DEFAULT_CAST_KEY
    window = win.find_game_window()
    if window is None:
        print(
            f"окно не найдено: заголовок «{win.GAME_WINDOW_TITLE}», "
            f"классы {'/'.join(win.GAME_WINDOW_CLASSES)}"
        )
        return 1

    print(f"клиент: {win.describe(window)}")
    print("предусловия на персонаже: у воды, удочка в сумке, Auto Loot = ON, Click-to-move = OFF")
    print(f"проверяемая клавиша каста: {cast_key!r}")

    if not win.start_countdown(window):
        return 2
    print("шаг 1/4: фокус в клиенте — ок")

    windows_before = {window.hwnd for window in win.iter_top_level_windows()}

    with capture.Grabber() as grabber:
        baseline = grabber.grab_roi(window)
        if baseline is None:
            print(f"шаг 2/4: ROI не получен ({grabber.last_state})")
            return 3
        print(f"шаг 2/4: базовый ROI {baseline.shape[1]}x{baseline.shape[0]} — ок")

        try:
            input_layer.cast(window, cast_key)
        except (input_layer.InputBlocked, OSError, ValueError) as exc:
            print(f"шаг 3/4: каст не отправлен: {exc}")
            return 4
        cast_at = time.monotonic()
        print("шаг 3/4: каст отправлен")

        pacer = capture.Pacer()
        peak = 0.0
        bobber_frames = 0
        samples = 0
        while time.monotonic() - cast_at < ROI_SAMPLE_SECONDS:
            if win.stop_requested():
                print("F12: остановка по требованию пользователя")
                return 5
            frame = grabber.grab_roi(window)
            if frame is not None:
                samples += 1
                peak = max(peak, vision.frame_delta(baseline, frame))
                # точка поплавка на pre-flight неизвестна, поэтому сигнал грубый:
                # любой красный кластер в ROI, включая берег и доспех персонажа
                if vision.bobber_present(frame):
                    bobber_frames += 1
            pacer.wait()

    scene_changed = peak >= SCENE_CHANGE_THRESHOLD
    print(
        f"шаг 4/4: отклонение ROI {peak:.2f} при пороге {SCENE_CHANGE_THRESHOLD}; "
        f"кадров с красным кластером: {bobber_frames}/{samples}"
    )

    new_windows = [w for w in win.iter_top_level_windows() if w.hwnd not in windows_before]
    loot = [w for w in new_windows if "loot" in f"{w.class_name} {w.title}".casefold()]
    if loot:
        print("окно лута: появилось")
        for w in loot:
            print(f"  {w.hwnd:#010x} class={w.class_name} «{w.title}»")
        print("итог: pre-flight не пройден — окно добычи открылось, значит Auto Loot выключен")
        return 6

    print("окно лута: не появилось")
    if not scene_changed:
        print("итог: сцена после каста не изменилась — каст не подтверждён; pre-flight не пройден")
        return 7

    print("итог: pre-flight пройден (каст отправлен, окно лута не открылось)")
    print("проверьте в игре: канал Fishing начался, Click-to-move выключен")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
