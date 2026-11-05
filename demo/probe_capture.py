"""Спайк 1: окна -> класс/PID/exe -> GetClientRect -> ClientToScreen -> PNG.

Закрывает вопросы §3.4 п.1, п.7 плана и «не чёрный ли кадр».
Запуск: `python probe_capture.py`

Коды выхода: 0 — окно найдено и PNG записан; 1 — окно не найдено;
2 — кадр не получен; 3 — PNG не записан.
"""

from __future__ import annotations

import sys

import capture
import win


def main() -> int:
    print(f"DPI-awareness: {win.DPI_AWARENESS}")

    print("\ntop-level окна с заголовком:")
    for window in win.iter_top_level_windows():
        print(
            f"  {window.hwnd:#010x} {window.class_name:<24} "
            f"pid={window.pid:<6} {window.exe or '<exe недоступен>'}"
            f"  «{window.title}»"
        )

    window = win.find_game_window()
    if window is None:
        print(
            f"\nокно не найдено: заголовок «{win.GAME_WINDOW_TITLE}», "
            f"классы {'/'.join(win.GAME_WINDOW_CLASSES)}"
        )
        return 1

    print(f"\nклиент: {win.describe(window)}")
    try:
        width, height = win.client_size(window)
        origin = win.client_origin(window)
        region = win.roi_region(window)
    except (OSError, ValueError) as exc:
        print(f"геометрия окна недоступна: {exc}")
        return 2

    print(f"клиентская область: {width}x{height}; экранная позиция: {origin}")
    print(
        f"ROI X {win.ROI_FRACTION_X} Y {win.ROI_FRACTION_Y}: left={region.left} top={region.top} "
        f"size={region.width}x{region.height}"
    )

    with capture.Grabber() as grabber:
        frame = grabber.grab_client(window)
        if frame is None:
            print(f"кадр не получен: {grabber.last_state}")
            return 2
        print(
            f"кадр: {frame.shape[1]}x{frame.shape[0]}, "
            f"mean={float(frame.mean()):.1f}, std={float(frame.std()):.1f}, "
            f"состояние={capture.frame_state(frame)}"
        )
        path = capture.run_dir() / "capture_client.png"
        if not capture.save_frame(frame, path):
            print(f"PNG не записан: {path}")
            return 3
        print(f"PNG: {path}")

        # факты спайка 1 (задача 1.4): класс, владелец и размер должны остаться в записи
        report = path.with_suffix(".txt")
        report.write_text(
            "\n".join(
                (
                    f"window_hwnd={window.hwnd:#x}",
                    f"window_class={window.class_name}",
                    f"window_title={window.title}",
                    f"pid={window.pid}",
                    f"exe={window.exe}",
                    f"client={width}x{height}",
                    f"origin={origin[0]},{origin[1]}",
                    f"roi={region.left},{region.top} {region.width}x{region.height}",
                    f"frame={frame.shape[1]}x{frame.shape[0]}",
                    f"mean={float(frame.mean()):.1f} std={float(frame.std()):.1f}",
                    f"dpi_awareness={win.DPI_AWARENESS}",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"факты: {report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
