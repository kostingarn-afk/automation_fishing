"""Спайк 2: принимает ли клиент инъекцию ввода (§3.4 п.2).

Проверка возможна только на реальной поклёвке: клик по поплавку до поклёвки клиент
игнорирует, а средняя разница кадров ROI от анимации воды составляет 3.5–3.6 (design.md §10).
Поэтому ветка проверяется одним циклом из `loop10.run_cycle`: каст -> ожидание всплеска ->
клик этой веткой -> поплавок исчез.

Ветки по возрастанию стоимости (план §5, спайк 2):
  (a) `SendInput` + `MOUSEEVENTF_ABSOLUTE` — основной путь, им же работает цикл
  (b) `SetCursorPos` + `mouse_event(RIGHTDOWN/RIGHTUP)` — устаревший путь
  (c) клавиатурный путь: подтверждается сам каст (без него поклёвки не было бы)
  (d) «схема не подтверждена» — штатный исход, а не провал работы

Запуск: `python probe_input.py [cast_key] [--countdown N]`

Коды выхода: 0 — ветка подтверждена; 1 — окно не найдено; 2 — фокус не в клиенте;
3 — схема не подтверждена (или поклёвки не было); 4 — F12.
"""

from __future__ import annotations

import sys

import capture
import input as input_layer
import loop10
import win

DEFAULT_KEY = "1"
DEFAULT_COUNTDOWN = win.START_COUNTDOWN


def legacy_branch(window, point: tuple[int, int], delay: tuple[float, float]) -> float:
    """Ветка (b): клик через `mouse_event` вместо `SendInput`."""
    return input_layer.strike(window, point, delay, sender=input_layer.send_right_click_legacy)


def main(argv: list[str]) -> int:
    args = argv[1:]
    key = DEFAULT_KEY
    countdown = DEFAULT_COUNTDOWN
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--countdown":
            if index + 1 >= len(args):
                print("--countdown требует число секунд")
                return 3
            countdown = int(args[index + 1])
            index += 2
        elif argument.startswith("--"):
            print(f"неизвестный аргумент: {argument}")
            return 3
        else:
            key = argument
            index += 1

    window = win.find_game_window()
    if window is None:
        print(
            f"окно не найдено: заголовок «{win.GAME_WINDOW_TITLE}», "
            f"классы {'/'.join(win.GAME_WINDOW_CLASSES)}"
        )
        return 1

    print(f"клиент: {win.describe(window)}")
    print(f"каст по {key!r}; каждая ветка проверяется одним циклом (до {loop10.BITE_TIMEOUT:.0f} с)")
    print("во время проверки не трогайте манипулятор; F12 — аварийная остановка")

    if not win.start_countdown(window, countdown):
        return 2

    branches = (
        ("(a) SendInput + MOUSEEVENTF_ABSOLUTE", input_layer.strike),
        ("(b) SetCursorPos + mouse_event", legacy_branch),
    )

    trace = loop10.Trace(capture.TRACE_ROOT / f"probe_input_{capture.timestamp()}.txt")
    markers: dict[str, str] = {}
    try:
        with capture.Grabber() as grabber:
            for name, click in branches:
                print(f"--- проверка ветки {name}")
                marker = loop10.run_cycle(1, window, grabber, trace, click=click, cast_key=key)
                markers[name] = marker
                if marker == loop10.OK:
                    print(f"\nитог: подтверждена ветка {name}")
                    return 0
                if marker == loop10.NO_STRIKE:
                    print(f"  {name}: клик отправлен, но поплавок остался — ветка не работает")
                elif marker in (loop10.MISS, loop10.LOST):
                    print(f"  {name}: поклёвки не было — ветка не проверена")
    except loop10.RunInterrupted as exc:
        print(f"\nпрерван: {exc}")
        return 4
    except (input_layer.InputBlocked, loop10.CaptureFailed) as exc:
        print(f"\nотказ во время проверки: {exc}")
        return 3
    finally:
        trace.close()

    print(f"\nтрасса: {trace.path}")
    if any(marker == loop10.NO_STRIKE for marker in markers.values()):
        print("итог: (c) клавиатурный путь подтверждён (поклёвка доходит до клиента),")
        print("      но клик мышью поплавок не убрал — демо без рабочего клика не запустится")
    else:
        print("итог: (d) схема не подтверждена — поклёвки не было, слои не проверить")
        print("      повторите запуск: нужна хотя бы одна поклёвка за 22 с")
    return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
