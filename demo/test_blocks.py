"""Офлайн-проверка блок-цикла `loop10.main`: без клиента, без ввода, без кадров.

Подменяет цикл, окно и захват, поэтому запускается на любой машине и не трогает игру.
Проверяет: правило вердикта, блок из 10 циклов, автоповтор следующего блока, «прерван»
при F12 и коды выхода.

Запуск: `python demo/test_blocks.py` (из любого каталога). Код выхода 0 — все сценарии прошли.
"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO))

import loop10  # noqa: E402


@contextmanager
def patched(target, **attributes):
    """Временно подменяет атрибуты модуля и возвращает их обратно."""
    saved = {name: getattr(target, name) for name in attributes}
    for name, value in attributes.items():
        setattr(target, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(target, name, value)


class FakeGrabber:
    """Заглушка захвата: `main()` открывает его контекстным менеджером."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


@contextmanager
def fake_environment(trace_dir: Path, run_cycle, verdicts: list[str], completed: dict):
    """Окружение `main()`: своя реализация цикла, окна, фокуса и каталога трасс."""
    original_summary = loop10.print_block_summary

    def spy(number, trace_path, markers, verdict):
        verdicts.append(verdict)
        completed["count"] += 1
        original_summary(number, trace_path, markers, verdict)

    def stop_requested() -> bool:
        # F12 «нажимается» между блоками после нужного их числа
        limit = completed.get("stop_after_blocks")
        return limit is not None and completed["count"] >= limit

    with patched(
        loop10.capture,
        Grabber=FakeGrabber,
        TRACE_ROOT=trace_dir,
    ), patched(
        loop10,
        run_cycle=run_cycle,
        print_block_summary=spy,
    ), patched(
        loop10.win,
        require_game_window=lambda: object(),
        describe=lambda window: "fake client",
        start_countdown=lambda window, seconds=0: True,
        stop_requested=stop_requested,
    ):
        yield


def run_session(markers: list[str], stop_after_blocks: int | None, trace_dir: Path):
    """Прогон `main()` на сценарии маркеров: `(код, вердикты блоков, файлы трасс, вывод)`."""
    pending = list(markers)
    verdicts: list[str] = []
    completed = {"count": 0, "stop_after_blocks": stop_after_blocks}

    def fake_run_cycle(cycle, window, grabber, trace, click=None, cast_key=None, block=1):
        marker = pending.pop(0) if pending else loop10.OK
        trace.write(cycle, 0, "0", "-", "-", marker)
        return marker

    output = io.StringIO()
    with fake_environment(trace_dir, fake_run_cycle, verdicts, completed), redirect_stdout(output):
        code = loop10.main(["--countdown", "0"])

    return code, verdicts, sorted(path.name for path in trace_dir.glob("run_*.txt")), output.getvalue()


def run_interrupted_session(trace_dir: Path, run_cycle, verdicts: list[str]) -> tuple[int, str]:
    """Прогон, который прерывается изнутри цикла (имитация F12 во время ожидания)."""
    completed = {"count": 0, "stop_after_blocks": None}
    output = io.StringIO()
    with fake_environment(trace_dir, run_cycle, verdicts, completed), redirect_stdout(output):
        code = loop10.main(["--countdown", "0"])
    return code, output.getvalue()


def scenario_verdict_rule() -> tuple[bool, str, str]:
    """Вердикт блока: 8 из 10 — «принято», 7 — «не принято», WIN_FAIL и авария — «не принято»."""
    cases = [
        ("8 OK + 2 LOST", loop10.compute_verdict([loop10.OK] * 8 + [loop10.LOST] * 2), "принято"),
        ("10 OK", loop10.compute_verdict([loop10.OK] * 10), "принято"),
        ("7 OK + 3 LOST", loop10.compute_verdict([loop10.OK] * 7 + [loop10.LOST] * 3), "не принято"),
        (
            "9 OK + WIN_FAIL в 10-м цикле",
            loop10.compute_verdict([loop10.OK] * 9 + [loop10.WIN_FAIL]),
            "не принято",
        ),
        (
            "9 OK, блок неполный",
            loop10.compute_verdict([loop10.OK] * 9 + [loop10.LOST], complete=False),
            "не принято",
        ),
        ("авария", loop10.compute_verdict([loop10.OK] * 10, crashed=True), "не принято"),
    ]
    failed = [title for title, verdict, expected in cases if verdict != expected]
    details = f"проверок {len(cases)}, не прошли: {failed}" if failed else f"проверок {len(cases)}"
    return not failed, details, ""


def scenario_full_blocks() -> tuple[bool, str, str]:
    """Два полных блока (8 OK + 2 LOST, затем 10 OK): оба приняты, две трассы, код 0."""
    with TemporaryDirectory() as tmp:
        trace_dir = Path(tmp)
        markers = [loop10.OK] * 8 + [loop10.LOST] * 2 + [loop10.OK] * 10
        code, verdicts, traces, output = run_session(
            markers, stop_after_blocks=2, trace_dir=trace_dir
        )
    passed = code == 0 and verdicts == ["принято", "принято"] and len(traces) == 2
    return passed, f"код {code}, вердикты {verdicts}, трасс {len(traces)}", output


def scenario_failed_block() -> tuple[bool, str, str]:
    """7 OK + 3 LOST — блок «не принято», сессия «не принята», код 1."""
    with TemporaryDirectory() as tmp:
        trace_dir = Path(tmp)
        code, verdicts, _, output = run_session(
            [loop10.OK] * 7 + [loop10.LOST] * 3, stop_after_blocks=1, trace_dir=trace_dir
        )
    passed = code == 1 and verdicts == ["не принято"]
    return passed, f"код {code}, вердикты {verdicts}", output


def scenario_interrupted_block() -> tuple[bool, str, str]:
    """F12 на 5-м цикле: блок «прерван», в итог сессии не входит, код 0."""
    calls = {"count": 0}

    def short_cycle(cycle, window, grabber, trace, click=None, cast_key=None, block=1):
        calls["count"] += 1
        if calls["count"] > 4:
            raise loop10.RunInterrupted("F12 в тесте")
        trace.write(cycle, 0, "0", "-", "-", loop10.OK)
        return loop10.OK

    verdicts: list[str] = []
    with TemporaryDirectory() as tmp:
        code, output = run_interrupted_session(Path(tmp), short_cycle, verdicts)
    passed = code == 0 and verdicts == ["прерван"] and calls["count"] == 5
    return passed, f"код {code}, вердикты {verdicts}, циклов {calls['count']}", output


def scenario_window_failure() -> tuple[bool, str, str]:
    """Отказ слоя окна в первом цикле: блок «не принято», сессия останавливается, код 1."""

    def failing_cycle(cycle, window, grabber, trace, click=None, cast_key=None, block=1):
        raise loop10.input_layer.InputBlocked("окно не в фокусе")

    verdicts: list[str] = []
    with TemporaryDirectory() as tmp:
        code, output = run_interrupted_session(Path(tmp), failing_cycle, verdicts)
    passed = code == 1 and verdicts == ["не принято"]
    return passed, f"код {code}, вердикты {verdicts}", output


def quiet(call):
    """Вызов с подавлением stdout: сценарии не должны печатать прогон `main()`."""
    with redirect_stdout(io.StringIO()):
        return call()


def scenario_cli_and_window_errors() -> tuple[bool, str, str]:
    """Коды 2: неверные аргументы, окно не найдено, фокус не возвращён."""
    cases = [
        ("--countdown abc", quiet(lambda: loop10.main(["--countdown", "abc"])), 2),
        ("--countdown", quiet(lambda: loop10.main(["--countdown"])), 2),
        ("--countdown -1", quiet(lambda: loop10.main(["--countdown", "-1"])), 2),
        ("--bogus", quiet(lambda: loop10.main(["--bogus"])), 2),
    ]

    def no_window():
        raise loop10.win.WindowNotFound("окно не найдено")

    with patched(loop10.win, require_game_window=no_window):
        cases.append(("окно не найдено", quiet(lambda: loop10.main(["--countdown", "0"])), 2))

    with patched(
        loop10.win,
        require_game_window=lambda: object(),
        describe=lambda window: "fake client",
        start_countdown=lambda window, seconds=0: False,
    ):
        cases.append(("фокус не возвращён", quiet(lambda: loop10.main(["--countdown", "0"])), 2))

    failed = [title for title, code, expected in cases if code != expected]
    details = f"проверок {len(cases)}, не прошли: {failed}" if failed else f"проверок {len(cases)}"
    return not failed, details, ""


def main() -> int:
    scenarios = (
        ("правило вердикта", scenario_verdict_rule),
        ("два полных блока", scenario_full_blocks),
        ("блок с 3 осечками", scenario_failed_block),
        ("F12 в середине блока", scenario_interrupted_block),
        ("отказ слоя окна", scenario_window_failure),
        ("коды 2 и аргументы", scenario_cli_and_window_errors),
    )
    failures = 0
    for title, scenario in scenarios:
        passed, details, output = scenario()
        print(f"{'OK    ' if passed else 'ПРОВАЛ'} {title}: {details}")
        if not passed and output:
            print("\n--- вывод прогона ---")
            print(output)
        failures += 0 if passed else 1
    print(f"\nитог: {'все сценарии прошли' if failures == 0 else f'провалов {failures}'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
