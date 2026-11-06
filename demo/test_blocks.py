"""Офлайн-проверка блок-цикла `loop10.main`: без клиента, без ввода, без кадров.

Подменяет цикл, окно и захват, поэтому запускается на любой машине и не трогает игру.
Проверяет: правило вердикта, блок из 10 циклов, автоповтор следующего блока, «прерван»
при F12 и коды выхода; прыжок — клавиша `space` с удержанием, момент `reason@<мс>` до
пауз, слот 3–10 и ровно один прыжок за блок, отказы ввода без смены маркера, колонку
`jump` в строке трассы.

Запуск: `python demo/test_blocks.py` (из любого каталога). Код выхода 0 — все сценарии прошли.
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

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
def fake_environment(
    trace_dir: Path,
    run_cycle,
    verdicts: list[str],
    completed: dict,
    detect_scene=None,
    calibration=None,
):
    """Окружение `main()`: своя реализация цикла, окна, фокуса и каталога трасс."""
    original_summary = loop10.print_block_summary
    scene_factory = detect_scene or (
        lambda window, grabber: loop10.Scene(region=loop10.win.Region(0, 0, 100, 50))
    )
    calibration_run = calibration or (
        lambda window, grabber, scene: (None, [], trace_dir / "calibration_fake.txt")
    )

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
        detect_scene=scene_factory,
        run_calibration=calibration_run,
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

    def fake_run_cycle(cycle, window, grabber, trace, click=None, cast_key=None, block=1, jump=False, scene=None, params=None):
        marker = pending.pop(0) if pending else loop10.OK
        trace.write(cycle, 0, "0", "-", "-", marker, jump="2500" if jump else "-")
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


class FixedRandom:
    """Заглушка `random` для слоя ввода: удержание клавиши задано сценарием."""

    def __init__(self, value: float) -> None:
        self.value = value

    def uniform(self, low: float, high: float) -> float:
        return self.value


def scenario_cast_packet() -> tuple[bool, str, str]:
    """Каст после правки `send_key`: один пакет `down`+`up` с VK клавиши «1»."""
    calls: list[object] = []
    input_layer = loop10.input_layer

    def fake_send(inputs):
        calls.append(inputs)
        return True

    with patched(input_layer, _send=fake_send):
        sent = input_layer.send_key("1")

    vk = [item.ki.wVk for item in calls[0]] if calls else []
    flags = [item.ki.dwFlags for item in calls[0]] if calls else []
    passed = (
        bool(sent)
        and len(calls) == 1
        and vk == [0x31, 0x31]
        and flags == [0, input_layer.KEYEVENTF_KEYUP]
    )
    return passed, f"пакетов {len(calls)}, VK {vk}, флаги {flags}", ""


def scenario_jump_key() -> tuple[bool, str, str]:
    """Прыжок: `VK_SPACE` двумя вызовами `_send` с паузой внутри `KEY_HOLD`."""
    calls: list[tuple[object, float]] = []

    def fake_send(inputs):
        calls.append((inputs, time.monotonic()))
        return True

    hold = 0.09
    input_layer = loop10.input_layer
    with patched(
        input_layer,
        _send=fake_send,
        require_foreground=lambda window: None,
        random=FixedRandom(hold),
    ):
        input_layer.jump(object())

    flags = [inputs[0].ki.dwFlags for inputs, _ in calls]
    vk_ok = all(
        len(inputs) == 1 and inputs[0].ki.wVk == input_layer.VK_SPACE for inputs, _ in calls
    )
    pause = calls[1][1] - calls[0][1] if len(calls) == 2 else -1.0
    low, high = input_layer.KEY_HOLD
    passed = (
        len(calls) == 2
        and vk_ok
        and flags == [0, input_layer.KEYEVENTF_KEYUP]
        and low <= pause <= high + 0.02
    )
    return passed, f"вызовов {len(calls)}, пауза {pause:.3f} с, флаги {flags}", ""


def hsv_bgr(hue: int, saturation: int, value: int) -> tuple[int, int, int]:
    """BGR-цвет по HSV: синтетические кадры задаются цветом, а не порядком каналов."""
    patch = np.uint8([[[hue, saturation, value]]])
    blue, green, red = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0]
    return int(blue), int(green), int(red)


def scene_frame(
    water_hsv: tuple[int, int, int] | None = (90, 150, 65),
    ground_hsv: tuple[int, int, int] = (40, 150, 90),
    size: tuple[int, int] = (200, 300),
    water_box: tuple[int, int, int, int] = (40, 20, 220, 140),
    red_patch: tuple[int, int] | None = None,
) -> np.ndarray:
    """Синтетический кадр клиента: земля, вода и красное пятно на земле."""
    height, width = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = hsv_bgr(*ground_hsv)
    if water_hsv is not None:
        left, top, box_width, box_height = water_box
        frame[top : top + box_height, left : left + box_width] = hsv_bgr(*water_hsv)
    if red_patch is not None:
        x, y = red_patch
        frame[y : y + 6, x : x + 6] = hsv_bgr(0, 200, 210)
    return frame


class FakeClientGrabber:
    """Заглушка захвата для старта сессии: отдаёт заданный кадр клиента."""

    last_state = "fake"

    def __init__(self, frame) -> None:
        self.frame = frame

    def grab_client(self, window):
        return self.frame


def mosaic_frame(hues: tuple[int, ...] = (0, 16, 32, 48, 144, 160)) -> np.ndarray:
    """Кадр из равных полос разных оттенков вне пределов воды: преобладающего цвета нет."""
    height, width = 200, 300
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    stripe = width // len(hues)
    for index, hue in enumerate(hues):
        left = index * stripe
        frame[:, left : left + stripe] = hsv_bgr(hue, 150, 90)
    frame[:, stripe * len(hues) :] = hsv_bgr(hues[-1], 150, 90)
    return frame


def scenario_water_mask() -> tuple[bool, str, str]:
    """Маска воды: дневная и тёмная сцена, мутная вода, кадр без преобладающего цвета."""
    day = scene_frame((90, 150, 120), red_patch=(15, 55))
    night = scene_frame((88, 150, 55))
    murky = scene_frame((25, 115, 90))
    dry = mosaic_frame()

    assumed = loop10.vision.detect_water(day)
    dark = loop10.vision.detect_water(night)
    brown = loop10.vision.detect_water(murky)
    empty = loop10.vision.detect_water(dry)

    passed = (
        assumed is not None
        and abs(assumed.left - 40) <= 2
        and abs(assumed.top - 20) <= 2
        and abs(assumed.width - 220) <= 4
        and abs(assumed.height - 140) <= 4
        and bool(assumed.mask[80, 100])
        and not bool(assumed.mask[55, 15])
        and dark is not None
        and brown is not None
        and brown.width >= 200
        and empty is None
    )
    details = (
        f"дневная {None if assumed is None else (assumed.left, assumed.top, assumed.width, assumed.height)}, "
        f"тёмная {'найдена' if dark is not None else 'нет'}, "
        f"мутная {None if brown is None else (brown.left, brown.top, brown.width, brown.height)}, "
        f"без воды {'None' if empty is None else 'найдена'}"
    )
    return passed, details, ""


def scenario_water_scene() -> tuple[bool, str, str]:
    """Сцена сессии: область по воде в клиентских координатах, без воды — фолбэк."""
    frame = scene_frame()
    with patched(
        loop10.win,
        client_origin=lambda window: (100, 50),
        client_size=lambda window: (300, 200),
        roi_region=lambda window: loop10.win.Region(100, 50, 576, 324),
    ):
        with redirect_stdout(io.StringIO()):
            scene = loop10.detect_scene(object(), FakeClientGrabber(frame))
            fallback = loop10.detect_scene(object(), FakeClientGrabber(mosaic_frame()))

    region = scene.region
    center_screen = region.to_screen(region.width / 2.0, region.height / 2.0)
    expected_center = (
        100 + 40 + 8 + region.width // 2,
        50 + 20 + 8 + region.height // 2,
    )
    passed = (
        scene.mask is not None
        and scene.mask.shape == (region.height, region.width)
        and 100 + 40 + 6 <= region.left <= 100 + 40 + 10
        and 50 + 20 + 6 <= region.top <= 50 + 20 + 10
        and center_screen == expected_center
        and fallback.mask is None
        and fallback.region == loop10.win.Region(100, 50, 576, 324)
    )
    return passed, f"область {region}, центр {center_screen}, фолбэк {fallback.where()}", ""


def scenario_preset_mask() -> tuple[bool, str, str]:
    """Кандидаты: красное вне воды отбрасывается, поплавок в воде принимается."""
    # берег насыщеннее и ярче воды — так же, как в живой сцене (трава S≈210, V≈150)
    frame = scene_frame(ground_hsv=(40, 210, 150), red_patch=(15, 55))
    frame[80:86, 150:156] = hsv_bgr(0, 200, 210)  # поплавок в воде
    water = loop10.vision.detect_water(frame)
    if water is None:
        return False, "вода не найдена на синтетическом кадре", ""

    masked = loop10.vision.find_tracker_candidates(
        frame, loop10.vision.VERIFY_PRESET, water.limits
    )
    everything = loop10.vision.find_tracker_candidates(frame, loop10.vision.VERIFY_PRESET)
    bobber_inside = any(146 <= blob.cx <= 158 and 76 <= blob.cy <= 88 for blob in masked)
    armor_inside = any(blob.cx < 40 for blob in masked)
    passed = len(everything) == 2 and len(masked) == 1 and bobber_inside and not armor_inside
    return passed, f"без маски {len(everything)}, с маской {len(masked)}, в воде {bobber_inside}", ""


def scenario_params_bounds() -> tuple[bool, str, str]:
    """Параметр вне границ заменяется значением по умолчанию с сообщением."""
    defaults = loop10.DetectionParams.defaults()
    broken = loop10.DetectionParams(
        preset=loop10.vision.BobberPreset(
            hue_max=99, saturation_min=10, value_min=105, area_min=12, area_max=200
        ),
        burst_threshold=500,
        splash_rise=18,
        window_radius=15,
        confirm_frames=2,
    )
    output = io.StringIO()
    with redirect_stdout(output):
        fixed = broken.normalized()

    passed = (
        fixed.preset.hue_max == defaults.preset.hue_max
        and fixed.preset.saturation_min == defaults.preset.saturation_min
        and fixed.burst_threshold == defaults.burst_threshold
        and fixed.splash_rise == 18
        and fixed.preset.value_min == 105
        and "вне границ" in output.getvalue()
    )
    return passed, f"H<={fixed.preset.hue_max}, S>={fixed.preset.saturation_min}, порог {fixed.burst_threshold}", ""


def scenario_cycle_scene_params() -> tuple[bool, str, str]:
    """`run_cycle` передаёт в детекцию область, маску, пресет и параметры всплеска."""
    seen: dict[str, object] = {}

    class SpyTracker:
        def __init__(self, baseline, preset=None, mask=None):
            seen["preset"] = preset
            seen["mask"] = mask

        def update(self, frame):
            return None

        def seen(self):
            return True

        def point(self):
            return (10.0, 20.0)

        def points(self, limit=4):
            return [(10.0, 20.0)]

        def confirmed(self, frame, radius=12.0):
            return loop10.vision.Blob(cx=10.0, cy=20.0, area=30)

        def confirmed_at(self, frame, point, radius=12.0):
            return loop10.vision.Blob(cx=point[0], cy=point[1], area=30)

        def any_confirmed(self, frame, radius=12.0):
            return True

    class SpyMonitor:
        def __init__(self, point, threshold=130, rise=18, radius=15, confirm_frames=2):
            seen["monitor"] = (threshold, rise, radius, confirm_frames)
            self.point = point
            self.baseline = 5.0
            self.last = 7
            self.peak = 7

        def area(self, frame):
            return 7

        def observe(self, frame):
            return True

        def quiet(self, value=None):
            return True

    mask = np.ones((50, 100), bool)
    water = loop10.vision.WaterLimits(
        saturation_max=200.0, value_max=150.0, burst_value=110.0
    )
    scene = loop10.Scene(
        region=loop10.win.Region(10, 20, 100, 50),
        mask=mask,
        source="маска воды",
        water=water,
    )
    params = loop10.DetectionParams(
        burst_threshold=111, splash_rise=22, window_radius=33, confirm_frames=3
    )
    stats: list[loop10.CycleStats] = []
    with TemporaryDirectory() as tmp:
        trace = loop10.Trace(Path(tmp) / "run_test.txt")
        try:
            with patched(
                loop10.win,
                client_size=lambda window: (1920, 1080),
                roi_box=lambda width, height: (0, 0, 100, 50),
                client_origin=lambda window: (10, 20),
                stop_requested=lambda: False,
            ), patched(
                loop10.capture,
                save_fail_frame=lambda *args, **kwargs: None,
                Pacer=NoWaitPacer,
            ), patched(
                loop10.vision,
                find_tracker_candidates=lambda *args, **kwargs: [],
                bright_centroid=lambda *args, **kwargs: None,
                colour_at=lambda *args, **kwargs: (5.0, 210.0, 120.0),
            ), patched(
                loop10.input_layer,
                cast=lambda window, key: None,
            ), patched(
                loop10,
                BobberTracker=SpyTracker,
                BurstMonitor=SpyMonitor,
                wait_bobber_gone=lambda *args, **kwargs: True,
            ), redirect_stdout(io.StringIO()):
                marker = loop10.run_cycle(
                    1,
                    object(),
                    FakeCycleGrabber(),
                    trace,
                    click=lambda window, point, delay: 0.0,
                    scene=scene,
                    params=params,
                    stats=stats,
                )
        finally:
            trace.close()

    passed = (
        marker == loop10.OK
        and seen.get("preset") is params.preset
        and seen.get("mask") is water
        and seen.get("monitor") == (111, 22, 33, 3)
        and len(stats) == 1
        and stats[0].area == 30
        and stats[0].peak == 7
    )
    return passed, f"маркер {marker}, монитор {seen.get('monitor')}, наблюдений {len(stats)}", ""


def scenario_start_scene() -> tuple[bool, str, str]:
    """Старт сессии: найденная область печатается, фолбэк тоже, блок стартует."""
    cases = (
        ("маска воды", loop10.Scene(loop10.win.Region(50, 60, 200, 100), source="маска воды")),
        ("доли по умолчанию", loop10.Scene(loop10.win.Region(0, 0, 576, 324))),
    )
    details: list[str] = []
    passed = True
    for title, scene in cases:
        with TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            pending = [loop10.OK] * loop10.BLOCK_CYCLES
            verdicts: list[str] = []
            completed = {"count": 0, "stop_after_blocks": 1}

            def fake_run_cycle(cycle, window, grabber, trace, click=None, cast_key=None,
                               block=1, jump=False, scene=None, params=None):
                marker = pending.pop(0) if pending else loop10.OK
                trace.write(cycle, 0, "0", "-", "-", marker)
                return marker

            output = io.StringIO()
            with fake_environment(
                trace_dir,
                fake_run_cycle,
                verdicts,
                completed,
                detect_scene=lambda window, grabber, chosen=scene: chosen,
            ), redirect_stdout(output):
                code = loop10.main(["--countdown", "0"])
            text = output.getvalue()
        case_ok = (
            code == 0
            and verdicts == ["принято"]
            and "область поиска:" in text
            and scene.source in text
        )
        passed = passed and case_ok
        details.append(f"{title}: {'ок' if case_ok else 'нет'}")
    return passed, "; ".join(details), ""


def scenario_calibration_stops() -> tuple[bool, str, str]:
    """Калибровка: останавливается на 8-м `OK`, а без `OK` — на лимите циклов."""
    results: list[str] = []
    passed = True
    for title, always_ok, expected in (
        ("восемь OK", True, loop10.CALIBRATION_TARGET_OK),
        ("без OK", False, loop10.CALIBRATION_MAX_CYCLES),
    ):
        calls = {"count": 0}

        def fake_cycle(cycle, window, grabber, trace, scene=None, params=None, stats=None,
                       **kwargs):
            calls["count"] += 1
            marker = loop10.OK if always_ok else loop10.MISS
            trace.write(cycle, 0, "0", "-", "-", marker)
            if marker == loop10.OK and stats is not None:
                stats.append(
                    loop10.CycleStats(
                        hue=5.0, saturation=210.0, value=120.0, area=30, baseline=4.0, peak=30.0
                    )
                )
            return marker

        with TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            with patched(loop10.capture, TRACE_ROOT=trace_dir), patched(
                loop10, run_cycle=fake_cycle
            ), patched(loop10.win, stop_requested=lambda: False), redirect_stdout(io.StringIO()):
                params, markers, trace_path = loop10.run_calibration(
                    object(), FakeCycleGrabber(), loop10.Scene(loop10.win.Region(0, 0, 100, 50))
                )
            report = trace_path.with_name(trace_path.stem + "_report.md")
            report_text = report.read_text(encoding="utf-8") if report.exists() else ""

        case_ok = (
            calls["count"] == expected
            and len(markers) == expected
            and trace_path.name.startswith("calibration_")
            and bool(report_text)
            and ("источник параметров: калибровка" in report_text) == always_ok
            and (params is not None) == always_ok
        )
        passed = passed and case_ok
        results.append(f"{title}: циклов {calls['count']}, параметры {'есть' if params else 'нет'}")
    return passed, "; ".join(results), ""


def scenario_calibration_params() -> tuple[bool, str, str]:
    """Параметры по 8 наблюдениям: квантили, границы, окно и кадры подтверждения."""
    stats = [
        loop10.CycleStats(
            hue=5.0,
            saturation=200.0 + index,
            value=110.0 + index,
            area=28 + index,
            baseline=60.0,
            peak=80.0 + index,
        )
        for index in range(8)
    ]
    params = loop10.DetectionParams.from_stats(stats, water_burst_value=120.0)
    passed = (
        params.preset.hue_max == 9
        and params.preset.saturation_min == 181
        and params.preset.value_min == 96
        and params.preset.area_min == 25
        and params.preset.area_max == 42
        and params.burst_threshold == 120
        and params.splash_rise == 21
        and params.window_radius == 13
        and params.confirm_frames == 2
    )
    return passed, params.summary(), ""


def scenario_calibration_applied() -> tuple[bool, str, str]:
    """`main()` применяет параметры калибровки к блоку; без 8 `OK` — значения по умолчанию."""
    results: list[str] = []
    passed = True
    calibrated = loop10.DetectionParams(
        burst_threshold=111, splash_rise=22, window_radius=33, confirm_frames=3
    )
    for title, returned in (("калибровка", calibrated), ("без калибровки", None)):
        recorded: dict[str, object] = {}
        with TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)

            def fake_block(number, window, grabber, scene=None, params=None):
                recorded["params"] = params
                return trace_dir / "run_fake.txt", [loop10.OK] * loop10.BLOCK_CYCLES, False, None

            def fake_calibration(window, grabber, scene):
                return returned, [loop10.OK] * 8, trace_dir / "calibration_fake.txt"

            pending = [loop10.OK] * loop10.BLOCK_CYCLES
            verdicts: list[str] = []
            completed = {"count": 0, "stop_after_blocks": 1}
            output = io.StringIO()
            with fake_environment(
                trace_dir,
                lambda *args, **kwargs: loop10.OK,
                verdicts,
                completed,
                calibration=fake_calibration,
            ), patched(loop10, run_block=fake_block), redirect_stdout(output):
                code = loop10.main(["--countdown", "0"])
            text = output.getvalue()

        expected = calibrated if returned is not None else loop10.DetectionParams.defaults()
        case_ok = (
            code in (0, 1)
            and recorded.get("params") == expected
            and "параметры детекции:" in text
            and "отчёт калибровки:" in text
            and (("не набрала 8 OK" in text) == (returned is None))
        )
        passed = passed and case_ok
        results.append(f"{title}: {'ок' if case_ok else 'нет'}")
    return passed, "; ".join(results), ""


def scenario_calibration_failure() -> tuple[bool, str, str]:
    """Отказ ввода в калибровке: `WIN_FAIL` в маркерах, отчёт сохранён, без исключения."""
    def failing_cycle(cycle, window, grabber, trace, **kwargs):
        raise loop10.input_layer.InputBlocked("окно не в фокусе")

    with TemporaryDirectory() as tmp:
        trace_dir = Path(tmp)
        with patched(loop10.capture, TRACE_ROOT=trace_dir), patched(
            loop10, run_cycle=failing_cycle
        ), patched(loop10.win, stop_requested=lambda: False), redirect_stdout(io.StringIO()):
            params, markers, trace_path = loop10.run_calibration(
                object(), FakeCycleGrabber(), loop10.Scene(loop10.win.Region(0, 0, 100, 50))
            )
        report = trace_path.with_name(trace_path.stem + "_report.md")
        report_text = report.read_text(encoding="utf-8") if report.exists() else ""

    passed = (
        markers == [loop10.WIN_FAIL]
        and params is None
        and "значения по умолчанию" in report_text
        and "INPUT_BLOCKED" not in report_text  # причина в трассе, не в отчёте
    )
    return passed, f"маркеры {markers}, отчёт {'есть' if report_text else 'нет'}", ""


def scenario_safe_zone() -> tuple[bool, str, str]:
    """Обрезка области: интерфейс снизу и кромки экрана исключаются, точка внутри — нет."""
    full = loop10.clip_to_safe((0, 0, 1920, 1080), 1920, 1080)
    inside = loop10.clip_to_safe((700, 300, 500, 300), 1920, 1080)
    tiny = loop10.clip_to_safe((0, 0, 10, 10), 1920, 1080)
    passed = (
        full == (38, 21, 1843, 778)
        and inside == (700, 300, 500, 300)
        and tiny == (0, 0, 10, 10)  # пустую зону не возвращаем
    )
    return passed, f"полный кадр {full}, внутри {inside}, вырожденный {tiny}", ""


def scenario_candidate_choice() -> tuple[bool, str, str]:
    """Из нескольких кандидатов выбирается тот, у кого произошёл всплеск, а не самый голосистый."""
    clicked: dict[str, tuple[int, int]] = {}

    class TwoTracker:
        def __init__(self, baseline, preset=None, mask=None):
            return None

        def update(self, frame):
            return None

        def seen(self):
            return True

        def point(self):
            return (10.0, 20.0)

        def points(self, limit=4):
            return [(10.0, 20.0), (300.0, 200.0)]

        def confirmed(self, frame, radius=12.0):
            return loop10.vision.Blob(cx=10.0, cy=20.0, area=30)

        def confirmed_at(self, frame, point, radius=12.0):
            return loop10.vision.Blob(cx=point[0], cy=point[1], area=30)

        def any_confirmed(self, frame, radius=12.0):
            return True

    class PickyMonitor:
        """Всплеск только у второго кандидата: первый — ложный (крупный красный объект)."""

        def __init__(self, point, threshold=130, rise=18, radius=15, confirm_frames=2):
            self.point = point
            self.baseline = 5.0
            self.last = 7
            self.peak = 7

        def area(self, frame):
            return 7

        def observe(self, frame):
            return self.point[0] > 20

        def quiet(self, value=None):
            return True

    def fake_click(window, point, delay):
        clicked["point"] = point
        return 0.0

    scene = loop10.Scene(region=loop10.win.Region(100, 50, 500, 400), mask=None)
    with TemporaryDirectory() as tmp:
        trace = loop10.Trace(Path(tmp) / "run_test.txt")
        try:
            with patched(
                loop10.win,
                client_size=lambda window: (1920, 1080),
                roi_box=lambda width, height: (0, 0, 500, 400),
                client_origin=lambda window: (100, 50),
                stop_requested=lambda: False,
            ), patched(
                loop10.capture,
                save_fail_frame=lambda *args, **kwargs: None,
                Pacer=NoWaitPacer,
            ), patched(
                loop10.vision,
                find_tracker_candidates=lambda *args, **kwargs: [],
                bright_centroid=lambda *args, **kwargs: None,
                colour_at=lambda *args, **kwargs: (5.0, 210.0, 120.0),
            ), patched(
                loop10.input_layer,
                cast=lambda window, key: None,
            ), patched(
                loop10,
                BobberTracker=TwoTracker,
                BurstMonitor=PickyMonitor,
                wait_bobber_gone=lambda *args, **kwargs: True,
                ACQUIRE_DELAY=0.0,
            ), redirect_stdout(io.StringIO()):
                marker = loop10.run_cycle(
                    1,
                    object(),
                    FakeCycleGrabber(),
                    trace,
                    click=fake_click,
                    scene=scene,
                )
        finally:
            trace.close()

    expected = scene.region.to_screen(300.0, 200.0)
    passed = marker == loop10.OK and clicked.get("point") == expected
    return passed, f"маркер {marker}, клик {clicked.get('point')}, ожидалось {expected}", ""


def scenario_relock_late_bobber() -> tuple[bool, str, str]:
    """Поплавок появляется позже травы: окна пополняются и всплеск ловится на нём."""
    clicked: dict[str, tuple[int, int]] = {}
    state = {"calls": 0}

    class LateTracker:
        """Сначала виден только ложный кандидат, через три кадра — настоящий поплавок."""

        def __init__(self, baseline, preset=None, mask=None):
            return None

        def update(self, frame):
            return None

        def seen(self):
            return True

        def point(self):
            return (10.0, 20.0)

        def points(self, limit=6):
            state["calls"] += 1
            return [(10.0, 20.0)] if state["calls"] <= 3 else [(10.0, 20.0), (300.0, 200.0)]

        def confirmed(self, frame, radius=12.0):
            return loop10.vision.Blob(cx=10.0, cy=20.0, area=30)

        def confirmed_at(self, frame, point, radius=12.0):
            return loop10.vision.Blob(cx=point[0], cy=point[1], area=30)

        def any_confirmed(self, frame, radius=12.0):
            return True

    class PickyMonitor:
        def __init__(self, point, threshold=130, rise=18, radius=15, confirm_frames=2):
            self.point = point
            self.baseline = 5.0
            self.last = 7
            self.peak = 7

        def area(self, frame):
            return 7

        def observe(self, frame):
            return self.point[0] > 20

        def quiet(self, value=None):
            return True

    def fake_click(window, point, delay):
        clicked["point"] = point
        return 0.0

    scene = loop10.Scene(region=loop10.win.Region(100, 50, 500, 400), mask=None)
    with TemporaryDirectory() as tmp:
        trace = loop10.Trace(Path(tmp) / "run_test.txt")
        try:
            with patched(
                loop10.win,
                client_size=lambda window: (1920, 1080),
                roi_box=lambda width, height: (0, 0, 500, 400),
                client_origin=lambda window: (100, 50),
                stop_requested=lambda: False,
            ), patched(
                loop10.capture,
                save_fail_frame=lambda *args, **kwargs: None,
                Pacer=NoWaitPacer,
            ), patched(
                loop10.vision,
                find_tracker_candidates=lambda *args, **kwargs: [],
                bright_centroid=lambda *args, **kwargs: None,
                colour_at=lambda *args, **kwargs: (5.0, 210.0, 120.0),
            ), patched(
                loop10.input_layer,
                cast=lambda window, key: None,
            ), patched(
                loop10,
                BobberTracker=LateTracker,
                BurstMonitor=PickyMonitor,
                wait_bobber_gone=lambda *args, **kwargs: True,
                ACQUIRE_DELAY=0.0,
            ), redirect_stdout(io.StringIO()):
                marker = loop10.run_cycle(
                    1,
                    object(),
                    FakeCycleGrabber(),
                    trace,
                    click=fake_click,
                    scene=scene,
                )
        finally:
            trace.close()

    expected = scene.region.to_screen(300.0, 200.0)
    passed = marker == loop10.OK and clicked.get("point") == expected
    return (
        passed,
        f"маркер {marker}, клик {clicked.get('point')}, ожидалось {expected}, опросов {state['calls']}",
        "",
    )


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

    def short_cycle(cycle, window, grabber, trace, click=None, cast_key=None, block=1, jump=False, scene=None, params=None):
        calls["count"] += 1
        if calls["count"] > 4:
            raise loop10.RunInterrupted("F12 в тесте")
        trace.write(cycle, 0, "0", "-", "-", loop10.OK, jump="2500" if jump else "-")
        return loop10.OK

    verdicts: list[str] = []
    with TemporaryDirectory() as tmp:
        code, output = run_interrupted_session(Path(tmp), short_cycle, verdicts)
    passed = code == 0 and verdicts == ["прерван"] and calls["count"] == 5
    return passed, f"код {code}, вердикты {verdicts}, циклов {calls['count']}", output


def scenario_window_failure() -> tuple[bool, str, str]:
    """Отказ слоя окна в первом цикле: блок «не принято», сессия останавливается, код 1."""

    def failing_cycle(cycle, window, grabber, trace, click=None, cast_key=None, block=1, jump=False, scene=None, params=None):
        raise loop10.input_layer.InputBlocked("окно не в фокусе")

    verdicts: list[str] = []
    with TemporaryDirectory() as tmp:
        code, output = run_interrupted_session(Path(tmp), failing_cycle, verdicts)
    passed = code == 1 and verdicts == ["не принято"]
    return passed, f"код {code}, вердикты {verdicts}", output


def trace_columns(line: str) -> list[str]:
    """Колонки строки трассы без пробелов по краям."""
    return [part.strip() for part in line.split("|")]


def scenario_trace_format() -> tuple[bool, str, str]:
    """Строка трассы: 8 колонок, `jump` перед маркером, маркер и причина — последние."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "run_test.txt"
        trace = loop10.Trace(path)
        trace.write(5, 0, "-", "-", "-", loop10.LOST, reason="NO_BURST@11921")
        trace.write(7, 0, "8321", "8942", "11203", loop10.OK, jump="11400")
        # WIN_FAIL пишет `run_block` мимо `close`: формат держится на значении по умолчанию
        trace.write(9, 0, "-", "-", "-", loop10.WIN_FAIL, reason="INPUT_BLOCKED@15")
        trace.close()
        failed_line, ok_line, win_fail_line = path.read_text(encoding="utf-8").splitlines()

    failed_columns = trace_columns(failed_line)
    ok_columns = trace_columns(ok_line)
    win_fail_columns = trace_columns(win_fail_line)
    passed = (
        len(failed_columns) == 8
        and failed_columns[5] == "jump -"
        and failed_columns[6] == loop10.LOST
        and failed_columns[7] == "NO_BURST@11921"
        and len(ok_columns) == 8
        and ok_columns[5] == "jump 11400"
        and ok_columns[6] == loop10.OK
        and ok_columns[7] == "-"
        # подстрока, которую ищет PowerShell-однострочник из README
        and "| OK |" in ok_line
        and len(win_fail_columns) == 8
        and win_fail_columns[5] == "jump -"
        and win_fail_columns[6] == loop10.WIN_FAIL
        and win_fail_columns[7] == "INPUT_BLOCKED@15"
    )
    return passed, f"колонок {len(ok_columns)}: {ok_line}", ""


def scenario_sleep_checked() -> tuple[bool, str, str]:
    """`sleep_checked` поднимает `RunInterrupted` не позже одного шага после F12."""
    calls = {"count": 0}

    def stop_after_one() -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    started = time.monotonic()
    with patched(loop10.win, stop_requested=stop_after_one):
        try:
            loop10.sleep_checked(1.0)
        except loop10.RunInterrupted:
            elapsed = time.monotonic() - started
            passed = elapsed <= loop10.JUMP_SLEEP_STEP + 0.05
            return passed, f"опросов {calls['count']}, остановка за {elapsed:.3f} с", ""
        except Exception as exc:  # непредвиденное исключение — провал сценария
            return False, f"неожиданное исключение: {exc!r}", ""
    return False, "исключение не поднято", ""


class FakeCycleGrabber:
    """Заглушка захвата для `run_cycle`: кадр есть, детекция подменена."""

    last_state = "fake"

    def grab_roi(self, window):
        return object()

    def grab_region(self, region):
        return object()


class NoWaitPacer:
    """Заглушка темпа захвата: цикл в тесте не ждёт 20 Гц."""

    def wait(self) -> None:
        return None


@contextmanager
def fake_cycle_environment(
    jump_input,
    pauses: tuple[float, float],
    stop_requested=lambda: False,
):
    """Окружение `run_cycle` без клиента: окно, захват, детекция, ввод и паузы прыжка.

    Поплавок не находится (`find_tracker_candidates` пуст), а `ACQUIRE_TIMEOUT = 0`
    закрывает цикл отказом `MISS`/`NO_BOBBER` на первом кадре — это кратчайший путь
    до закрывателя цикла, где выполняется прыжок. `stop_requested` подменяет F12.
    """
    with patched(
        loop10.win,
        client_size=lambda window: (1920, 1080),
        roi_box=lambda width, height: (0, 0, 100, 100),
        client_origin=lambda window: (0, 0),
        stop_requested=stop_requested,
    ), patched(
        loop10.capture,
        save_fail_frame=lambda *args, **kwargs: None,
        Pacer=NoWaitPacer,
    ), patched(
        loop10.vision,
        find_tracker_candidates=lambda *args, **kwargs: [],
    ), patched(
        loop10.input_layer,
        cast=lambda window, key: None,
        jump=jump_input,
    ), patched(
        loop10,
        ACQUIRE_TIMEOUT=0.0,
        JUMP_PAUSE_BEFORE=pauses,
        JUMP_PAUSE_AFTER=pauses,
    ):
        yield


def run_fake_cycle(jump_enabled: bool, jump_input, pauses, output=None):
    """Один `run_cycle` в подменённом окружении: `(маркер, колонки строки трассы)`."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "run_test.txt"
        trace = loop10.Trace(path)
        try:
            with fake_cycle_environment(jump_input, pauses), redirect_stdout(
                output or io.StringIO()
            ):
                marker = loop10.run_cycle(3, object(), FakeCycleGrabber(), trace, jump=jump_enabled)
        finally:
            trace.close()
        line = path.read_text(encoding="utf-8").splitlines()[0]
    return marker, trace_columns(line)


def scenario_jump_sent() -> tuple[bool, str, str]:
    """Прыжок после отказа: `reason@<мс>` снят до пауз, колонка `jump` — момент нажатия."""
    pauses = (0.2, 0.2)
    calls: list[float] = []
    marker, columns = run_fake_cycle(True, lambda window: calls.append(time.monotonic()), pauses)

    reason = columns[7]
    moment = int(reason.split("@")[1]) if "@" in reason else -1
    passed = (
        marker == loop10.MISS
        and columns[6] == loop10.MISS
        and 0 <= moment < int(pauses[0] * 1000)
        and len(calls) == 1
        and columns[5] not in ("jump -", "")
    )
    return passed, f"маркер {columns[6]}, причина {reason}, колонка jump {columns[5]}", ""


def scenario_jump_failures() -> tuple[bool, str, str]:
    """Отказ `jump` (`InputBlocked` и `OSError`): предупреждение, маркер не меняется."""
    pauses = (0.3, 0.3)

    def blocked(window):
        raise loop10.input_layer.InputBlocked("окно не в фокусе")

    def send_failed(window):
        raise OSError(5, "SendInput: прыжок не отправлен")

    cases = (
        ("InputBlocked", blocked, "прыжок пропущен"),
        ("OSError", send_failed, "прыжок не отправлен"),
    )
    details: list[str] = []
    passed = True
    for title, jump_input, expected_message in cases:
        output = io.StringIO()
        started = time.monotonic()
        marker, columns = run_fake_cycle(True, jump_input, pauses, output)
        elapsed = time.monotonic() - started
        case_ok = (
            marker == loop10.MISS
            and columns[6] == loop10.MISS
            and columns[5] == "jump -"
            and expected_message in output.getvalue()
            # вторая пауза пропущена: прошла только одна из двух
            and elapsed < pauses[0] + 0.1
        )
        passed = passed and case_ok
        details.append(f"{title}: маркер {columns[6]}, jump {columns[5]}, цикл {elapsed:.2f} с")
    return passed, "; ".join(details), ""


def run_interrupted_cycle(jump_input, stop_requested, pauses, output=None):
    """`run_cycle` с F12 внутри: `(исключение, вызовы прыжка, текст трассы)`."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "run_test.txt"
        trace = loop10.Trace(path)
        error: BaseException | None = None
        try:
            with fake_cycle_environment(jump_input, pauses, stop_requested), redirect_stdout(
                output or io.StringIO()
            ):
                loop10.run_cycle(3, object(), FakeCycleGrabber(), trace, jump=True)
        except BaseException as exc:  # ожидается RunInterrupted; сценарий проверит тип
            error = exc
        finally:
            trace.close()
        text = path.read_text(encoding="utf-8")
    return error, text


def scenario_jump_interrupted() -> tuple[bool, str, str]:
    """F12 в паузе перед прыжком: клавиша не отправляется, строка цикла не пишется."""
    calls: list[str] = []
    stop = {"count": 0}

    def stop_soon() -> bool:
        # первый опрос (фаза ожидания) — False, второй (пауза перед прыжком) — True
        stop["count"] += 1
        return stop["count"] > 1

    error, text = run_interrupted_cycle(lambda window: calls.append("jump"), stop_soon, (0.3, 0.3))
    passed = isinstance(error, loop10.RunInterrupted) and calls == [] and text == ""
    return passed, f"исключение {type(error).__name__}, прыжков {len(calls)}, строк трассы {len(text.splitlines())}", ""


def scenario_jump_slot() -> tuple[bool, str, str]:
    """Один прыжок за блок: слот 3–10, выбор печатается до первого цикла."""
    with TemporaryDirectory() as tmp:
        trace_dir = Path(tmp)
        recorded: list[tuple[int, bool]] = []
        pending = [loop10.OK] * loop10.BLOCK_CYCLES
        verdicts: list[str] = []
        completed = {"count": 0, "stop_after_blocks": 1}

        def spy_cycle(cycle, window, grabber, trace, click=None, cast_key=None, block=1, jump=False, scene=None, params=None):
            recorded.append((cycle, jump))
            marker = pending.pop(0) if pending else loop10.OK
            trace.write(cycle, 0, "0", "-", "-", marker, jump="2500" if jump else "-")
            return marker

        output = io.StringIO()
        with fake_environment(trace_dir, spy_cycle, verdicts, completed), redirect_stdout(output):
            code = loop10.main(["--countdown", "0"])
        lines = output.getvalue().splitlines()
        traces = sorted(path.name for path in trace_dir.glob("run_*.txt"))
        trace_lines = (trace_dir / traces[0]).read_text(encoding="utf-8").splitlines()

    jumped = [cycle for cycle, flag in recorded if flag]
    choice = next((index for index, line in enumerate(lines) if "прыжок в цикле" in line), -1)
    first_cycle = next((index for index, line in enumerate(lines) if "цикл 1/" in line), -1)
    announced = -1
    if 0 <= choice:
        tail = lines[choice].split("цикле")[-1].strip()
        head = tail.split()[0] if tail else ""
        announced = int(head) if head.isdigit() else -1
    jump_columns = [trace_columns(line)[5] for line in trace_lines]
    passed = (
        code == 0
        and verdicts == ["принято"]
        and len(recorded) == loop10.BLOCK_CYCLES
        and len(jumped) == 1
        and loop10.JUMP_CYCLES[0] <= jumped[0] <= loop10.JUMP_CYCLES[1]
        and announced == jumped[0]
        and 0 <= choice < first_cycle
        and jump_columns.count("jump 2500") == 1
        and jump_columns.count("jump -") == loop10.BLOCK_CYCLES - 1
    )
    return passed, f"слот {jumped}, объявлен {announced}, выбор в строке {choice}", ""


def scenario_jump_after_markers() -> tuple[bool, str, str]:
    """Прыжок выполняется и после `OK`, и после отказа: маркер цикла не меняется."""
    cases = (
        ("после OK", [loop10.OK] * loop10.BLOCK_CYCLES, loop10.OK),
        ("после LOST", [loop10.OK, loop10.OK, loop10.LOST] + [loop10.OK] * 7, loop10.LOST),
    )
    details: list[str] = []
    passed = True
    for title, markers, expected in cases:
        with TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            with patched(loop10, JUMP_CYCLES=(3, 3)):
                code, verdicts, traces, _ = run_session(
                    markers, stop_after_blocks=1, trace_dir=trace_dir
                )
            lines = (trace_dir / traces[0]).read_text(encoding="utf-8").splitlines()

        target = trace_columns(lines[2])
        others = [trace_columns(line)[5] for index, line in enumerate(lines) if index != 2]
        case_ok = (
            code == 0
            and verdicts == ["принято"]
            and target[5] == "jump 2500"
            and target[6] == expected
            and all(column == "jump -" for column in others)
        )
        passed = passed and case_ok
        details.append(f"{title}: {target[5]} | {target[6]}")
    return passed, "; ".join(details), ""


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
        ("маска воды", scenario_water_mask),
        ("сцена по воде", scenario_water_scene),
        ("зона без интерфейса", scenario_safe_zone),
        ("кандидаты в воде", scenario_preset_mask),
        ("границы параметров", scenario_params_bounds),
        ("параметры в цикле", scenario_cycle_scene_params),
        ("выбор кандидата", scenario_candidate_choice),
        ("повторный захват", scenario_relock_late_bobber),
        ("старт: область", scenario_start_scene),
        ("калибровка: лимит", scenario_calibration_stops),
        ("калибровка: параметры", scenario_calibration_params),
        ("калибровка: отказ ввода", scenario_calibration_failure),
        ("калибровка: применение", scenario_calibration_applied),
        ("пакет каста", scenario_cast_packet),
        ("клавиша прыжка", scenario_jump_key),
        ("формат трассы", scenario_trace_format),
        ("пауза с опросом F12", scenario_sleep_checked),
        ("прыжок после отказа", scenario_jump_sent),
        ("отказ прыжка", scenario_jump_failures),
        ("F12 в паузе прыжка", scenario_jump_interrupted),
        ("слот прыжка", scenario_jump_slot),
        ("прыжок после маркеров", scenario_jump_after_markers),
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
