# Автоматизация рыбной ловли для World of Warcraft

> Конспект внешней статьи: метаданные, содержание, ключевые технические детали и критические замечания.
> Источник: <https://habr.com/ru/articles/335580/>
> Дата конспекта: 2026-11-04

## Метаданные

| Параметр | Значение |
|---|---|
| Заголовок | Автоматизация рыбной ловли для World of Warcraft |
| Автор | Давид, ник `kio_tk` (<https://habr.com/ru/users/kio_tk/>) |
| Дата публикации | 14 августа 2017, 09:31 |
| Хабы / теги | Python; `world of warcraft`, `боты`, `python`, `opencv` |
| Блог | «Из песочницы» |
| Статистика | 66 голосов (↑63 / ↓3), +60, 148 закладок, 89 комментариев, охват 98K, ~5 мин чтения |
| Репозиторий | <https://github.com/kioltk/wow-fishipy> |

## Краткое содержание

Автор описывает самодельного бота для рыбалки в World of Warcraft, написанного на Python
как первый опыт работы с языком. Принципиальный выбор архитектуры — **не читать и не менять
память процесса игры** (палится античитом и легко детектится), а работать «снаружи»:
перехватывать управление мышью, делать скриншоты игрового экрана и находить поплавок через
OpenCV template matching. По координатам поплавка бот двигает курсор (`autopy`), кликает —
забрасывает удочку. Затем слушает системный звук всплеска (`pyaudio`; на macOS — через
перенаправление вывода во вход средствами Soundflower) и по превышению порога громкости
делает правый клик — подсечку. За неделю ночного теста бот выполнил ~7000 забросов и поймал
~5000 рыб (КПД ~70%). Автор прямо предупреждает: автоматизация нарушает пользовательское
соглашение и грозит баном.

Пайплайн:

```
скриншот центра экрана → поиск поплавка (OpenCV template matching)
   → smooth_move курсора (autopy) → клик (заброс)
   → прослушивание звука (pyaudio) → порог громкости → правый клик (подсечка)
```

## Структура статьи

1. **Введение / мотивация** — рыбалка нудная; не хочется трогать память процесса; нет ботов под macOS.
2. **Выбор инструментов** — OpenCV для поиска поплавка, `pyscreenshot` для скриншотов,
   `autopy` для мыши, `pyaudio` для звука.
3. **Захват экрана** — `check_screen_size()` / `make_screenshot()`: определение размера экрана,
   «коэффициент 2» на Retina-макбуке, обрезка центральной области ~9 % экрана.
4. **Поиск поплавка** — 8 шаблонов изображений поплавка, цикл по шаблонам с `cv2.matchTemplate`
   и порогом 0.6.
5. **Движение мыши** — `autopy.mouse.smooth_move` к найденным координатам.
6. **Детект поклёвки по звуку** — OpenCV слишком медленный (~0.5 с на кадр), поэтому анализ
   громкости через `audioop.avg`, порог 1200, ограничение прослушивания 20 секундами.
7. **Подсечка** — правый клик `autopy.mouse.click(autopy.mouse.RIGHT_BUTTON)` при успешном детекте.
8. **Результаты и дисклеймер** — статистика ~7000 бросков / ~5000 рыб, ссылка на GitHub,
   предупреждение о бане за нарушение ToS.

## Ключевые тезисы и технические детали

- Принципиальный выбор: **эмуляция пользователя** (скриншоты + мышь + звук) вместо чтения/записи
  памяти игры — ниже риск детекта античитом.
- Особенность Retina/macOS: `ImageGrab.grab(bbox=...)` возвращает изображение с коэффициентом 2
  по координатам, поэтому все координаты делят на 2.
- Template matching с пониженным порогом 0.8 → 0.6: поплавок качается и меняет цвет от освещения;
  ниже опускать нельзя — рыба начинает детектиться как поплавок.
- Звук надёжнее картинки: OpenCV тратит ~0.5 с на кадр, а поплавок «прыгает» быстрее — момент
  поклёвки ловится по громкости.
- На macOS нельзя слушать output напрямую — использован Soundflower для маршрутизации вывода
  в виртуальный вход.
- Слушание ограничено 20 секундами (максимальная длительность заброса).
- Итоговая точность ~70 %: промахи из-за неудачного освещения, поворота поплавка и пропущенного звука.

### Код (дословно; Python 2 — `print` без скобок)

Захват экрана:

```python
import pyscreenshot as ImageGrab

screen_size = None
screen_start_point = None
screen_end_point = None

# Сперва мы проверяем размер экрана и берём начальную и конечную точку для будущих скриншотов
def check_screen_size():
	print "Checking screen size"
	img = ImageGrab.grab()
	# img.save('temp.png')
	global screen_size
	global screen_start_point
	global screen_end_point

	# я так и не смог найти упоминания о коэффициенте в методе grab с параметром bbox, но на моем макбуке коэффициент составляет 2. то есть при создании скриншота с координатами x1=100, y1=100, x2=200, y2=200), размер картинки будет 200х200 (sic!), поэтому делим на 2
	coefficient = 2
	screen_size = (img.size[0] / coefficient, img.size[1] / coefficient)

	# берем примерно девятую часть экрана примерно посередине.
	screen_start_point = (screen_size[0] * 0.35, screen_size[1] * 0.35)
	screen_end_point = (screen_size[0] * 0.65, screen_size[1] * 0.65)
	print ("Screen size is " + str(screen_size))

def make_screenshot():
	print 'Capturing screen'
	screenshot = ImageGrab.grab(bbox=(screen_start_point[0], screen_start_point[1], screen_end_point[0], screen_end_point[1]))
	# сохраняем скриншот, чтобы потом скормить его в OpenCV
	screenshot_name = 'var/fishing_session_' + str(int(time.time())) + '.png'
	screenshot.save(screenshot_name)
	return screenshot_name

def main():
	check_screensize()
	make_screenshot()
```

Поиск поплавка:

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

def find_float(screenshot_name):
	print 'Looking for a float'
	for x in range(0, 7):
		# загружаем шаблон
		template = cv2.imread('var/fishing_float_' + str(x) + '.png', 0)
		# загружаем скриншот и изменяем его на чернобелый
		src_rgb = cv2.imread(screenshot_name)
		src_gray = cv2.cvtColor(src_rgb, cv2.COLOR_BGR2GRAY)
		# берем ширину и высоту шаблона
		w, h = template.shape[::-1]
		# магия OpenCV, которая и находит наш темплейт на картинке
		res = cv2.matchTemplate(src_gray, template, cv2.TM_CCOEFF_NORMED)
		# понижаем порог соответствия нашего шаблона с 0.8 до 0.6, ибо поплавок шатается и освещение в локациях иногда изменяет его цвета, но не советую ставить ниже, а то и рыба будет похожа на поплавок
		threshold = 0.6
		# numpy фильтрует наши результаты по порогу
		loc = np.where( res >= threshold)
		# выводим результаты на картинку
		for pt in zip(*loc[::-1]):
		    cv2.rectangle(src_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
		# и если результаты всё же есть, то возвращаем координаты и сохраняем картинку
		if loc[0].any():
			print 'Found float at ' + str(x)
			cv2.imwrite('var/fishing_session_' + str(int(time.time())) + '_success.png', src_rgb)
			return (loc[1][0] + w / 2) / 2, (loc[0][0] + h / 2) / 2 # опять мы ведь помним, что макбук играется с разрешениями? поэтому снова приходится делить на 2

def main():
	check_screensize()
	img_name = make_screenshot()
	find_float(img_name)
```

Движение мыши:

```python
import autopy

def move_mouse(place):
	x,y = place[0], place[1]
	print("Moving cursor to " + str(place))
	autopy.mouse.smooth_move(int(screen_start_point[0]) + x , int(screen_start_point[1]) + y)

def main():
	check_screensize()
	img_name = make_screenshot()
	cords = find_float(img_name)
	move_mouse(cords)
```

Детект поклёвки по звуку:

```python
import pyaudio
import wave
import audioop
from collections import deque
import time
import math

def listen():
	print 'Listening for loud sounds...'
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 18000 # битрейт звука, который мы хотим слушать
	THRESHOLD = 1200  # порог интенсивности звука, если интенсивность ниже, значит звук по нашим меркам слишком тихий
	SILENCE_LIMIT = 1  # длительность тишины, если мы не слышим ничего это время, то начинаем слушать заново

	# открываем стрим
	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
	                channels=CHANNELS,
	                rate=RATE,
	                # output=True, # на мак ос нет возможности слушать output, поэтому мне пришлось прибегнуть к использованию Soundflower, который умеет перенаправлять канал output в input, таким образом мы перехватываем звук игры будто это микрофон
	                input=True,
	                frames_per_buffer=CHUNK)
	cur_data = ''
	rel = RATE/CHUNK
	slid_win = deque(maxlen=SILENCE_LIMIT * rel)

	# начинаем слушать и по истечении 20 секунд (столько максимум длится каждый заброс поплавка), отменяем нашу слушалку.
	success = False
	listening_start_time = time.time()
	while True:
		try:
			cur_data = stream.read(CHUNK)
			slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
			if(sum([x > THRESHOLD for x in slid_win]) > 0):
				print 'I heart something!'
				success = True
				break
			if time.time() - listening_start_time > 20:
				print 'I don\'t hear anything during 20 seconds!'
				break
		except IOError:
			break
	# обязательно закрываем стрим
	stream.close()
	p.terminate()
	return success

def main():
	check_screensize()
	img_name = make_screenshot()
	cords = find_float(img_name)
	move_mouse(cords)
	listen()
```

Подсечка:

```python
def snatch():
	print('Snatching!')
	autopy.mouse.click(autopy.mouse.RIGHT_BUTTON)

def main():
	check_screensize()
	img_name = make_screenshot()
	cords = find_float(img_name)
	move_mouse(cords)
	if listen():
		snatch()
```

## Ссылки из статьи

| Что | URL |
|---|---|
| Полный код бота | <https://github.com/kioltk/wow-fishipy> |
| Гайд OpenCV template matching | <https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html> |
| Источник кода прослушивания | <https://gist.github.com/spadgenske/9162330> |
| Soundflower | <https://github.com/RogueAmoeba/Soundflower-Original> |

## Оценка и замечания

### Актуальность (статья 2017 года)

| Компонент | Статус | Современная альтернатива |
|---|---|---|
| Python 2 (`print` как оператор) | Не запускается на Python 3 без правок | Python 3.11+ |
| `autopy` | Заброшен (~2017–2018), не собирается на современных Python/macOS | `pyautogui`, `pynput` |
| `pyscreenshot` | Морально устарел, медленный | `mss`, `dxcam` (Windows), `Quartz` (macOS) |
| `audioop` | **Удалён из Python 3.13** (PEP 594) | `numpy`/`audioop-lts`, `sounddevice` |
| `pyaudio` | Жив, подход «порог громкости» крайне хрупок | `sounddevice` + RMS/spectral flux |
| Retina-коэффициент 2 | Костыль под конкретный MacBook; на Windows/другом DPI не работает | `mss` с явным DPI-awareness |
| `cv2.matchTemplate` | Концепция не устарела, но фиксированные PNG-шаблоны не переживут смену графики WoW | YOLO/детекция по признакам, либо шаблоны, обновляемые из скриншотов |
| Soundflower | Не работает на современных macOS | BlackHole, Loopback |

### Дефекты листингов

- Нет `import time` в первом фрагменте; `check_screensize()` vs `check_screen_size()` — опечатка в вызовах.
- `main()` переопределяется в каждом листинге — фрагменты не складываются в один файл.
- `find_float()` может вернуть `None` → падение в `move_mouse()` (нет обработки ошибок).
- Нет задержек и рандомизации — поведение палится по таймингам.
- Метрики (~7000/5000) — самоотчёт без методологии, воспроизвести нельзя.
