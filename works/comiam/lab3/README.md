# Lab 3: Sobel Operator — Оператор Собеля

## Описание

Контрольная сдача для проверки CI. `global` читает окрестность `3×3` из глобальной памяти. В `shared` блок из `16×16` потоков загружает участок `18×18`: тайл и halo в один пиксель, включая углы. Все потоки блока доходят до барьера, в том числе в неполном тайле. На внешней рамке оба режима записывают ноль.

## Окружение

Для сборки нужны CMake 3.18+, CUDA Toolkit с `nvcc` и компилятор C++17. Для запуска нужен CUDA-совместимый GPU. PNG читает и записывает [LodePNG](https://github.com/lvandeve/lodepng/tree/7b2893e35891c0c85bee5595caafaf3c2044b409); исходники и лицензия библиотеки лежат в `vendor/`, загрузка при сборке не нужна. Для открытого примера ниже используются Python и Pillow.

## Компиляция

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target submission
```

## Запуск

```bash
./build/submission --help
./build/submission --input sample.png --output global.png --mode global
./build/submission --input sample.png --output shared.png --mode shared
```

По умолчанию выбирается `global`. Выход — 8-битный одноканальный PNG того же размера. Коды: `0` — успех, `2` — неверные аргументы или вход, `1` — ошибка CUDA или записи.

## Проверка корректности

У открытого градиента `3×3` в центре `Gx=24`, `Gy=32`; модуль градиента равен `40`. Ожидаемый результат — `40` в центре и восемь нулей на рамке. Ниже проверяются сами пиксели, а не байты сжатого PNG.

## Результаты

Приёмочные запуски проверяют сборку, наличие результата и его корректность. Значения из отчётов CI относятся к соответствующему коммиту. Замеры производительности здесь не заявлены.

## Профилирование и оптимизация

Тайл переиспользует значения соседних пикселей, но требует загрузки halo и барьера. Чтобы сравнить варианты, отдельно замерьте интервал выполнения ядра после прогрева и повторите замеры на нескольких размерах изображения. В этом черновике проверяется CI, измерения и сравнение до/после не проводились.

## Выводы

Способ чтения пикселей различается, а маски, округление, насыщение и обработка внешней рамки одинаковы. Какой вариант быстрее, по проверке корректности судить нельзя.

## Воспроизводимость

### Данные

Для открытого примера использован градиент со строками `20 23 26`, `24 27 30` и `28 31 34`.

### Полный запуск

```bash
python3 - <<'PY'
from PIL import Image
sample = Image.frombytes('L', (3, 3), bytes([20, 23, 26, 24, 27, 30, 28, 31, 34]))
sample.save('sample.png')
PY
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target submission
./build/submission --input sample.png --output global.png --mode global
./build/submission --input sample.png --output shared.png --mode shared
./build/submission --input sample.png --output default.png
python3 - <<'PY'
from PIL import Image
for path in ('global.png', 'shared.png', 'default.png'):
    with Image.open(path) as result:
        assert (result.format, result.mode, result.size) == ('PNG', 'L', (3, 3))
        assert result.tobytes() == bytes([0, 0, 0, 0, 40, 0, 0, 0, 0])
PY
```
