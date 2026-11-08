# Lab 1: Vector Operations and Activations — тестовая сдача

## Описание

Тестовая работа для проверки оформления PR и автоматического CI. Программа вычисляет add, SAXPY и ReLU одним CUDA-ядром. Режимы `coalesced` и `noncoalesced` отличаются порядком обращения потоков к элементам.

## Окружение

Workflow использует GitHub-hosted Ubuntu 24.04 и Python 3.12. Для сборки примера нужны CMake 3.20+, CUDA Toolkit с nvcc и поддержка C++17; для выполнения — GPU NVIDIA.

## Компиляция

Из `works/comiam/lab1/`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target submission
```

## Запуск

```bash
./build/submission --help
./build/submission --input data/input.txt --output result.txt
./build/submission --input data/input.txt --output result-other.txt --mode noncoalesced
```

Вход — текстовый файл: сначала `N alpha`, затем N пар `x y`. В этом примере N допускается от 0 до 1000000. На выходе N строк с тремя значениями: `x + y`, `alpha * x + y`, `max(x, 0)`.

Режим по умолчанию — `coalesced`. Каждый блок содержит 32 потока; в `noncoalesced` индексы переставляются внутри блока. Выход за границы проверяется в ядре.

## Проверка корректности

В [data/expected.txt](data/expected.txt) приведены ожидаемые значения для восьми пар из [data/input.txt](data/input.txt). Все входы и результаты — небольшие целые числа, точно представимые в FP32.

После запуска сравните результаты:

```bash
diff -u data/expected.txt result.txt
diff -u result.txt result-other.txt
```

## Результаты

Ожидаемые значения первых трёх строк:

| x | y | add | SAXPY при alpha = 2 | ReLU |
| --- | --- | --- | --- | --- |
| -2 | 5 | 3 | 1 | 0 |
| -1 | 2 | 1 | 0 | 0 |
| 0 | -1 | -1 | -1 | 0 |

При проверке PR результат оформления появится в статусе `submission/static`. Подробности доступны в отчёте workflow.

## Выводы

Эта ветка даёт небольшой воспроизводимый пример сдачи: исходник CUDA, сборку CMake, вход, ожидаемый результат и отчёт. На ней можно проверить успешный CI, затем изменить оформление и проверить сообщения об ошибках.

## Воспроизводимость

### Данные

Вход и ожидаемые значения находятся в каталоге `data/`. Используются фиксированные пары и коэффициент SAXPY 2.

### Полный запуск

Из каталога лабораторной:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target submission
./build/submission --input data/input.txt --output result.txt
./build/submission --input data/input.txt --output result-other.txt --mode noncoalesced
diff -u data/expected.txt result.txt
diff -u result.txt result-other.txt
```
