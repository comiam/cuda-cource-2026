# Lab 1 — контрольная сдача для проверки CI

## Описание

Рабочая тестовая сдача для проверки положительного результата GPU CI. CUDA-ядро вычисляет сложение, SAXPY и ReLU в обоих режимах. Это проверка инфраструктуры, а не отчёт о производительности.

## Окружение

Linux, CUDA с поддержкой C++17 и CMake 3.20 или новее. Для запуска нужен GPU NVIDIA.

## Компиляция

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target submission
```

## Запуск

```bash
./build/submission --help
./build/submission --input example.txt --output coalesced.txt
./build/submission --input example.txt --output noncoalesced.txt --mode noncoalesced
```

Без `--mode` используется `coalesced`. Для `N=0` программа создаёт пустой выходной файл и возвращает `0` без запуска ядра.

## Проверка корректности

Для открытого примера из задания ожидаем одинаковые значения в обоих режимах:

```text
-1 0.5 0
-4 -4.5 1
4.5 2.5 4
```

Операции выполняются в FP32, для SAXPY используется FMA. Индекс выходной тройки совпадает с индексом входной пары.

## Результаты

Контроль этой сдачи — статическая проверка и GPU-вердикт в тестовом PR. Сравнение производительности не входит в этот прогон.

## Выводы

В коалесцированном режиме соседние потоки читают соседние элементы. В некоалесцированном — потоки варпа обращаются к элементам с шагом восемь внутри блока. Оба режима должны записать одинаковый результат.

## Воспроизводимость

### Данные

Открытый пример Lab 1: три пары чисел с положительными и отрицательными компонентами.

### Полный запуск

```bash
printf '3 0.5\n-3 2\n1 -5\n4 0.5\n' > example.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target submission
./build/submission --input example.txt --output coalesced.txt
./build/submission --input example.txt --output noncoalesced.txt --mode noncoalesced
cmp coalesced.txt noncoalesced.txt
cat coalesced.txt
```
