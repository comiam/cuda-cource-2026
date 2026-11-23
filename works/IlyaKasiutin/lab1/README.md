# Lab 1: Vector Operations and Activations - Векторные операции и активации

## Описание

Реализовано 3 операции: `x + y`, SAXPY `alpha * x + y` и ReLU `max(0, x)`.

Каждая из операций представлена в 2-х различных вариантах:  

- В режиме `coalesced` соседние потоки warp читают соседние элементы. 
- В `noncoalesced` те же самые операции идут с шагом 32. Т.е. `thread_idx = 0` обрабатывает индексы массива `[0, 32, 64, ...]`, `thread_idx = 1` `[1, 33, 65, ...]` и т.д.

Эталон - те же выражения на CPU в FP32, с тем же исходным `y`. Отдельная библиотека не используется.

## Окружение

- GPU: NVIDIA L40S, 46068 MiB
- Драйвер: 610.57.04
- CUDA Toolkit: 13.5.37 (`nvcc` в `/usr/local/cuda/bin`)
- ОС: Debian GNU/Linux 12 (bookworm), ядро 6.1.0-52-amd64
- Компилятор: g++ 12.2.0
- CMake: 3.25.1
- Флаги: `-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89`

## Компиляция

Из каталога лабораторной, с `nvcc` в `PATH`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target submission
```

## Запуск

Оба режима считают все три операции. По умолчанию `coalesced`, размер блока 256.
Вход: `N alpha` и `N` пар `x y`. Выход `--output` без заголовка: `x+y`, `alpha*x+y`, `max(0, x)` через пробел, 9 значащих цифр.
При `N = 0` ядро не запускается, в `--output` записывается пустой файл.
`--csv` дописывает медиану замеров. `--synthetic N` и `--alpha` строят вход на хосте (по умолчанию `alpha = 0.5`).

```bash
./build/submission --help
./build/submission --input data/lab1.txt --output coalesced.txt
./build/submission --input data/lab1.txt --output noncoalesced.txt --mode noncoalesced
```

## Реализация

Индекс потока `tid = blockIdx.x * blockDim.x + threadIdx.x`. Везде есть проверки вида `tid < n`.
В коалесцированном режиме элемент совпадает с `tid`: варп читает `0, 1, …, 31`.
В некоалесцированном массив делится на плитки по 1024. Внутри плитки варп читает `0, 32, 64, …, 992`.
Реализация некоалесцированного режима:

```
__device__ int nonCoalescedIndex(int tid, int n) {
    constexpr int warpSize = 32;
    constexpr int tile = warpSize * warpSize;
    const int full = n - n % tile;
    if (tid >= full) {
        return tid;
    }
    const int local = tid % tile;
    const int lane = local % warpSize;
    const int warpId = local / warpSize;
    return tid - local + lane * warpSize + warpId;
}
```

На GPU выделяются пять буферов: `x`, `y` и три результата. Буферы выделяются следующим образом:

```
cudaMalloc(&d_x, bytes);
cudaMalloc(&d_y, bytes);
cudaMalloc(&d_add, bytes);
cudaMalloc(&d_saxpy, bytes);
cudaMalloc(&d_relu, bytes);
```

Далее, `x` и `y` копируются на GPU с помощью стандартных функций:

```
cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_y, h_h.data(), bytes, cudaMemcpyHostToDevice);
```

Все операции с CUDA обернуты в фукнцию-валидатор check:

```
void checkCuda(cudaError_t status) {
    if (status == cudaSuccess) {
        return;
    }
    std::cerr << "ошибка CUDA: " << cudaGetErrorString(status) << '\n';
    std::exit(1);
}
```

## Проверка корректности

Допуск - точное совпадение в FP32: максимальная абсолютная ошибка 0. Числовой `+0` равен `-0`.
Для SAXPY ядро и эталон получают один и тот же `y`. На границах блока во входе есть нули и значения обоих знаков.

| Тест-кейс                                           | Ожидаемый результат                    | Фактический результат   |
| --------------------------------------------------- | -------------------------------------- | ----------------------- |
| `data/lab1.txt`, `N = 3`, `alpha = 0.5`, оба режима | `-1 0.5 0`, `-4 -4.5 1`, `4.5 2.5 4`   | файлы совпали, ошибка 0 |
| `N = 0`                                             | пустой `--output`, ядро не запускается | пустой файл             |
| `N = 255`, `256`, `257`, оба режима                 | совпадение add, SAXPY и ReLU с CPU     | максимальная ошибка 0   |

## Результаты

В таблице приведены основные результаты из таблицы [data/bench.csv](data/bench.csv).

Один прогревочный прогон, затем 10 повторов. Таймеры - `cudaEvent` с синхронизацией в конце интервала. В таблице приведены медианные результаты.  
Полный цикл - один интервал H2D -> выполнение трех ядер -> D2H.  
Столбец «Ядро» - сумма времен add, SAXPY и ReLU.

| Вариант | Блок | Размер | H2D, мс | Ядро, мс | D2H, мс | Полный цикл, мс |
| --- | --- | --- | --- | --- | --- | --- |
| coalesced | 128 | 2^20 | 0.411 | 0.027 | 0.712 | 1.149 |
| coalesced | 256 | 2^20 | 0.410 | 0.021 | 0.683 | 1.116 |
| coalesced | 512 | 2^20 | 0.410 | 0.022 | 0.848 | 1.283 |
| coalesced | 1024 | 2^20 | 0.410 | 0.022 | 0.665 | 1.098 |
| coalesced | 128 | 2^24 | 5.396 | 0.679 | 13.628 | 19.703 |
| coalesced | 256 | 2^24 | 5.448 | 0.668 | 8.143 | 14.255 |
| coalesced | 512 | 2^24 | 5.421 | 0.667 | 8.122 | 14.221 |
| coalesced | 1024 | 2^24 | 5.430 | 0.663 | 8.137 | 14.236 |
| noncoalesced | 128 | 2^20 | 0.412 | 0.094 | 0.667 | 1.176 |
| noncoalesced | 256 | 2^20 | 0.411 | 0.087 | 0.671 | 1.173 |
| noncoalesced | 512 | 2^20 | 0.410 | 0.088 | 0.661 | 1.158 |
| noncoalesced | 1024 | 2^20 | 0.410 | 0.085 | 0.672 | 1.170 |
| noncoalesced | 128 | 2^24 | 5.412 | 1.338 | 8.122 | 14.869 |
| noncoalesced | 256 | 2^24 | 5.404 | 1.175 | 10.610 | 17.189 |
| noncoalesced | 512 | 2^24 | 5.400 | 1.181 | 8.130 | 14.734 |
| noncoalesced | 1024 | 2^24 | 5.412 | 1.154 | 8.144 | 14.710 |

## Выводы

Цель лабораторной работы - сравнить разные способы доступа к памяти, поэтому анализировать надо колонку "Ядро".

- Коалесцированный доступ быстрее: warp забирает подряд идущие индексы одной транзакцией (об этом как раз говорится в [официальной документации](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/));
- Шаг 32 разводит данные для потоков одного warp по разным сегментам, из за чего Memory Coalescer не может оптимизировать обращение к памяти: чтение будет выполняться в несколько транзакций (скорее всего);
- Размер блока на коалесцированном доступе почти ничего не меняет;
- Большую часть итогового времени занимает перемещение данных с CPU на GPU и наоборот;

## Воспроизводимость

### Данные

Пример задания - `data/lab1.txt`: `N = 3`, `alpha = 0.5`.
Пустой вход - одна строка `0 0.5`.
Размеры 2^20 и 2^24 программа строит по `--synthetic` и `--alpha 0.5`. Каждый 17-й элемент равен нулю, остальные зависят только от индекса и имеют разные знаки. Отдельного seed нет.

### Полный запуск

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target submission
./build/submission --input data/lab1.txt --output coalesced.txt
./build/submission --input data/lab1.txt --output noncoalesced.txt --mode noncoalesced
rm -f data/bench.csv
for mode in coalesced noncoalesced; do
  for block in 128 256 512 1024; do
    for n in 1048576 16777216; do
      extra=()
      if [ "$mode" = noncoalesced ]; then extra=(--mode noncoalesced); fi
      ./build/submission --synthetic "$n" --alpha 0.5 --block "$block" "${extra[@]}" \
        --output /dev/null --csv data/bench.csv
    done
  done
done
```
