# Lab 2: Matrix Multiplication — Матричное умножение

## Описание

Контрольная сдача для проверки CI. Обе реализации считают `C = A * B` в FP32 и хранят матрицы по строкам. Каждый поток вычисляет один элемент выходного тайла. `naive` читает операнды из global memory; `tiled` загружает два тайла `16×16` в shared memory и переиспользует их внутри блока. Размер shared memory — `2 * 16 * 16 * sizeof(float) = 2048` байт на блок.

При неполном тайле потоки загружают нули за границами входа. Все потоки доходят до обоих барьеров: перед чтением из shared memory и перед её заполнением следующими данными. Только потоки внутри размеров `M×N` записывают результат. Оба ядра используют FP32 FMA.

## Окружение

Для сборки нужны CUDA Toolkit с `nvcc`, поддерживаемый им компилятор C++17 и CMake 3.18+. Для запуска нужен CUDA-совместимый GPU и драйвер. Библиотечный эталон использует cuBLAS из того же CUDA Toolkit. Флаги пониженной точности не включаются. Локальный запуск CUDA и измерения производительности в рамках этой проверки не проводились.

## Компиляция

Из каталога лабораторной:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target submission
```

Дополнительная цель `cublas_reference` не нужна для сборки обязательной цели. Чтобы собрать её:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLAB2_CUBLAS_REFERENCE=ON
cmake --build build --target cublas_reference
```

## Запуск

```bash
./build/submission --help
./build/submission --input lab2.txt --output naive.txt --mode naive
./build/submission --input lab2.txt --output tiled.txt --mode tiled
./build/submission --input lab2.txt --output default.txt
```

На входе `M K N`, затем A и B по строкам. В файле результата — `M N`, затем C по строкам. Без `--mode` выбирается `tiled`. Нулевые и отрицательные размеры, некорректный или нечитаемый вход и ошибки CLI приводят к коду `2` и сообщению в `stderr`; сбой CUDA или записи — к коду `1`.

## Проверка корректности

В примере из задания произведение матриц `2×3` и `3×2` имеет строки `58 64` и `139 154`. Для проверки неполных тайлов по всем осям используется отдельный вход `127×65` на `65×257` с небольшими целыми элементами. Для обоих входов произведения и частичные суммы точно представимы в FP32, поэтому файлы от `naive`, `tiled`, режима по умолчанию и cuBLAS должны совпасть.

Эталон — `cublasSgemm` с `CUBLAS_PEDANTIC_MATH`, без TF32. Порядок операндов для cuBLAS меняется: row-major C интерпретируется как column-major `Cᵀ = Bᵀ * Aᵀ`. Для общих дробных входов библиотека и ядра могут суммировать в разном порядке; побайтовое сравнение файлов в таком случае не годится. Допуск автопроверки из задания считается от отдельного FP64-эталона по входам, прочитанным в FP32.

## Результаты

Цель этих прогонов — проверить четыре вердикта CI, а не замерить производительность. Команды ниже воспроизводят публичную проверку корректности. Время ядра, GFLOP/s, влияние размера тайла и bank conflicts здесь не измерялись.

## Выводы

Загрузка двух тайлов позволяет переиспользовать элементы A вдоль строки потоков, а элементы B — вдоль столбца. У этого выигрыша есть цена: две синхронизации на итерацию и загрузка дополненных нулями хвостов. Сравнить стоимость на разных размерах и проверить влияние bank conflicts можно отдельными замерами; прохождение CI таких замеров не заменяет.

## Воспроизводимость

### Данные

Первый вход — открытый пример из задания Lab 2, приведённый целиком ниже. Для второго входа используется детерминированный шаблон из остатков от деления, без случайных чисел.

### Полный запуск

```bash
cat > lab2.txt <<'EOF'
2 3 2
1 2 3
4 5 6
7 8
9 10
11 12
EOF
printf '2 2\n58 64\n139 154\n' > expected.txt
python3 - <<'PY' > rectangular.txt
m, k, n = 127, 65, 257
print(m, k, n)
for row in range(m):
    print(*((row + 3 * col) % 13 - 6 for col in range(k)))
for row in range(k):
    print(*((2 * row + col) % 11 - 5 for col in range(n)))
PY
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLAB2_CUBLAS_REFERENCE=ON
cmake --build build --target submission cublas_reference
for input in lab2.txt rectangular.txt; do
  ./build/submission --input "$input" --output naive.txt --mode naive
  ./build/submission --input "$input" --output tiled.txt --mode tiled
  ./build/submission --input "$input" --output default.txt
  ./build/cublas_reference --input "$input" --output reference.txt
  diff -u reference.txt naive.txt
  diff -u reference.txt tiled.txt
  diff -u reference.txt default.txt
  if [ "$input" = lab2.txt ]; then diff -u expected.txt reference.txt; fi
done
```
