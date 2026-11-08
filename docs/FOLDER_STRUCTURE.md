# Структура папок с лабораторными

Задания находятся в [works/README.md](../works/README.md), правила сборки и оформления — в [SUBMISSION_CONTRACT.md](SUBMISSION_CONTRACT.md).

## Папка студента

Создайте папку `works/<github-login>/` с вашим логином GitHub. Регистр букв значения не имеет.

```text
works/ivan-ivanov/
├── lab1/
├── lab2/
├── lab3/
├── lab4/
└── lab5/
```

В каждый PR включайте файлы только одной лабораторной.

## Структура лабораторной

```text
lab1/
├── README.md          # Отчёт
├── CMakeLists.txt     # Цель submission
├── src/
│   ├── main.cu
│   └── kernels.cu
├── include/           # По необходимости
│   └── kernels.h
├── tests/             # По необходимости
├── data/              # Небольшие примеры, например PNG
└── build/             # Локальная сборка, в Git не добавлять
```

Обязательны `README.md`, `CMakeLists.txt` и `src/` с хотя бы одним `.cu`. Имена и разбиение исходников выбирайте сами.

## CMake и запуск

Используем цель `submission`. Для показанной структуры подойдёт такой `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(lab1 LANGUAGES CXX CUDA)

add_executable(submission src/main.cu src/kernels.cu)
target_include_directories(submission PRIVATE include)
set_target_properties(submission PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
```

Укажите свои исходники и библиотеки. Архитектуру GPU и версии CUDA выбирайте под оборудование и приведите в отчёте. Собранная программа должна находиться в `build/submission`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target submission
./build/submission --help
./build/submission --input data/example --output results
```

Режимы необязательного `--mode` приведены в [правилах запуска](SUBMISSION_CONTRACT.md#cli-программы). Форматы входов и результатов опишите в README.

## Отчёт

Оформите README по [шаблону](STUDENT_TEMPLATE.md). Сохраните обязательные разделы: `Описание`, `Окружение`, `Компиляция`, `Запуск`, `Проверка корректности`, `Результаты`, `Выводы`, `Воспроизводимость`. Для Lab 3–5 добавьте `Профилирование и оптимизация`.

В `Воспроизводимость` нужны подразделы `### Данные` и `### Полный запуск`; для Lab 4–5 — ещё `### Параметры обучения` с seed и `### Загрузка весов`. Приведите команды сборки, запуска и получения результатов, а также собственные измерения и выводы.

## Файлы в Git

Добавляйте исходники, отчёт и небольшие тестовые примеры. Сборочные файлы, `submission`, библиотеки, виртуальные окружения, датасеты, большие веса и сырые профили оставляйте вне Git. Симлинки и submodules для сдачи не допускаются.

Лимиты: 10 MiB на файл, 25 MiB и 1000 файлов на лабораторную. Небольшие PNG и вспомогательные данные разрешены.

Свои тесты храните в `works/<login>/lab<N>/tests/`. Правило `.gitignore` для тестов обвеса касается только корневой папки `tests/submission_check/`.

## Форматирование

Используйте конфиги курса: clang-format для C/C++/CUDA, Ruff format для Python, cmake-format для CMake. У основного README проверяются разделы и Markdown. Остальные текстовые файлы и данные не линтуются.

## Сдача

1. Работайте в своей папке и отдельной ветке форка.
2. Пишите понятные сообщения коммитов, например «Добавлена Lab 1: векторные операции и ReLU».
3. Откройте PR по [инструкции](SUBMISSIONS.md).
4. Дождитесь результата GitHub Actions. После исправлений запушьте изменения в ту же ветку — проверка запустится снова.
