# CUDA Course 2026 @ ITMO

Курс по CUDA для студентов ITMO, 2026 год: от собственных GPU-ядер до сборки и обучения моделей ML/DL. Основные темы — организация памяти, коалесцированный доступ, bank conflicts, Map/Reduce/Scan, профилирование и оптимизация.

## Как сдавать лабораторные?

- [Как сделать форк, вести ветки лабораторных и открыть PR](docs/SUBMISSIONS.md).
- [Как разложить файлы и что добавить в Git](docs/FOLDER_STRUCTURE.md).
- [Как собрать, запустить и отформатировать программу](docs/SUBMISSION_CONTRACT.md).
- [Как оформить README](docs/STUDENT_TEMPLATE.md) и [как проверить Lab 1](works/lab1.md#пример-для-самопроверки) и [Lab 2](works/lab2.md#пример-для-самопроверки).

В PR появится результат проверки оформления, а для Lab 1–2 — и проверки корректности. Подробности и замечания откройте через **Details**.

## Ресурсы курса

### 📚 Лекции

Все лекции доступны в Google Drive:
[Перейти к лекциям](https://drive.google.com/drive/folders/1nD6wztlwnD9DzYuP7p8BT7sEf7VrR1WM?usp=drive_link)

### 📊 Успеваемость

Таблица с оценками и статусом сдачи работ:
[Таблица успеваемости](https://docs.google.com/spreadsheets/d/15mnUixh-tuvF0MDaxjjgQkzAt1khLK3N0QR0J9miKbU/edit?usp=sharing)

### 🖥️ Кластер

Ссылка на инструкции по доступу к кластеру для выполнения лабораторных работ:
[Инструкции по кластеру](https://disk.yandex.ru/i/k28sZe_xws4bAg)

## 📋 Лабораторные работы

Задания лежат в папке [`works/`](works/). В [оглавлении](works/README.md) собраны ссылки на все пять работ и общие требования. Свои решения кладите в `works/<ваш-логин>/lab<N>/`.

### Список лабораторных

1. **Vector Operations and Activations** — Векторные операции и активации: индексация, coalescing, передачи CPU↔GPU.
2. **Matrix Multiplication** — Матричное умножение: naive → tiled, shared memory и сравнение с cuBLAS.
3. **Sobel Operator** — Оператор Собеля: двумерные stencil-вычисления, halo и работа с памятью.
4. **cuDNN Perceptron** — Создание и обучение перцептрона для sin(x): forward/backward, Map/Reduce и SGD.
5. **AlexNet-style CNN on CIFAR-100** — Самостоятельная сборка и обучение CV-модели: LayerNorm, softmax/cross-entropy, Scan и профилирование.

В Lab 4–5 вы сами собираете сеть и организуете обучение из своих и библиотечных операций. Свёртки можно брать из библиотек; готовая модель и autograd разрешены только для эталона.

Правила ведения своей папки с лабораторными см. в [**FOLDER_STRUCTURE.md**](docs/FOLDER_STRUCTURE.md).
