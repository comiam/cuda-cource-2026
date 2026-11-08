# CUDA Course 2026 @ ITMO

Курс по CUDA для студентов ITMO, 2026 год: от собственных GPU-ядер до сборки и обучения моделей ML/DL. Основные темы — организация памяти, коалесцированный доступ, bank conflicts, Map/Reduce/Scan, профилирование и оптимизация.

## Как сдавать лабораторные?

Подробные инструкции по сдаче работ и оформлению репозитория см. в документе [**SUBMISSIONS.md**](docs/SUBMISSIONS.md).

При открытии и обновлении PR GitHub Actions автоматически проверяет папку лабораторной, описание PR, форматирование исходников и README. Результат появится в PR как `submission/static`. [Правила сборки, запуска и оформления](docs/SUBMISSION_CONTRACT.md).

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

Полное описание работ, обязательных реализаций, проверок и критериев оценки см. в [**заданиях курса**](works/README.md).

### Список лабораторных

1. **Vector Operations and Activations** — Векторные операции и активации: индексация, coalescing, передачи CPU↔GPU.
2. **Matrix Multiplication** — Матричное умножение: naive → tiled, shared memory и сравнение с cuBLAS.
3. **Sobel Operator** — Оператор Собеля: двумерные stencil-вычисления, halo и работа с памятью.
4. **cuDNN Perceptron** — Создание и обучение перцептрона для sin(x): forward/backward, Map/Reduce и SGD.
5. **AlexNet-style CNN on CIFAR-100** — Самостоятельная сборка и обучение CV-модели: LayerNorm, softmax/cross-entropy, Scan и профилирование.

В Lab 4–5 вы сами собираете сеть и организуете обучение из своих и библиотечных операций. Свёртки можно брать из библиотек; готовая модель и autograd разрешены только для эталона.

Правила ведения своей папки с лабораторными см. в [**FOLDER_STRUCTURE.md**](docs/FOLDER_STRUCTURE.md).
