# Lab 5: TensorRT RetinaNet - Детекция объектов

## Описание

Этот проект реализует высокопроизводительный инференс модели RetinaNet с использованием NVIDIA TensorRT с INT8 квантованием.

## Подготовка данных

### Калибровочные изображения

Для INT8 квантования требуются калибровочные изображения:

```bash
python scripts/download.py
```

### Экспорт модели

```bash
python scripts/export_model.py
```

### INT8 квантование

```bash
python scripts/int8_quantization.py
```

## Сборка проекта

```bash
./build.sh
```

## Запуск инференса

### Для изображения:

```bash
./run_inference.sh models/retinanet_int8_raw.trt input/test_image.jpg output/result.jpg 0.6
```

### Для видео:

```bash
./run_inference.sh models/retinanet_int8_raw.trt input/test_video.mp4 output/result.mp4 0.5
```

## Особенности реализации

### GPU постобработка

- Декодирование боксов на GPU
- Фильтрация по порогу уверенности
- NMS на GPU
- Сортировка кандидатов на GPU

### INT8 квантование

- Используется кастомный калибратор
- Поддержка реальных изображений из COCO dataset
- Кэширование результатов калибровки

### Архитектура RetinaNet

- Обработка "сырых" выходов модели (cls_logits, bbox_regression)
- Совместимость с динамическими формами
- Поддержка различных размеров входа

## Результаты

- Высокая производительность за счет INT8 квантования
- GPU ускорение для всей цепочки обработки
- Совместимость с современными версиями TensorRT
- Поддержка видео и изображений

## Зависимости

- CUDA 12.x
- TensorRT 10.x
- OpenCV 4.x
- Python 3.8+
- PyTorch
- ONNX
- CMake 3.18+
