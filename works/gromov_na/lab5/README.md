# Lab 5: TensorRT RetinaNet - Детекция объектов

## Описание

Этот проект реализует высокопроизводительный инференс модели RetinaNet с использованием NVIDIA TensorRT с INT8 квантованием. Поддерживаются как ResNet50, так и MobileNetV3 backbone архитектуры.

## Подготовка данных

### Калибровочные изображения

Для INT8 квантования требуются калибровочные изображения:

```bash
python scripts/download.py
```

### Экспорт модели

#### ResNet50 FPN

```bash
python scripts/export_model.py
```

Файлы:

- `models/retinanet_raw_heads.onnx` - ONNX модель ResNet50
- `models/labels.txt` - файл с названиями классов

#### MobileNetV3 FPN

```bash
python scripts/export_and_transfer_weights.py
```

Файлы:

- `models/retinanet_mobilenet_int8_transferred_weights.onnx` - ONNX модель MobileNet с перенесенными весами
- `models/labels_mobilenet.txt` - файл с названиями классов

### INT8 квантование

#### Для ResNet50:

```bash
python scripts/int8_quantization.py
```

Файлы:

- `models/retinanet_int8_raw.trt` - INT8 TensorRT engine для ResNet50

#### Для MobileNet:

```bash
python scripts/quantize_mobilenet.py
```

Файлы:

- `models/retinanet_mobilenet_int8_raw_transferred_weights.trt` - INT8 TensorRT engine для MobileNet

## Сборка проекта

```bash
./build.sh
```

## Запуск инференса

### Для изображения:

```bash
# ResNet50
./run_inference.sh models/retinanet_int8_raw.trt input/test_image.jpg output/result.jpg 0.6

# MobileNet
./run_inference.sh models/retinanet_mobilenet_int8_raw_transferred_weights.trt input/test_image.jpg output/result.jpg 0.6
```

### Для видео:

```bash
# ResNet50
./run_inference.sh models/retinanet_int8_raw.trt input/test_video.mp4 output/result.mp4 0.5

# MobileNet
./run_inference.sh models/retinanet_mobilenet_int8_raw_transferred_weights.trt input/test_video.mp4 output/result.mp4 0.5
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

### MobileNetV3 FPN

- Адаптация архитектуры для совместимости с C++ пост-обработкой
- Перенос весов головы из предобученной ResNet50 модели
- Корректное количество анкоров (76,725) для совместимости с пост-обработкой

## Сравнение моделей

Сравнение производилось на видео с confidence=0.5 в обоих случаях

| Модель      | Время обработки | Среднее время инференса | Обнаруженные объекты |
| ----------- | --------------- | ----------------------- | -------------------- |
| ResNet50    | 8.95 сек        | 18.50 ms/кадр           | 2084                 |
| MobileNetV3 | 7.10 сек        | 12.50 ms/кадр           | 6854                 |

**Выводы:**

- MobileNetV3 показывает **на 32% быстрее** среднее время инференса по сравнению с ResNet50
- MobileNetV3 обнаруживает **в 3.3 раза больше объектов**, но в данном случае это указывает на худшее качество модели
- Общее время обработки у MobileNetV3 на **20% меньше**

## Результаты

- Высокая производительность за счет INT8 квантования
- GPU ускорение для всей цепочки обработки
- Совместимость с современными версиями TensorRT
- Поддержка видео и изображений
- Поддержка различных backbone архитектур (ResNet50, MobileNetV3)

## Зависимости

- CUDA 12.x
- TensorRT 10.x
- OpenCV 4.x
- Python 3.8+
- PyTorch
- ONNX
- CMake 3.18+
