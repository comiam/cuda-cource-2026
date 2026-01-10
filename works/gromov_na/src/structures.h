#ifndef SCTRUCTURES_H
#define SCTRUCTURES_H

#include <vector>
#include <string>

/**
 * @brief структура для представления одного обнаружения объекта
 */
struct Detection {
    float bbox[4];          // x1, y1, x2, y2 координаты в оригинальном изображении
    float confidence;       // уверенность в обнаружении
    int class_id;           // id класса объекта
    std::string class_name; // имя класса объекта
};

/**
 * @brief структура для представления анкор бокса
 */
struct AnchorBox {
    float x_center;
    float y_center;
    float width;
    float height;
};

/**
 * @brief структура для хранения параметров модели
 */
struct ModelParams {
    int input_width;
    int input_height;
    int num_classes;
    float conf_threshold;
    float nms_threshold;
    std::string model_path;
    std::string engine_path;
    std::string labels_path;
};

#endif