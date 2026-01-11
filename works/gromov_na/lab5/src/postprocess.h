#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include "structures.h"

// структура для представления кандидата обнаружения
struct Candidate
{
    float x1, y1, x2, y2;
    float score;
    int classId;
};

// функция для gpu постобработки
std::vector<Detection> retinanet_postprocess_gpu(
    const float *d_cls_logits,
    const float *d_bbox_deltas,
    int numAnchors,
    int numClasses,
    int inputW,
    int inputH,
    int origW,
    int origH,
    float confThreshold,
    float nmsThreshold,
    int maxCandidates,
    int topK,
    cudaStream_t stream = nullptr);

// функция для освобождения глобального потока
extern "C" void cleanup_postprocess_stream();

#endif