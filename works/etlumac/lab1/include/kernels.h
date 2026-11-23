#pragma once

#include <vector>

// Времена этапов одного запуска, в миллисекундах.
struct Timings {
    float alloc_ms = 0.0f; // cudaMalloc пяти массивов (CPU-таймер)
    float h2d_ms = 0.0f; // копирование x и y на GPU (GPU-таймер, события CUDA)
    float kernel_ms = 0.0f; // три ядра подряд: add, saxpy, relu (события CUDA)
    float d2h_ms = 0.0f; // копирование трёх результатов на CPU (события CUDA)
    float total_ms = 0.0f; // H2D + ядра + D2H по событиям CUDA, без cudaMalloc
    float cpu_total_ms = 0.0f; // тот же интервал, измеренный CPU-таймером
};

// Выполняет на GPU три операции над массивами x и y одинаковой длины n > 0:
//   h_add   = x + y
//   h_saxpy = alpha * x + y
//   h_relu  = max(0, x)
// strided = false: режим coalesced, соседние потоки читают соседние элементы.
// strided = true:  режим noncoalesced, соседние потоки читают элементы, разнесённые в памяти.
// Результаты в обоих режимах одинаковые, отличается только порядок обращений к памяти.
// threads - размер блока: 32, 64, ..., 1024 (степень двойки).
// Если передан timings, в него записываются времена этапов.
void compute_on_gpu(float alpha, const std::vector<float> &h_x, const std::vector<float> &h_y,
                    bool strided, std::vector<float> &h_add, std::vector<float> &h_saxpy,
                    std::vector<float> &h_relu, int threads = 256, Timings *timings = nullptr);