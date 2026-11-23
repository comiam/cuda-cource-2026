#include "kernels.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Проверка кода ошибки CUDA (лекция 2, слайд 16)
static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// Номер элемента для режима noncoalesced.
// Потоки делим на группы по 1024 (по 32 warp'а по 32 потока). Внутри группы элементы
// раскладываем "по столбцам": поток с номером lane в warp'е номер w берёт элемент lane * 32 + w.
// Соседние потоки одного warp'а (lane и lane + 1) попадают на элементы, которые лежат
// в памяти на расстоянии 32 float = 128 байт. Каждый элемент группы по-прежнему
// достаётся ровно одному потоку.
__device__ int strided_index(int i) {
    int group = i / 1024; // номер группы из 1024 потоков
    int t = i % 1024;     // номер потока внутри группы
    int lane = t % 32;    // номер потока внутри warp'а
    int warp = t / 32;    // номер warp'а внутри группы
    return group * 1024 + lane * 32 + warp;
}

__global__ void add_kernel(const float *x, const float *y, float *out, int n, bool strided) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = strided ? strided_index(i) : i; // j - номер элемента, который обрабатывает поток
    if (j < n) {
        out[j] = x[j] + y[j];
    }
}

__global__ void saxpy_kernel(float alpha, const float *x, const float *y, float *out, int n,
                             bool strided) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = strided ? strided_index(i) : i;
    if (j < n) {
        out[j] = alpha * x[j] + y[j];
    }
}

__global__ void relu_kernel(const float *x, float *out, int n, bool strided) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = strided ? strided_index(i) : i;
    if (j < n) {
        out[j] = x[j] > 0.0f ? x[j] : 0.0f;
    }
}

void compute_on_gpu(float alpha, const std::vector<float> &h_x, const std::vector<float> &h_y,
                    bool strided, std::vector<float> &h_add, std::vector<float> &h_saxpy,
                    std::vector<float> &h_relu, int threads, Timings *timings) {
    // Размер блока должен делить 1024: группы потоков в режиме noncoalesced по 1024
    if (threads < 32 || threads > 1024 || 1024 % threads != 0) {
        std::fprintf(stderr, "Bad block size: %d (use 32, 64, ..., 1024)\n", threads);
        std::exit(1);
    }
    const int n = static_cast<int>(h_x.size());
    const size_t bytes = n * sizeof(float);
    h_add.resize(n);
    h_saxpy.resize(n);
    h_relu.resize(n);

    // Память на GPU. Время выделения меряем CPU-таймером отдельно от остального.
    auto alloc_start = std::chrono::steady_clock::now();
    float *d_x = nullptr, *d_y = nullptr;
    float *d_add = nullptr, *d_saxpy = nullptr, *d_relu = nullptr;
    check(cudaMalloc(&d_x, bytes));
    check(cudaMalloc(&d_y, bytes));
    check(cudaMalloc(&d_add, bytes));
    check(cudaMalloc(&d_saxpy, bytes));
    check(cudaMalloc(&d_relu, bytes));
    auto alloc_end = std::chrono::steady_clock::now();

    // Сколько потоков запускать. В режиме noncoalesced группы по 1024 потока должны быть
    // полными, поэтому округляем число потоков вверх до кратного 1024.
    // Лишние потоки (их индекс получается >= n) ничего не делают благодаря проверке j < n.
    int total_threads = n;
    if (strided) {
        total_threads = (n + 1023) / 1024 * 1024;
    }
    const int blocks = (total_threads + threads - 1) / threads;

    // События CUDA для измерения времени на GPU (лекция 2, слайд 27):
    // ev[0] - начало, ev[1] - после H2D, ev[2] - после ядер, ev[3] - после D2H.
    cudaEvent_t ev[4];
    if (timings != nullptr) {
        for (int k = 0; k < 4; ++k) {
            check(cudaEventCreate(&ev[k]));
        }
    }
    auto cpu_start = std::chrono::steady_clock::now();

    // CPU -> GPU
    if (timings != nullptr) {
        check(cudaEventRecord(ev[0]));
    }
    check(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));
    if (timings != nullptr) {
        check(cudaEventRecord(ev[1]));
    }

    // Три ядра
    add_kernel<<<blocks, threads>>>(d_x, d_y, d_add, n, strided);
    check(cudaGetLastError());
    saxpy_kernel<<<blocks, threads>>>(alpha, d_x, d_y, d_saxpy, n, strided);
    check(cudaGetLastError());
    relu_kernel<<<blocks, threads>>>(d_x, d_relu, n, strided);
    check(cudaGetLastError());
    if (timings != nullptr) {
        check(cudaEventRecord(ev[2]));
    }

    // GPU -> CPU
    check(cudaMemcpy(h_add.data(), d_add, bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_saxpy.data(), d_saxpy, bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(h_relu.data(), d_relu, bytes, cudaMemcpyDeviceToHost));
    if (timings != nullptr) {
        check(cudaEventRecord(ev[3]));
    }

    // Ждём, пока GPU закончит всё; здесь видны ошибки выполнения ядер
    check(cudaDeviceSynchronize());
    auto cpu_end = std::chrono::steady_clock::now();

    if (timings != nullptr) {
        check(cudaEventElapsedTime(&timings->h2d_ms, ev[0], ev[1]));
        check(cudaEventElapsedTime(&timings->kernel_ms, ev[1], ev[2]));
        check(cudaEventElapsedTime(&timings->d2h_ms, ev[2], ev[3]));
        check(cudaEventElapsedTime(&timings->total_ms, ev[0], ev[3]));
        timings->alloc_ms =
            std::chrono::duration<float, std::milli>(alloc_end - alloc_start).count();
        timings->cpu_total_ms =
            std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
        for (int k = 0; k < 4; ++k) {
            check(cudaEventDestroy(ev[k]));
        }
    }

    // Освобождаем память GPU
    check(cudaFree(d_x));
    check(cudaFree(d_y));
    check(cudaFree(d_add));
    check(cudaFree(d_saxpy));
    check(cudaFree(d_relu));
}