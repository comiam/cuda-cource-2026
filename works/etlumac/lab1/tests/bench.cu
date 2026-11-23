// Замеры времени: H2D, ядра, D2H и полный цикл для обоих режимов доступа к памяти.
// Запуск: ./build/bench   (результат - таблицы в формате Markdown)
#include "kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

const int WARMUP = 3; // прогревочные запуски, не учитываются
const int REPS = 20;  // число повторов, по ним считаем медиану

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Делает WARMUP + REPS запусков и возвращает медиану по каждому показателю.
Timings measure(int n, bool strided, int threads) {
    std::mt19937 gen(2026); // seed
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = dist(gen);
        y[i] = dist(gen);
    }
    std::vector<float> r_add, r_saxpy, r_relu;

    std::vector<float> alloc, h2d, kernel, d2h, total, cpu_total;
    for (int run = 0; run < WARMUP + REPS; ++run) {
        Timings t;
        compute_on_gpu(0.75f, x, y, strided, r_add, r_saxpy, r_relu, threads, &t);
        if (run < WARMUP) {
            continue;
        }
        alloc.push_back(t.alloc_ms);
        h2d.push_back(t.h2d_ms);
        kernel.push_back(t.kernel_ms);
        d2h.push_back(t.d2h_ms);
        total.push_back(t.total_ms);
        cpu_total.push_back(t.cpu_total_ms);
    }
    Timings m;
    m.alloc_ms = median(alloc);
    m.h2d_ms = median(h2d);
    m.kernel_ms = median(kernel);
    m.d2h_ms = median(d2h);
    m.total_ms = median(total);
    m.cpu_total_ms = median(cpu_total);
    return m;
}

int main() {
    cudaDeviceProp prop;
    int driver = 0, runtime = 0;
    cudaGetDeviceProperties(&prop, 0);
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::printf("GPU: %s, %.1f GB, compute capability %d.%d\n", prop.name,
                prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    std::printf("CUDA driver %d, CUDA runtime %d\n", driver, runtime);
    std::printf("Warmup: %d runs, repeats: %d, values are medians\n\n", WARMUP, REPS);

    // Таблица 1: разные размеры массива, блок 256 потоков
    std::printf("Table 1: block = 256 threads\n\n");
    std::printf("| Mode | N | alloc, ms | H2D, ms | kernels, ms | D2H, ms | GPU total, ms | "
                "CPU total, ms |\n");
    std::printf("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    const int sizes[] = {1 << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 24};
    for (int n : sizes) {
        for (int strided = 0; strided < 2; ++strided) {
            Timings t = measure(n, strided == 1, 256);
            std::printf("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
                        strided ? "noncoalesced" : "coalesced", n, t.alloc_ms, t.h2d_ms,
                        t.kernel_ms, t.d2h_ms, t.total_ms, t.cpu_total_ms);
        }
    }

    // Таблица 2: большой массив, разные размеры блока.
    // Эффективная пропускная способность: три ядра читают и пишут 8 массивов по n float
    // (add: 2 чтения + 1 запись, saxpy: 2 + 1, relu: 1 + 1), всего 32 * n байт.
    const int big = 1 << 24;
    std::printf("\nTable 2: N = %d\n\n", big);
    std::printf("| Mode | block | kernels, ms | effective GB/s | GPU total, ms |\n");
    std::printf("| --- | --- | --- | --- | --- |\n");
    const int blocks[] = {32, 64, 128, 256, 512, 1024};
    for (int strided = 0; strided < 2; ++strided) {
        for (int threads : blocks) {
            Timings t = measure(big, strided == 1, threads);
            double gbps = 32.0 * big / (t.kernel_ms * 1e-3) / 1e9;
            std::printf("| %s | %d | %.3f | %.1f | %.3f |\n",
                        strided ? "noncoalesced" : "coalesced", threads, t.kernel_ms, gbps,
                        t.total_ms);
        }
    }
    return 0;
}