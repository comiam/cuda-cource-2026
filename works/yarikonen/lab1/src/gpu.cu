#include "gpu.hpp"
#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;
using Vector = std::vector<float>;

void cuda_check(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

struct DeviceMemory {
    float *data = nullptr;
    ~DeviceMemory() {
        if (data) {
            cudaFree(data);
        }
    }
};

struct KernelEvents {
    std::array<cudaEvent_t, 4> marks{};
    ~KernelEvents() {
        for (auto mark : marks) {
            if (mark) {
                cudaEventDestroy(mark);
            }
        }
    }

    void create() {
        for (auto &mark : marks) {
            cuda_check(cudaEventCreate(&mark), "cudaEventCreate");
        }
    }

    void record(unsigned i) const { cuda_check(cudaEventRecord(marks[i]), "cudaEventRecord"); }

    float duration(unsigned i) const {
        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, marks[i], marks[i + 1]), "cudaEventElapsedTime");
        return ms;
    }
};

double ms(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

using Timings = std::array<double, 7>;

void report_timings(std::vector<Timings> &samples, double allocation) {
    constexpr std::array labels{"h2d", "add", "saxpy", "relu", "kernels", "d2h", "cycle"};
    std::cout << std::fixed << std::setprecision(6) << "allocation_ms=" << allocation;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        std::sort(samples.begin(), samples.end(),
                  [i](const auto &a, const auto &b) { return a[i] < b[i]; });
        const auto count = samples.size();
        const double median = (samples[(count - 1) / 2][i] + samples[count / 2][i]) / 2;
        std::cout << ' ' << labels[i] << "_ms=" << median;
    }
    std::cout << '\n';
}

} // namespace

Output compute(const Input &input, bool noncoalesced, unsigned block, unsigned repeat) {
    const auto n = input.hX.size();
    Output out{Vector(n), Vector(n), Vector(n)};
    if (n == 0) {
        return out;
    }

    const auto bytes = n * sizeof(float);
    const auto grid = static_cast<unsigned>((n + block - 1) / block);

    auto &hAdd = out[0];
    auto &hSaxpy = out[1];
    auto &hRelu = out[2];
    DeviceMemory dX, dY, dAdd, dSaxpy, dRelu;
    if (repeat) {
        cuda_check(cudaFree(nullptr), "initialize CUDA");
    }
    const auto allocation_start = Clock::now();
    cuda_check(cudaMalloc(&dX.data, bytes), "allocate x");
    cuda_check(cudaMalloc(&dY.data, bytes), "allocate y");
    cuda_check(cudaMalloc(&dAdd.data, bytes), "allocate add");
    cuda_check(cudaMalloc(&dSaxpy.data, bytes), "allocate SAXPY");
    cuda_check(cudaMalloc(&dRelu.data, bytes), "allocate ReLU");
    const double allocation = ms(Clock::now() - allocation_start);

    KernelEvents events;
    if (repeat) {
        events.create();
    }
    auto cycle = [&](bool measure) -> Timings {
        const auto start = Clock::now();
        cuda_check(cudaMemcpy(dX.data, input.hX.data(), bytes, cudaMemcpyHostToDevice), "H2D x");
        cuda_check(cudaMemcpy(dY.data, input.hY.data(), bytes, cudaMemcpyHostToDevice), "H2D y");
        // Pageable H2D may return while DMA is pending.
        cuda_check(cudaStreamSynchronize(nullptr), "complete H2D");
        const auto h2d_end = Clock::now();

        if (measure)
            events.record(0);
        add<<<grid, block>>>(dX.data, dY.data, dAdd.data, n, noncoalesced);
        cuda_check(cudaGetLastError(), "launch add");
        if (measure)
            events.record(1);
        saxpy<<<grid, block>>>(dX.data, dY.data, dSaxpy.data, input.alpha, n, noncoalesced);
        cuda_check(cudaGetLastError(), "launch SAXPY");
        if (measure)
            events.record(2);
        relu<<<grid, block>>>(dX.data, dRelu.data, n, noncoalesced);
        cuda_check(cudaGetLastError(), "launch ReLU");
        if (measure)
            events.record(3);

        cuda_check(cudaStreamSynchronize(nullptr), "complete kernels");
        const auto d2h_start = Clock::now();
        cuda_check(cudaMemcpy(hAdd.data(), dAdd.data, bytes, cudaMemcpyDeviceToHost), "D2H add");
        cuda_check(cudaMemcpy(hSaxpy.data(), dSaxpy.data, bytes, cudaMemcpyDeviceToHost),
                   "D2H SAXPY");
        cuda_check(cudaMemcpy(hRelu.data(), dRelu.data, bytes, cudaMemcpyDeviceToHost), "D2H ReLU");
        const auto end = Clock::now();

        if (!measure) {
            return {};
        }
        const double add_ms = events.duration(0);
        const double saxpy_ms = events.duration(1);
        const double relu_ms = events.duration(2);
        return {ms(h2d_end - start), add_ms,         saxpy_ms, relu_ms, add_ms + saxpy_ms + relu_ms,
                ms(end - d2h_start), ms(end - start)};
    };

    cycle(false);
    if (repeat) {
        std::vector<Timings> samples;
        for (unsigned i = 0; i < repeat; ++i) {
            samples.push_back(cycle(true));
        }
        report_timings(samples, allocation);
    }
    return out;
}
