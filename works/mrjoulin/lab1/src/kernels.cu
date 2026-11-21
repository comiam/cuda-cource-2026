/**
 * @file kernels.cu
 * @brief Implementation of add / SAXPY / ReLU kernels in two access patterns.
 */

#include "kernels.h"

#include <algorithm>
#include <cuda_runtime.h>

namespace lab1 {

namespace {

// Each thread in the noncoalesced variants owns a contiguous chunk of
// kChunk elements. Adjacent threads therefore access memory with a stride of
// kChunk floats, breaking warp coalescing.
constexpr std::size_t kChunk = 64;

} // namespace

// ---------------------------------------------------------------------------
// Coalesced kernels: consecutive threads read/write consecutive floats.
// ---------------------------------------------------------------------------

__global__ void vectorAddCoalesced(const float *a, const float *b, float *c, std::size_t n) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;

    for (std::size_t i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorSaxpyCoalesced(const float *x, float *y, float alpha, std::size_t n) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;

    for (std::size_t i = idx; i < n; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}

__global__ void vectorReluCoalesced(const float *x, float *y, std::size_t n) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;

    for (std::size_t i = idx; i < n; i += stride) {
        y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Noncoalesced kernels: each thread owns a contiguous chunk.
// ---------------------------------------------------------------------------

__global__ void vectorAddNoncoalesced(const float *a, const float *b, float *c, std::size_t n) {
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t start = tid * kChunk;
    const std::size_t end = min(start + kChunk, n);

    for (std::size_t i = start; i < end; ++i) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorSaxpyNoncoalesced(const float *x, float *y, float alpha, std::size_t n) {
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t start = tid * kChunk;
    const std::size_t end = min(start + kChunk, n);

    for (std::size_t i = start; i < end; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

__global__ void vectorReluNoncoalesced(const float *x, float *y, std::size_t n) {
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t start = tid * kChunk;
    const std::size_t end = min(start + kChunk, n);

    for (std::size_t i = start; i < end; ++i) {
        y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

void launchVectorOp(Op op, AccessMode mode, const float *d_a, const float *d_b, float *d_c,
                    float alpha, std::size_t n, int blockSize, cudaStream_t stream) {
    if (n == 0) {
        return;
    }

    const int block = blockSize;
    int grid = 0;

    if (mode == AccessMode::Coalesced) {
        grid = static_cast<int>((n + block - 1) / block);
    } else {
        const std::size_t totalThreads = (n + kChunk - 1) / kChunk;
        grid = static_cast<int>((totalThreads + block - 1) / block);
    }

    // Run kernel
    if (mode == AccessMode::Coalesced) {
        switch (op) {
        case Op::Add:
            vectorAddCoalesced<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
            break;
        case Op::Saxpy:
            vectorSaxpyCoalesced<<<grid, block, 0, stream>>>(d_a, d_c, alpha, n);
            break;
        case Op::Relu:
            vectorReluCoalesced<<<grid, block, 0, stream>>>(d_a, d_c, n);
            break;
        }
    } else {
        switch (op) {
        case Op::Add:
            vectorAddNoncoalesced<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
            break;
        case Op::Saxpy:
            vectorSaxpyNoncoalesced<<<grid, block, 0, stream>>>(d_a, d_c, alpha, n);
            break;
        case Op::Relu:
            vectorReluNoncoalesced<<<grid, block, 0, stream>>>(d_a, d_c, n);
            break;
        }
    }
}

} // namespace lab1
