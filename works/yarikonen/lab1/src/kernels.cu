#include "kernels.cuh"

#include <cuda_runtime.h>

namespace {

constexpr unsigned tile_side = 32;
constexpr unsigned tile_size = tile_side * tile_side;

__device__ std::size_t element_index(std::size_t t, std::size_t n, bool noncoalesced) {
    if (noncoalesced) {
        const auto base = t / tile_size * tile_size;
        const auto local = t % tile_size;
        if (n - base >= tile_size) {
            return base + (local % tile_side) * tile_side + local / tile_side;
        }
    }
    return t;
}

} // namespace

__global__ void add(const float *dX, const float *dY, float *dAdd, std::size_t n,
                    bool noncoalesced) {
    const auto t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t < n) {
        const auto i = element_index(t, n, noncoalesced);
        dAdd[i] = dX[i] + dY[i];
    }
}

__global__ void saxpy(const float *dX, const float *dY, float *dSaxpy, float alpha, std::size_t n,
                      bool noncoalesced) {
    const auto t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t < n) {
        const auto i = element_index(t, n, noncoalesced);
        dSaxpy[i] = __fadd_rn(__fmul_rn(alpha, dX[i]), dY[i]);
    }
}

__global__ void relu(const float *dX, float *dRelu, std::size_t n, bool noncoalesced) {
    const auto t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t < n) {
        const auto i = element_index(t, n, noncoalesced);
        dRelu[i] = fmaxf(0.0f, dX[i]);
    }
}
