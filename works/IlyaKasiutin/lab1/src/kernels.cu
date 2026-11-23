#include "kernels.h"

__device__ int nonCoalescedIndex(int tid, int n) {
    constexpr int warpSize = 32;
    constexpr int tile = warpSize * warpSize;
    const int full = n - n % tile;
    if (tid >= full) {
        return tid;
    }
    const int local = tid % tile;
    const int lane = local % warpSize;
    const int warpId = local / warpSize;
    return tid - local + lane * warpSize + warpId;
}

__global__ void addCoalesced(const float* d_x, const float* d_y, float* d_out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_out[i] = d_x[i] + d_y[i];
    }
}

__global__ void addNonCoalesced(const float* d_x, const float* d_y, float* d_out, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        const int i = nonCoalescedIndex(tid, n);
        d_out[i] = d_x[i] + d_y[i];
    }
}

__global__ void saxpyCoalesced(const float* d_x, const float* d_y, float* d_out, float alpha,
                               int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_out[i] = alpha * d_x[i] + d_y[i];
    }
}

__global__ void saxpyNonCoalesced(const float* d_x, const float* d_y, float* d_out, float alpha,
                                  int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        const int i = nonCoalescedIndex(tid, n);
        d_out[i] = alpha * d_x[i] + d_y[i];
    }
}

__global__ void reluCoalesced(const float* d_x, float* d_out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_out[i] = d_x[i] > 0.f ? d_x[i] : 0.f;
    }
}

__global__ void reluNonCoalesced(const float* d_x, float* d_out, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        const int i = nonCoalescedIndex(tid, n);
        d_out[i] = d_x[i] > 0.f ? d_x[i] : 0.f;
    }
}
