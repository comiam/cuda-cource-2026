/**
 * @file kernels.h
 * @brief CUDA kernels for Lab 1: vector add, SAXPY and ReLU.
 *
 * Two memory access patterns are provided:
 *   - Coalesced: consecutive threads touch consecutive floats.
 *   - Noncoalesced: consecutive threads touch far apart chunks of memory.
 */

#ifndef LAB1_KERNELS_H
#define LAB1_KERNELS_H

#include <cstddef>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
using cudaStream_t = void *;
#endif

namespace lab1 {

/** Vector operation supported by this submission. */
enum class Op {
    Add,   /**< c[i] = a[i] + b[i]            */
    Saxpy, /**< y[i] = alpha * x[i] + y[i]    */
    Relu   /**< y[i] = max(0.0f, x[i])        */
};

/** Global-memory access pattern. */
enum class AccessMode {
    Coalesced,   /**< Stride-1 contiguous access within a warp. */
    Noncoalesced /**< Chunked access, threads scatter across memory. */
};

/**
 * @brief Launch one of the vector kernels.
 */
void launchVectorOp(Op op, AccessMode mode, const float *d_a, const float *d_b, float *d_c,
                    float alpha, std::size_t n, int blockSize, cudaStream_t stream = 0);

} // namespace lab1

#endif // LAB1_KERNELS_H
