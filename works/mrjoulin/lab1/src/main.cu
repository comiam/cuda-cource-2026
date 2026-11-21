/**
 * @file main.cu
 * @brief Lab 1 submission: vector add, SAXPY and ReLU with coalesced and
 *        noncoalesced global-memory access.
 *
 * The program reads a small text configuration from --input, runs a correctness
 * suite on small sizes and a timing benchmark on a large size, then writes a
 * report to --output.
 */

#include "config.h"
#include "kernels.h"
#include "report.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace lab1 {

namespace {

constexpr int kWarmupIterations = 2;
constexpr int kDefaultValidationBlockSize = 256;
constexpr float kTolerance = 1e-5f;

} // namespace

// ---------------------------------------------------------------------------
// Error checking helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                         cudaGetErrorString(err));                                                 \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                                        \
    do {                                                                                           \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "Kernel launch error at %s:%d: %s\n", __FILE__, __LINE__,         \
                         cudaGetErrorString(err));                                                 \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// Host-side reference results
// ---------------------------------------------------------------------------

static void computeReference(Op op, const std::vector<float> &a, const std::vector<float> &b,
                             std::vector<float> &out, float alpha) {
    const std::size_t n = out.size();

    switch (op) {
    case Op::Add:
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
        break;

    case Op::Saxpy:
        // SAXPY overwrites y; here b holds the original y.
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = alpha * a[i] + b[i];
        }
        break;

    case Op::Relu:
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
        }
        break;
    }
}

static void generateHostInputs(std::size_t n, unsigned seed, std::vector<float> &x,
                               std::vector<float> &y) {
    x.resize(n);
    y.resize(n);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = dist(gen);
        y[i] = dist(gen);
    }
}

static void compareWithReference(const std::vector<float> &ref, const std::vector<float> &got,
                                 float &maxErr, std::size_t &mismatches, float tol = kTolerance) {
    maxErr = 0.0f;
    mismatches = 0;

    for (std::size_t i = 0; i < ref.size(); ++i) {
        float err = std::fabs(ref[i] - got[i]);

        if (err > maxErr) {
            maxErr = err;
        }

        if (err > tol) {
            ++mismatches;
        }
    }
}

// ---------------------------------------------------------------------------
// Validation on small sizes
// ---------------------------------------------------------------------------

static ValidationResult validateOp(Op op, AccessMode mode, std::size_t n, float alpha,
                                   int blockSize) {
    std::vector<float> x, y;
    generateHostInputs(n, 2026u, x, y);

    std::vector<float> ref(n);
    computeReference(op, x, y, ref, alpha);

    std::vector<float> result(n, 0.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_a, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        if (op == Op::Add || op == Op::Saxpy) {
            CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_b, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        }

        // For SAXPY the launcher treats d_c as y.
        if (op == Op::Saxpy) {
            CUDA_CHECK(cudaMemcpy(d_c, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        }

        int block = blockSize;
        if (block > static_cast<int>(n)) {
            block = static_cast<int>(n);
            if (block == 0) {
                block = 1;
            }
        }

        launchVectorOp(op, mode, d_a, d_b, d_c, alpha, n, block);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(result.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_c));
        if (d_b != nullptr) {
            CUDA_CHECK(cudaFree(d_b));
        }
    }

    ValidationResult res;
    res.opName = (op == Op::Add ? "add" : (op == Op::Saxpy ? "saxpy" : "relu"));
    res.n = n;
    compareWithReference(ref, result, res.maxErr, res.mismatches);

    return res;
}

static std::vector<ValidationResult> runValidation(AccessMode mode, float alpha) {
    const std::vector<std::size_t> sizes = {0, 1, 255, 256, 257, 1023, 1024, 1025};

    std::vector<ValidationResult> results;

    for (std::size_t n : sizes) {
        for (auto op : {Op::Add, Op::Saxpy, Op::Relu}) {
            results.push_back(validateOp(op, mode, n, alpha, kDefaultValidationBlockSize));
        }
    }

    return results;
}

// ---------------------------------------------------------------------------
// Benchmark on the configured large size
// ---------------------------------------------------------------------------

static float medianOf(std::vector<float> &values) {
    size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

static BenchmarkResult benchmarkOp(Op op, AccessMode mode, const std::vector<float> &x,
                                   const std::vector<float> &y, float alpha, int blockSize,
                                   int repeats) {
    const std::size_t n = x.size();

    BenchmarkResult result;
    result.opName = (op == Op::Add ? "add" : (op == Op::Saxpy ? "saxpy" : "relu"));
    result.blockSize = blockSize;

    // Allocate device memory once and measure allocation time separately.
    cudaEvent_t allocStart, allocStop;
    CUDA_CHECK(cudaEventCreate(&allocStart));
    CUDA_CHECK(cudaEventCreate(&allocStop));

    CUDA_CHECK(cudaEventRecord(allocStart));

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
    if (op == Op::Add || op == Op::Saxpy) {
        CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    }

    CUDA_CHECK(cudaEventRecord(allocStop));
    CUDA_CHECK(cudaEventSynchronize(allocStop));
    CUDA_CHECK(cudaEventElapsedTime(&result.allocMs, allocStart, allocStop));
    CUDA_CHECK(cudaEventDestroy(allocStart));
    CUDA_CHECK(cudaEventDestroy(allocStop));

    std::vector<float> resultHost(n);

    cudaEvent_t h2dStart, h2dStop, kernelStart, kernelStop, d2hStart, d2hStop, fullStart, fullStop;
    CUDA_CHECK(cudaEventCreate(&h2dStart));
    CUDA_CHECK(cudaEventCreate(&h2dStop));
    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelStop));
    CUDA_CHECK(cudaEventCreate(&d2hStart));
    CUDA_CHECK(cudaEventCreate(&d2hStop));
    CUDA_CHECK(cudaEventCreate(&fullStart));
    CUDA_CHECK(cudaEventCreate(&fullStop));

    std::vector<Timing> timings;
    timings.reserve(repeats + kWarmupIterations);

    for (int iter = 0; iter < kWarmupIterations + repeats; ++iter) {
        CUDA_CHECK(cudaEventRecord(fullStart));

        // H2D copies: both inputs are re-uploaded every iteration so the
        // measurement includes the real transfer cost.
        CUDA_CHECK(cudaEventRecord(h2dStart));
        CUDA_CHECK(cudaMemcpy(d_a, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        if (op == Op::Add) {
            CUDA_CHECK(cudaMemcpy(d_b, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        } else if (op == Op::Saxpy) {
            // y is the second operand and the destination.
            CUDA_CHECK(cudaMemcpy(d_c, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaEventRecord(h2dStop));

        // Kernel.
        CUDA_CHECK(cudaEventRecord(kernelStart));
        launchVectorOp(op, mode, d_a, d_b, d_c, alpha, n, blockSize);
        CUDA_CHECK(cudaEventRecord(kernelStop));

        // D2H.
        CUDA_CHECK(cudaEventRecord(d2hStart));
        CUDA_CHECK(cudaMemcpy(resultHost.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(d2hStop));

        CUDA_CHECK(cudaEventRecord(fullStop));
        CUDA_CHECK(cudaEventSynchronize(fullStop));

        if (iter >= kWarmupIterations) {
            Timing t;
            CUDA_CHECK(cudaEventElapsedTime(&t.h2dMs, h2dStart, h2dStop));
            CUDA_CHECK(cudaEventElapsedTime(&t.kernelMs, kernelStart, kernelStop));
            CUDA_CHECK(cudaEventElapsedTime(&t.d2hMs, d2hStart, d2hStop));
            CUDA_CHECK(cudaEventElapsedTime(&t.fullMs, fullStart, fullStop));
            timings.push_back(t);
        }
    }

    CUDA_CHECK(cudaEventDestroy(h2dStart));
    CUDA_CHECK(cudaEventDestroy(h2dStop));
    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelStop));
    CUDA_CHECK(cudaEventDestroy(d2hStart));
    CUDA_CHECK(cudaEventDestroy(d2hStop));
    CUDA_CHECK(cudaEventDestroy(fullStart));
    CUDA_CHECK(cudaEventDestroy(fullStop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_c));
    if (d_b != nullptr) {
        CUDA_CHECK(cudaFree(d_b));
    }

    std::vector<float> h2d, kern, d2h, full;
    for (const auto &t : timings) {
        h2d.push_back(t.h2dMs);
        kern.push_back(t.kernelMs);
        d2h.push_back(t.d2hMs);
        full.push_back(t.fullMs);
    }

    result.median.h2dMs = medianOf(h2d);
    result.median.kernelMs = medianOf(kern);
    result.median.d2hMs = medianOf(d2h);
    result.median.fullMs = medianOf(full);

    return result;
}

// ---------------------------------------------------------------------------
// Helpers for the final large-array sanity check
// ---------------------------------------------------------------------------

static void checkLargeOperation(Op op, AccessMode mode, const std::vector<float> &x,
                                const std::vector<float> &y, const std::vector<float> &ref,
                                float alpha, std::size_t n, int blockSize, float *d_a, float *d_b,
                                float *d_c, std::vector<float> &result) {
    if (op == Op::Saxpy) {
        CUDA_CHECK(cudaMemcpy(d_c, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }

    launchVectorOp(op, mode, d_a, d_b, d_c, alpha, n, blockSize);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(result.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    float maxErr;
    std::size_t mismatches;
    compareWithReference(ref, result, maxErr, mismatches);

    std::cerr << (op == Op::Add ? "Add" : (op == Op::Saxpy ? "SAXPY" : "ReLU"))
              << " large check: max error = " << maxErr << ", mismatches = " << mismatches << "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static std::string modeName(AccessMode mode) {
    return mode == AccessMode::Coalesced ? "coalesced" : "noncoalesced";
}

int main(int argc, char **argv) {
    Config cfg = parseArgs(argc, argv);
    readConfigFile(cfg.inputPath, cfg);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cerr << "Device: " << prop.name << " (compute " << prop.major << "." << prop.minor
              << ")\n";
    std::cerr << "Mode: " << modeName(cfg.mode) << "\n";
    std::cerr << "Benchmark size: " << cfg.n << "\n";

    std::cerr << "Running validation...\n";
    std::vector<ValidationResult> validation = runValidation(cfg.mode, cfg.alpha);

    std::cerr << "Generating inputs...\n";
    std::vector<float> x, y;
    generateHostInputs(cfg.n, cfg.seed, x, y);

    std::cerr << "Running benchmark...\n";
    std::vector<BenchmarkResult> benchmarks;

    if (cfg.blockSize == 0) {
        for (int bs : {64, 128, 256, 512, 1024}) {
            benchmarks.push_back(benchmarkOp(Op::Add, cfg.mode, x, y, cfg.alpha, bs, cfg.repeats));
            benchmarks.push_back(
                benchmarkOp(Op::Saxpy, cfg.mode, x, y, cfg.alpha, bs, cfg.repeats));
            benchmarks.push_back(benchmarkOp(Op::Relu, cfg.mode, x, y, cfg.alpha, bs, cfg.repeats));
        }
    } else {
        benchmarks.push_back(
            benchmarkOp(Op::Add, cfg.mode, x, y, cfg.alpha, cfg.blockSize, cfg.repeats));
        benchmarks.push_back(
            benchmarkOp(Op::Saxpy, cfg.mode, x, y, cfg.alpha, cfg.blockSize, cfg.repeats));
        benchmarks.push_back(
            benchmarkOp(Op::Relu, cfg.mode, x, y, cfg.alpha, cfg.blockSize, cfg.repeats));
    }

    // Write report to txt file
    writeReport(cfg, validation, benchmarks);

    std::cerr << "Checking large-result correctness...\n";
    std::vector<float> refAdd(cfg.n), refSaxpy(cfg.n), refRelu(cfg.n);
    computeReference(Op::Add, x, y, refAdd, cfg.alpha);
    computeReference(Op::Saxpy, x, y, refSaxpy, cfg.alpha);
    computeReference(Op::Relu, x, y, refRelu, cfg.alpha);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, cfg.n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, cfg.n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, cfg.n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, x.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, y.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> result(cfg.n);
    int effectiveBlockSize = cfg.blockSize != 0 ? cfg.blockSize : 256;

    checkLargeOperation(Op::Add, cfg.mode, x, y, refAdd, cfg.alpha, cfg.n, effectiveBlockSize, d_a,
                        d_b, d_c, result);
    checkLargeOperation(Op::Saxpy, cfg.mode, x, y, refSaxpy, cfg.alpha, cfg.n, effectiveBlockSize,
                        d_a, d_b, d_c, result);
    checkLargeOperation(Op::Relu, cfg.mode, x, y, refRelu, cfg.alpha, cfg.n, effectiveBlockSize,
                        d_a, nullptr, d_c, result);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return EXIT_SUCCESS;
}

} // namespace lab1

int main(int argc, char **argv) { return lab1::main(argc, argv); }
