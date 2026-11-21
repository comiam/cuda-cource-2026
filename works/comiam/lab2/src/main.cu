#include <cuda_runtime.h>
#if defined(LAB2_CUBLAS_REFERENCE)
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int tile_side = 16;

struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Options {
    std::string input;
    std::string output;
    bool tiled = true;
};

struct Matrices {
    int m = 0;
    int k = 0;
    int n = 0;
    std::vector<float> a;
    std::vector<float> b;
};

void print_help() {
    std::cout << "Usage: --input PATH --output PATH [--mode naive|tiled]\n"
                 "       --help\n"
                 "Default mode: tiled. Input: M K N, then A and B in row-major order.\n"
                 "Output: M N, then the FP32 product A * B in row-major order.\n"
                 "Dimensions must be positive.\n";
}

Options parse_options(int argc, char **argv) {
    Options options;
    bool seen_mode = false;
    for (int i = 1; i < argc; i += 2) {
        const std::string_view flag = argv[i];
        if (i + 1 == argc || *argv[i + 1] == '\0') {
            throw InputError("Missing argument value");
        }
        const std::string_view value = argv[i + 1];
        if (flag == "--input" && options.input.empty()) {
            options.input = value;
        } else if (flag == "--output" && options.output.empty()) {
            options.output = value;
        } else if (flag == "--mode" && !seen_mode) {
            if (value != "naive" && value != "tiled") {
                throw InputError("Mode must be naive or tiled");
            }
            options.tiled = value == "tiled";
            seen_mode = true;
        } else {
            throw InputError("Unknown or repeated argument");
        }
    }
    if (options.input.empty() || options.output.empty()) {
        throw InputError("Both --input and --output are required");
    }
    return options;
}

std::size_t elements(int rows, int cols) {
    const auto r = static_cast<std::size_t>(rows);
    const auto c = static_cast<std::size_t>(cols);
    if (r > std::vector<float>{}.max_size() / c) {
        throw InputError("Matrix is too large");
    }
    return r * c;
}

std::vector<float> read_values(std::istream &input, std::size_t count) {
    std::vector<float> values;
    values.reserve(std::min<std::size_t>(count, 65536));
    for (std::size_t i = 0; i < count; ++i) {
        float value;
        if (!(input >> value) || !std::isfinite(value)) {
            throw InputError("Missing or invalid matrix value");
        }
        values.push_back(value);
    }
    return values;
}

Matrices read_matrices(const std::string &path) {
    std::ifstream input(path);
    Matrices matrices;
    if (!(input >> matrices.m >> matrices.k >> matrices.n) || matrices.m <= 0 || matrices.k <= 0 ||
        matrices.n <= 0) {
        throw InputError("Cannot read positive dimensions M K N");
    }
    matrices.a = read_values(input, elements(matrices.m, matrices.k));
    matrices.b = read_values(input, elements(matrices.k, matrices.n));
    input >> std::ws;
    if (!input.eof() || input.bad()) {
        throw InputError("Input contains extra data or could not be read");
    }
    return matrices;
}

void write_result(const std::string &path, int m, int n, const std::vector<float> &values) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open --output for writing");
    }
    output << m << ' ' << n << '\n' << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            if (col != 0) {
                output << ' ';
            }
            output << values[static_cast<std::size_t>(row) * n + col];
        }
        output << '\n';
    }
    output.close();
    if (!output) {
        throw std::runtime_error("Cannot write --output");
    }
}

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

class DeviceBuffer {
    float *data_ = nullptr;

  public:
    explicit DeviceBuffer(std::size_t size) {
        check_cuda(cudaMalloc(&data_, size * sizeof(float)));
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    float *data() const { return data_; }
};

#if defined(LAB2_CUBLAS_REFERENCE)

void check_blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error: " + std::to_string(status));
    }
}

class BlasHandle {
    cublasHandle_t handle_ = nullptr;

  public:
    BlasHandle() { check_blas(cublasCreate(&handle_)); }
    ~BlasHandle() { cublasDestroy(handle_); }
    BlasHandle(const BlasHandle &) = delete;
    BlasHandle &operator=(const BlasHandle &) = delete;
    cublasHandle_t get() const { return handle_; }
};

void launch(const Matrices &sizes, bool, const float *a, const float *b, float *c) {
    const BlasHandle blas;
    check_blas(cublasSetMathMode(blas.get(), CUBLAS_PEDANTIC_MATH));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // The row-major arrays look transposed to cuBLAS: C^T = B^T * A^T.
    check_blas(cublasSgemm(blas.get(), CUBLAS_OP_N, CUBLAS_OP_N, sizes.n, sizes.m, sizes.k, &alpha,
                           b, sizes.n, a, sizes.k, &beta, c, sizes.n));
    check_cuda(cudaDeviceSynchronize());
}

#else

__global__ void naive_kernel(const float *a, const float *b, float *c, int m, int k, int n,
                             std::size_t tile_cols, std::size_t tile_count) {
    for (std::size_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
        const std::size_t row = tile / tile_cols * tile_side + threadIdx.y;
        const std::size_t col = tile % tile_cols * tile_side + threadIdx.x;
        if (row < static_cast<std::size_t>(m) && col < static_cast<std::size_t>(n)) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                sum = __fmaf_rn(a[row * k + inner], b[static_cast<std::size_t>(inner) * n + col],
                                sum);
            }
            c[row * n + col] = sum;
        }
    }
}

__global__ void tiled_kernel(const float *a, const float *b, float *c, int m, int k, int n,
                             std::size_t tile_cols, std::size_t tile_count) {
    __shared__ float a_tile[tile_side][tile_side];
    __shared__ float b_tile[tile_side][tile_side];
    for (std::size_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
        const std::size_t row = tile / tile_cols * tile_side + threadIdx.y;
        const std::size_t col = tile % tile_cols * tile_side + threadIdx.x;
        const bool valid_row = row < static_cast<std::size_t>(m);
        const bool valid_col = col < static_cast<std::size_t>(n);
        const auto inner_size = static_cast<std::size_t>(k);
        float sum = 0.0f;
        for (std::size_t offset = 0; offset < inner_size; offset += tile_side) {
            const std::size_t a_col = offset + threadIdx.x;
            const std::size_t b_row = offset + threadIdx.y;
            a_tile[threadIdx.y][threadIdx.x] =
                valid_row && a_col < inner_size ? a[row * k + a_col] : 0.0f;
            b_tile[threadIdx.y][threadIdx.x] =
                valid_col && b_row < inner_size ? b[b_row * n + col] : 0.0f;
            __syncthreads();
            for (int inner = 0; inner < tile_side; ++inner) {
                sum = __fmaf_rn(a_tile[threadIdx.y][inner], b_tile[inner][threadIdx.x], sum);
            }
            __syncthreads();
        }
        if (valid_row && valid_col) {
            c[row * n + col] = sum;
        }
    }
}

void launch(const Matrices &sizes, bool tiled, const float *a, const float *b, float *c) {
    const auto tile_rows = (static_cast<std::size_t>(sizes.m) + tile_side - 1) / tile_side;
    const auto tile_cols = (static_cast<std::size_t>(sizes.n) + tile_side - 1) / tile_side;
    const auto tile_count = tile_rows * tile_cols;
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>(tile_count, 65535));
    const dim3 threads(tile_side, tile_side);
    if (tiled) {
        tiled_kernel<<<blocks, threads>>>(a, b, c, sizes.m, sizes.k, sizes.n, tile_cols,
                                          tile_count);
    } else {
        naive_kernel<<<blocks, threads>>>(a, b, c, sizes.m, sizes.k, sizes.n, tile_cols,
                                          tile_count);
    }
    check_cuda(cudaGetLastError());
}

#endif

std::vector<float> multiply(const Matrices &matrices, bool tiled) {
    std::vector<float> result(elements(matrices.m, matrices.n));
    const DeviceBuffer a(matrices.a.size());
    const DeviceBuffer b(matrices.b.size());
    const DeviceBuffer c(result.size());
    check_cuda(cudaMemcpy(a.data(), matrices.a.data(), matrices.a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(b.data(), matrices.b.data(), matrices.b.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    launch(matrices, tiled, a.data(), b.data(), c.data());
    check_cuda(
        cudaMemcpy(result.data(), c.data(), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        print_help();
        return 0;
    }
    try {
        const auto options = parse_options(argc, argv);
        const auto matrices = read_matrices(options.input);
        const auto result = multiply(matrices, options.tiled);
        write_result(options.output, matrices.m, matrices.n, result);
    } catch (const InputError &error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
