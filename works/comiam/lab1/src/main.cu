#include <cuda_runtime.h>

#include <charconv>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr unsigned warp_size = 32;
constexpr unsigned threads_per_block = 256;

struct Options {
    const char *input = nullptr;
    const char *output = nullptr;
    bool noncoalesced = false;
};

bool parse_options(int argc, char **argv, Options &options) {
    if (argc % 2 == 0) {
        return false;
    }
    for (int i = 1; i < argc; i += 2) {
        const std::string_view flag = argv[i];
        const std::string_view value = argv[i + 1];
        if (value.empty() || value.substr(0, 2) == "--") {
            return false;
        }
        if (flag == "--input") {
            options.input = argv[i + 1];
        } else if (flag == "--output") {
            options.output = argv[i + 1];
        } else if (flag == "--mode" && (value == "coalesced" || value == "noncoalesced")) {
            options.noncoalesced = value == "noncoalesced";
        } else {
            return false;
        }
    }
    return options.input != nullptr && options.output != nullptr;
}

template <bool Noncoalesced>
__global__ void operations(const float *input, float *output, std::size_t n, float alpha) {
    unsigned offset = threadIdx.x;
    if (Noncoalesced) {
        // Transpose the 8-by-32 thread indices: lanes in a warp are eight elements apart.
        offset = (offset % warp_size) * (threads_per_block / warp_size) + offset / warp_size;
    }
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * threads_per_block + offset;
    if (index < n) {
        const float x = input[index];
        const float y = input[n + index];
        output[index] = x + y;
        output[n + index] = __fmaf_rn(alpha, x, y);
        output[2 * n + index] = x > 0.0f ? x : 0.0f;
    }
}

void check_cuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

struct DeviceBuffer {
    float *data = nullptr;
    ~DeviceBuffer() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }
};

int run(const Options &options) {
    std::ifstream input(options.input);
    std::string count_token;
    float alpha = 0;
    if (!(input >> count_token >> alpha) || !std::isfinite(alpha)) {
        std::cerr << "Invalid input header\n";
        return 2;
    }
    unsigned count = 0;
    const char *end = count_token.data() + count_token.size();
    const char *begin = count_token.data() + (count_token.front() == '+' ? 1 : 0);
    const auto parsed = std::from_chars(begin, end, count);
    if (parsed.ec != std::errc{} || parsed.ptr != end || count > std::numeric_limits<int>::max()) {
        std::cerr << "Invalid vector length\n";
        return 2;
    }
    std::vector<float> x, y;
    for (unsigned i = 0; i < count; ++i) {
        float left = 0, right = 0;
        if (!(input >> left >> right) || !std::isfinite(left) || !std::isfinite(right)) {
            std::cerr << "Invalid or missing input pair\n";
            return 2;
        }
        x.push_back(left);
        y.push_back(right);
    }
    input >> std::ws;
    if (input.bad() || !input.eof()) {
        std::cerr << "Unexpected data after the input pairs\n";
        return 2;
    }

    std::ofstream output(options.output);
    if (!output) {
        std::cerr << "Cannot open output file\n";
        return 1;
    }
    if (count != 0) {
        const std::size_t n = count;
        const std::size_t bytes = n * sizeof(float);
        DeviceBuffer device;
        check_cuda(cudaMalloc(&device.data, 5 * bytes));
        check_cuda(cudaMemcpy(device.data, x.data(), bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(device.data + n, y.data(), bytes, cudaMemcpyHostToDevice));
        float *output_on_device = device.data + 2 * n;
        const unsigned blocks = (count + threads_per_block - 1) / threads_per_block;
        if (options.noncoalesced) {
            operations<true>
                <<<blocks, threads_per_block>>>(device.data, output_on_device, n, alpha);
        } else {
            operations<false>
                <<<blocks, threads_per_block>>>(device.data, output_on_device, n, alpha);
        }
        check_cuda(cudaGetLastError());
        std::vector<float> values(3 * n);
        check_cuda(cudaMemcpy(values.data(), output_on_device, 3 * bytes, cudaMemcpyDeviceToHost));

        output << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (std::size_t i = 0; i < n; ++i) {
            output << values[i] << ' ' << values[n + i] << ' ' << values[2 * n + i] << '\n';
        }
    }
    output.close();
    if (!output) {
        std::cerr << "Cannot write output file\n";
        return 1;
    }
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "submission --input FILE --output FILE [--mode coalesced|noncoalesced]\n"
                     "Default mode: coalesced.\n"
                     "Input: N alpha, then N pairs x y. Output: N triples add, SAXPY, ReLU.\n";
        return 0;
    }
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::cerr << "Invalid arguments; use --help\n";
        return 2;
    }
    try {
        return run(options);
    } catch (const std::exception &error) {
        std::cerr << "Execution failed: " << error.what() << '\n';
        return 1;
    }
}
