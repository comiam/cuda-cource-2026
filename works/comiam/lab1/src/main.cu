#include <cstddef>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct DeviceBuffer {
    float *data = nullptr;
    ~DeviceBuffer() { cudaFree(data); }
};

bool cuda_ok(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << cudaGetErrorString(status) << '\n';
        return false;
    }
    return true;
}

__global__ void vector_ops(float *values, std::size_t count, float alpha, bool noncoalesced) {
    const int lane = threadIdx.x;
    const int offset = noncoalesced ? (lane % 8) * 4 + lane / 8 : lane;
    const std::size_t index = blockIdx.x * blockDim.x + offset;
    if (index < count) {
        const float x = values[index];
        const float y = values[count + index];
        values[2 * count + index] = x + y;
        values[3 * count + index] = alpha * x + y;
        values[4 * count + index] = x > 0.0f ? x : 0.0f;
    }
}

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: submission --input FILE --output FILE [--mode MODE]\n";
        std::cout << "Modes: coalesced (default), noncoalesced\n";
        std::cout << "Input: N alpha, followed by N pairs of x and y\n";
        std::cout << "Output: N rows of add, SAXPY and ReLU\n";
        return 0;
    }

    std::string input_path;
    std::string output_path;
    std::string mode = "coalesced";
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            std::cerr << "Missing argument value\n";
            return 2;
        }
        const std::string option = argv[index];
        if (option == "--input") {
            input_path = argv[index + 1];
        } else if (option == "--output") {
            output_path = argv[index + 1];
        } else if (option == "--mode") {
            mode = argv[index + 1];
        } else {
            std::cerr << "Unknown argument: " << option << '\n';
            return 2;
        }
    }
    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Both --input and --output are required\n";
        return 2;
    }
    if (mode != "coalesced" && mode != "noncoalesced") {
        std::cerr << "Unknown mode: " << mode << '\n';
        return 2;
    }

    std::ifstream input(input_path);
    int count = 0;
    float alpha = 0.0f;
    if (!(input >> count >> alpha) || count < 0 || count > 1000000) {
        std::cerr << "Expected N alpha with 0 <= N <= 1000000\n";
        return 2;
    }
    const std::size_t size = static_cast<std::size_t>(count);
    std::vector<float> values(5 * size);
    for (std::size_t index = 0; index < size; ++index) {
        if (!(input >> values[index] >> values[size + index])) {
            std::cerr << "Missing input pair\n";
            return 2;
        }
    }

    if (count > 0) {
        DeviceBuffer gpu;
        const std::size_t bytes = values.size() * sizeof(float);
        const std::size_t input_bytes = 2 * size * sizeof(float);
        const std::size_t output_bytes = 3 * size * sizeof(float);
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&gpu.data), bytes))) {
            return 1;
        }
        if (!cuda_ok(cudaMemcpy(gpu.data, values.data(), input_bytes, cudaMemcpyHostToDevice))) {
            return 1;
        }
        const int blocks = (count + 31) / 32;
        vector_ops<<<blocks, 32>>>(gpu.data, size, alpha, mode == "noncoalesced");
        if (!cuda_ok(cudaGetLastError())) {
            return 1;
        }
        float *output = values.data() + 2 * size;
        float *gpu_output = gpu.data + 2 * size;
        if (!cuda_ok(cudaMemcpy(output, gpu_output, output_bytes, cudaMemcpyDeviceToHost))) {
            return 1;
        }
    }

    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "Cannot open output file\n";
        return 1;
    }
    for (std::size_t index = 0; index < size; ++index) {
        output << values[2 * size + index] << ' ';
        output << values[3 * size + index] << ' ';
        output << values[4 * size + index] << '\n';
    }
    if (!output) {
        std::cerr << "Cannot write output file\n";
        return 1;
    }
    std::cout << "Computed " << count << " rows\n";
    return 0;
}
