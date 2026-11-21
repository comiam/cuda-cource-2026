#error LAB3_BUILD_LOG_FINAL_COMPILER_MARKER

#include "lodepng.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr unsigned side = 16;
constexpr unsigned stride = side + 2;
using Pixel = unsigned char;

struct Options {
    std::string input;
    std::string output;
    bool shared = false;
};

struct Image {
    unsigned width = 0;
    unsigned height = 0;
    std::vector<Pixel> gray;
};

void help() {
    std::cout << "Usage: submission --input INPUT.png --output OUTPUT.png [--mode global|shared]\n"
                 "       submission --help\n"
                 "Input: 8-bit grayscale or RGB PNG. Output: single-channel 8-bit grayscale PNG.\n"
                 "Default mode: global.\n";
}

Options parse(int argc, char **argv) {
    Options result;
    bool seen_mode = false;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc || *argv[i + 1] == '\0') {
            throw std::invalid_argument("Missing argument value");
        }
        const std::string_view flag = argv[i];
        const std::string_view value = argv[i + 1];
        if (flag == "--input" && result.input.empty()) {
            result.input = value;
        } else if (flag == "--output" && result.output.empty()) {
            result.output = value;
        } else if (flag == "--mode" && !seen_mode && (value == "global" || value == "shared")) {
            result.shared = value == "shared";
            seen_mode = true;
        } else {
            throw std::invalid_argument("Unknown or repeated argument, or invalid mode");
        }
    }
    if (result.input.empty() || result.output.empty()) {
        throw std::invalid_argument("Both --input and --output are required");
    }
    return result;
}

Image read_png(const std::string &path) {
    Image image;
    std::vector<Pixel> rgb;
    const unsigned status = lodepng::decode(rgb, image.width, image.height, path, LCT_RGB, 8);
    if (status != 0) {
        throw std::invalid_argument(lodepng_error_text(status));
    }
    image.gray.resize(rgb.size() / 3);
    for (std::size_t i = 0; i < image.gray.size(); ++i) {
        image.gray[i] = static_cast<Pixel>(
            (299 * rgb[3 * i] + 587 * rgb[3 * i + 1] + 114 * rgb[3 * i + 2] + 500) / 1000);
    }
    return image;
}

void write_png(const std::string &path, unsigned width, unsigned height,
               const std::vector<Pixel> &pixels) {
    lodepng::State state;
    state.info_raw.colortype = LCT_GREY;
    state.info_raw.bitdepth = 8;
    state.info_png.color.colortype = LCT_GREY;
    state.info_png.color.bitdepth = 8;
    state.encoder.auto_convert = 0;
    std::vector<Pixel> png;
    unsigned status = lodepng::encode(png, pixels, width, height, state);
    if (status == 0) {
        status = lodepng::save_file(png, path);
    }
    if (status != 0) {
        throw std::runtime_error(lodepng_error_text(status));
    }
}

__device__ Pixel gradient(const Pixel *upper, const Pixel *middle, const Pixel *lower) {
    const int gx = upper[2] - upper[0] + 2 * (middle[2] - middle[0]) + lower[2] - lower[0];
    const int gy = lower[0] - upper[0] + 2 * (lower[1] - upper[1]) + lower[2] - upper[2];
    const int value = static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)) + 0.5f);
    return static_cast<Pixel>(value > 255 ? 255 : value);
}

__global__ void global_sobel(const Pixel *input, Pixel *output, unsigned width, unsigned height) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const std::size_t offset = static_cast<std::size_t>(y) * width + x;
    if (x == 0 || y == 0 || x + 1 == width || y + 1 == height) {
        output[offset] = 0;
        return;
    }
    const Pixel *middle = input + offset - 1;
    output[offset] = gradient(middle - width, middle, middle + width);
}

__global__ void shared_sobel(const Pixel *input, Pixel *output, unsigned width, unsigned height) {
    __shared__ Pixel tile[stride][stride];
    const unsigned bx = blockIdx.x * side;
    const unsigned by = blockIdx.y * side;
    for (unsigned i = threadIdx.y * side + threadIdx.x; i < stride * stride; i += side * side) {
        const unsigned row = i / stride;
        const unsigned col = i % stride;
        const unsigned source_y = by + row - 1;
        const unsigned source_x = bx + col - 1;
        tile[row][col] = source_y < height && source_x < width
                             ? input[static_cast<std::size_t>(source_y) * width + source_x]
                             : 0;
    }
    __syncthreads();
    const unsigned x = bx + threadIdx.x;
    const unsigned y = by + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const std::size_t offset = static_cast<std::size_t>(y) * width + x;
    const unsigned row = threadIdx.y;
    const unsigned col = threadIdx.x;
    output[offset] = x == 0 || y == 0 || x + 1 == width || y + 1 == height
                         ? 0
                         : gradient(&tile[row][col], &tile[row + 1][col], &tile[row + 2][col]);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t size) { check(cudaMalloc(&data, size)); }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    Pixel *data = nullptr;
};

std::vector<Pixel> filter(const Image &image, bool shared) {
    const auto size = image.gray.size();
    std::vector<Pixel> result(size);
    const DeviceBuffer input(size);
    const DeviceBuffer output(size);
    check(cudaMemcpy(input.data, image.gray.data(), size, cudaMemcpyHostToDevice));
    const dim3 threads(side, side);
    const dim3 blocks(1 + (image.width - 1) / side, 1 + (image.height - 1) / side);
    if (shared) {
        shared_sobel<<<blocks, threads>>>(input.data, output.data, image.width, image.height);
    } else {
        global_sobel<<<blocks, threads>>>(input.data, output.data, image.width, image.height);
    }
    check(cudaGetLastError());
    check(cudaMemcpy(result.data(), output.data, size, cudaMemcpyDeviceToHost));
    return result;
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        help();
        return 0;
    }
    try {
        const Options options = parse(argc, argv);
        const Image image = read_png(options.input);
        auto result = filter(image, options.shared);
        write_png(options.output, image.width, image.height, result);
    } catch (const std::invalid_argument &error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
