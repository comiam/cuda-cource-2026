#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

void checkCuda(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

__global__ void addKernel(const float *x, const float *y, float *result, int n, bool noncoalesced) {
    long long index;

    if (noncoalesced) {
        index = static_cast<long long>(threadIdx.x) * gridDim.x + blockIdx.x;
    } else {
        index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    }

    if (index < n) {
        result[index] = x[index] + y[index];
    }
}

__global__ void saxpyKernel(const float *x, const float *y, float *result, float alpha, int n,
                            bool noncoalesced) {
    long long index;

    if (noncoalesced) {
        index = static_cast<long long>(threadIdx.x) * gridDim.x + blockIdx.x;
    } else {
        index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    }

    if (index < n) {
        result[index] = alpha * x[index] + y[index];
    }
}

__global__ void reluKernel(const float *x, float *result, int n, bool noncoalesced) {
    long long index;

    if (noncoalesced) {
        index = static_cast<long long>(threadIdx.x) * gridDim.x + blockIdx.x;
    } else {
        index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    }

    if (index < n) {
        result[index] = x[index] > 0.0f ? x[index] : 0.0f;
    }
}

struct Options {
    std::string input;
    std::string output;
    std::string mode = "coalesced";
    int block = 256;
    int repeat = 10;
    int warmup = 3;
    bool help = false;
};

void printHelp() {
    std::cout << "Использование:\n"
              << "  ./build/submission --input <файл> --output <файл> [параметры]\n\n"
              << "Параметры:\n"
              << "  --input <файл>       Входной файл: N alpha, затем N пар x y\n"
              << "  --output <файл>      Выходной файл: add saxpy relu в каждой строке\n"
              << "  --mode <режим>       coalesced или noncoalesced (по умолчанию coalesced)\n"
              << "  --block <число>      Потоков в блоке, от 1 до 1024 (по умолчанию 256)\n"
              << "  --repeat <число>     Число измерений, от 1 до 1000 (по умолчанию 10)\n"
              << "  --warmup <число>     Число прогревочных запусков (по умолчанию 3)\n"
              << "  --help               Показать эту справку\n";
}

int parsePositiveInt(const std::string &value, const std::string &name, int minValue,
                     int maxValue) {
    size_t used = 0;
    int result = std::stoi(value, &used);

    if (used != value.size() || result < minValue || result > maxValue) {
        throw std::invalid_argument("Неверное значение для " + name + ": " + value);
    }

    return result;
}

Options parseArgs(int argc, char **argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            options.help = true;
        } else if (arg == "--input" || arg == "--output" || arg == "--mode" || arg == "--block" ||
                   arg == "--repeat" || arg == "--warmup") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Не указано значение для " + arg);
            }

            std::string value = argv[++i];

            if (arg == "--input") {
                options.input = value;
            } else if (arg == "--output") {
                options.output = value;
            } else if (arg == "--mode") {
                options.mode = value;
            } else if (arg == "--block") {
                options.block = parsePositiveInt(value, arg, 1, 1024);
            } else if (arg == "--repeat") {
                options.repeat = parsePositiveInt(value, arg, 1, 1000);
            } else if (arg == "--warmup") {
                options.warmup = parsePositiveInt(value, arg, 0, 1000);
            }
        } else {
            throw std::invalid_argument("Неизвестный аргумент: " + arg);
        }
    }

    if (options.help) {
        return options;
    }

    if (options.input.empty() || options.output.empty()) {
        throw std::invalid_argument("Нужно указать --input и --output");
    }

    if (options.mode != "coalesced" && options.mode != "noncoalesced") {
        throw std::invalid_argument("Режим должен быть coalesced или noncoalesced");
    }

    return options;
}

struct InputData {
    int n = 0;
    float alpha = 0.0f;
    std::vector<float> x;
    std::vector<float> y;
};

InputData readInput(const std::string &path) {
    std::ifstream file(path);
    file.imbue(std::locale::classic());

    if (!file) {
        throw std::invalid_argument("Не удалось открыть входной файл: " + path);
    }

    long long nValue;
    InputData data;

    if (!(file >> nValue >> data.alpha)) {
        throw std::invalid_argument("В начале файла должны быть N и alpha");
    }

    if (nValue < 0 || nValue > 2147483647LL) {
        throw std::invalid_argument("N должно быть от 0 до 2147483647");
    }

    data.n = static_cast<int>(nValue);
    data.x.resize(data.n);
    data.y.resize(data.n);

    for (int i = 0; i < data.n; ++i) {
        if (!(file >> data.x[i] >> data.y[i])) {
            throw std::invalid_argument("Не хватает пар x y во входном файле");
        }
    }

    std::string extra;
    if (file >> extra) {
        throw std::invalid_argument("Во входном файле есть лишние данные");
    }

    return data;
}

template <typename Function>
float measureEvents(cudaEvent_t start, cudaEvent_t stop, Function function) {
    CUDA_CHECK(cudaEventRecord(start));
    function();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
}

struct Timings {
    double h2d = 0.0;
    double kernel = 0.0;
    double d2h = 0.0;
    double full = 0.0;
};

Timings runOnce(const InputData &input, const Options &options, cudaEvent_t start, cudaEvent_t stop,
                std::vector<float> &addResult, std::vector<float> &saxpyResult,
                std::vector<float> &reluResult) {
    float *deviceX = nullptr;
    float *deviceY = nullptr;
    float *deviceAdd = nullptr;
    float *deviceSaxpy = nullptr;
    float *deviceRelu = nullptr;

    Timings timings;
    const size_t bytes = static_cast<size_t>(input.n) * sizeof(float);
    const int blocks = (input.n - 1) / options.block + 1;
    const bool noncoalesced = options.mode == "noncoalesced";

    auto fullStart = std::chrono::steady_clock::now();

    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&deviceX), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&deviceY), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&deviceAdd), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&deviceSaxpy), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&deviceRelu), bytes));

        timings.h2d = measureEvents(start, stop, [&]() {
            CUDA_CHECK(cudaMemcpy(deviceX, input.x.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(deviceY, input.y.data(), bytes, cudaMemcpyHostToDevice));
        });

        timings.kernel = measureEvents(start, stop, [&]() {
            addKernel<<<blocks, options.block>>>(deviceX, deviceY, deviceAdd, input.n,
                                                 noncoalesced);
            CUDA_CHECK(cudaGetLastError());

            saxpyKernel<<<blocks, options.block>>>(deviceX, deviceY, deviceSaxpy, input.alpha,
                                                   input.n, noncoalesced);
            CUDA_CHECK(cudaGetLastError());

            reluKernel<<<blocks, options.block>>>(deviceX, deviceRelu, input.n, noncoalesced);
            CUDA_CHECK(cudaGetLastError());
        });

        timings.d2h = measureEvents(start, stop, [&]() {
            CUDA_CHECK(cudaMemcpy(addResult.data(), deviceAdd, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(saxpyResult.data(), deviceSaxpy, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(reluResult.data(), deviceRelu, bytes, cudaMemcpyDeviceToHost));
        });

        CUDA_CHECK(cudaFree(deviceX));
        deviceX = nullptr;
        CUDA_CHECK(cudaFree(deviceY));
        deviceY = nullptr;
        CUDA_CHECK(cudaFree(deviceAdd));
        deviceAdd = nullptr;
        CUDA_CHECK(cudaFree(deviceSaxpy));
        deviceSaxpy = nullptr;
        CUDA_CHECK(cudaFree(deviceRelu));
        deviceRelu = nullptr;
    } catch (...) {
        if (deviceX != nullptr)
            cudaFree(deviceX);
        if (deviceY != nullptr)
            cudaFree(deviceY);
        if (deviceAdd != nullptr)
            cudaFree(deviceAdd);
        if (deviceSaxpy != nullptr)
            cudaFree(deviceSaxpy);
        if (deviceRelu != nullptr)
            cudaFree(deviceRelu);
        throw;
    }

    auto fullEnd = std::chrono::steady_clock::now();
    timings.full = std::chrono::duration<double, std::milli>(fullEnd - fullStart).count();

    return timings;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;

    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }

    return values[middle];
}

void writeOutput(const std::string &path, const std::vector<float> &addResult,
                 const std::vector<float> &saxpyResult, const std::vector<float> &reluResult) {
    std::ofstream file(path);
    file.imbue(std::locale::classic());

    if (!file) {
        throw std::runtime_error("Не удалось открыть выходной файл: " + path);
    }

    file << std::setprecision(9);

    for (size_t i = 0; i < addResult.size(); ++i) {
        file << addResult[i] << ' ' << saxpyResult[i] << ' ' << reluResult[i] << '\n';
    }

    if (!file) {
        throw std::runtime_error("Ошибка записи в выходной файл: " + path);
    }
}

int main(int argc, char **argv) {
    Options options;

    try {
        options = parseArgs(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << "Ошибка аргументов: " << error.what() << '\n';
        printHelp();
        return 2;
    }

    if (options.help) {
        printHelp();
        return 0;
    }

    InputData input;

    try {
        input = readInput(options.input);
    } catch (const std::exception &error) {
        std::cerr << "Ошибка входных данных: " << error.what() << '\n';
        return 2;
    }

    if (input.n == 0) {
        try {
            writeOutput(options.output, {}, {}, {});
            std::cout << "N=0: создан пустой выходной файл, CUDA-ядра не запускались.\n";
            return 0;
        } catch (const std::exception &error) {
            std::cerr << "Ошибка: " << error.what() << '\n';
            return 1;
        }
    }

    std::vector<float> addResult(input.n);
    std::vector<float> saxpyResult(input.n);
    std::vector<float> reluResult(input.n);

    try {
        cudaEvent_t start;
        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        for (int i = 0; i < options.warmup; ++i) {
            runOnce(input, options, start, stop, addResult, saxpyResult, reluResult);
        }

        std::vector<double> h2dTimes;
        std::vector<double> kernelTimes;
        std::vector<double> d2hTimes;
        std::vector<double> fullTimes;

        for (int i = 0; i < options.repeat; ++i) {
            Timings timings =
                runOnce(input, options, start, stop, addResult, saxpyResult, reluResult);

            h2dTimes.push_back(timings.h2d);
            kernelTimes.push_back(timings.kernel);
            d2hTimes.push_back(timings.d2h);
            fullTimes.push_back(timings.full);
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        writeOutput(options.output, addResult, saxpyResult, reluResult);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Режим: " << options.mode << '\n';
        std::cout << "N: " << input.n << ", block: " << options.block
                  << ", repeat: " << options.repeat << '\n';
        std::cout << "H2D: " << median(h2dTimes) << " ms\n";
        std::cout << "Kernel (3 ядра): " << median(kernelTimes) << " ms\n";
        std::cout << "D2H: " << median(d2hTimes) << " ms\n";
        std::cout << "Полный цикл (с cudaMalloc и cudaFree): " << median(fullTimes) << " ms\n";
    } catch (const std::exception &error) {
        std::cerr << "Ошибка выполнения CUDA: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
