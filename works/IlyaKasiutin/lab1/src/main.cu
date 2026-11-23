#include "kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Args {
    std::string input;
    std::string output;
    std::string mode;
    std::string csv;
    int block = 256;
    int repeats = 10;
    bool synthetic = false;
    int syntheticN = 0;
    float alpha = 0.5f;
};

struct Timings {
    float allocMs = 0.f;
    float h2dMs = 0.f;
    float kernelAddMs = 0.f;
    float kernelSaxpyMs = 0.f;
    float kernelReluMs = 0.f;
    float kernelMs = 0.f;
    float d2hMs = 0.f;
    float fullMs = 0.f;
    float maxErrAdd = 0.f;
    float maxErrSaxpy = 0.f;
    float maxErrRelu = 0.f;
};

namespace {

void fail(int code, const std::string &message) {
    std::cerr << message << '\n';
    std::exit(code);
}

void printHelp() {
    std::cout
        << "Использование:\n"
        << "  submission --input <путь> --output <путь> [--mode <режим>]\n"
        << "  submission --synthetic <N> --output <путь> [--alpha <число>] [--mode <режим>]\n"
        << "\n"
        << "Аргументы:\n"
        << "  --input <путь>     вход: N alpha, затем N пар x y\n"
        << "  --output <путь>    для каждой пары: x+y, alpha*x+y, max(0, x)\n"
        << "  --mode <режим>     coalesced (по умолчанию) или noncoalesced\n"
        << "  --block <int>      потоков в блоке (по умолчанию 256)\n"
        << "  --repeats <int>    измеряемые повторы после одного прогрева (по умолчанию 10)\n"
        << "  --csv <путь>       дописать строку замеров: медиана повторов, выделение отдельно\n"
        << "  --synthetic <N>    сгенерировать N пар на хосте вместо --input\n"
        << "  --alpha <число>    коэффициент SAXPY для --synthetic (по умолчанию 0.5)\n"
        << "  --help             показать эту справку и выйти с кодом 0\n"
        << "\n"
        << "Полный цикл в CSV — это H2D, три ядра и D2H. cudaMalloc в него не входит.\n";
}

int parseInt(const std::string &text) {
    char *end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        fail(2, "ожидалось целое число, получено: " + text);
    }
    return static_cast<int>(value);
}

float parseFloat(const std::string &text) {
    char *end = nullptr;
    const float value = std::strtof(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0') {
        fail(2, "ожидалось число, получено: " + text);
    }
    return value;
}

float medianOf(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const size_t count = values.size();
    if (count % 2 == 1) {
        return values[count / 2];
    }
    return 0.5f * (values[count / 2 - 1] + values[count / 2]);
}

void fillSynthetic(int n, std::vector<float> &h_x, std::vector<float> &h_y) {
    h_x.assign(static_cast<size_t>(n), 0.f);
    h_y.assign(static_cast<size_t>(n), 0.f);
    for (int i = 0; i < n; ++i) {
        if (i % 17 == 0) {
            continue;
        }
        float vx = static_cast<float>((i % 1000) - 500) * 0.01f;
        if ((i / 17) % 2 == 0) {
            vx = -vx;
        }
        h_x[static_cast<size_t>(i)] = vx;
        h_y[static_cast<size_t>(i)] = static_cast<float>(((i * 3) % 200) - 100) * 0.05f;
    }
}

} // namespace

Args parseArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            printHelp();
            std::exit(0);
        }
    }

    Args args;
    args.mode = "coalesced";
    bool hasInput = false;
    bool hasOutput = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&]() {
            if (i + 1 >= argc) {
                fail(2, "нет значения для " + arg);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--input") {
            args.input = needValue();
            hasInput = true;
        } else if (arg == "--output") {
            args.output = needValue();
            hasOutput = true;
        } else if (arg == "--mode") {
            args.mode = needValue();
        } else if (arg == "--block") {
            args.block = parseInt(needValue());
        } else if (arg == "--repeats") {
            args.repeats = parseInt(needValue());
        } else if (arg == "--csv") {
            args.csv = needValue();
        } else if (arg == "--synthetic") {
            args.syntheticN = parseInt(needValue());
            args.synthetic = true;
        } else if (arg == "--alpha") {
            args.alpha = parseFloat(needValue());
        } else {
            fail(2, "неизвестный аргумент: " + arg);
        }
    }

    if (args.mode != "coalesced" && args.mode != "noncoalesced") {
        fail(2, "режим должен быть coalesced или noncoalesced");
    }
    if (!hasOutput) {
        fail(2, "нужен аргумент --output");
    }
    if (!args.synthetic && !hasInput) {
        fail(2, "нужны аргументы --input и --output");
    }
    return args;
}

void readInput(const std::string &path, int &n, float &alpha, std::vector<float> &h_x,
               std::vector<float> &h_y) {
    std::ifstream in(path);
    if (!in) {
        fail(2, "не удалось открыть входной файл: " + path);
    }
    if (!(in >> n >> alpha) || n < 0) {
        fail(2, "ожидались N и alpha");
    }

    h_x.resize(static_cast<size_t>(n));
    h_y.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (!(in >> h_x[static_cast<size_t>(i)] >> h_y[static_cast<size_t>(i)])) {
            fail(2, "недостаточно пар x y");
        }
    }
}

void checkCuda(cudaError_t status) {
    if (status == cudaSuccess) {
        return;
    }
    std::cerr << "ошибка CUDA: " << cudaGetErrorString(status) << '\n';
    std::exit(1);
}

void run(const std::string &mode, const std::vector<float> &h_x, const std::vector<float> &h_y,
         float alpha, int block, int repeats, std::vector<float> &h_add,
         std::vector<float> &h_saxpy, std::vector<float> &h_relu, Timings &timings) {
    const int n = static_cast<int>(h_x.size());
    h_add.assign(static_cast<size_t>(n), 0.f);
    h_saxpy.assign(static_cast<size_t>(n), 0.f);
    h_relu.assign(static_cast<size_t>(n), 0.f);
    if (n == 0) {
        return;
    }

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_x = nullptr;
    float *d_y = nullptr;
    float *d_add = nullptr;
    float *d_saxpy = nullptr;
    float *d_relu = nullptr;
    cudaEvent_t marks[6] = {};
    bool eventsCreated = false;

    auto release = [&]() {
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_add);
        cudaFree(d_saxpy);
        cudaFree(d_relu);
        if (eventsCreated) {
            for (cudaEvent_t ev : marks) {
                cudaEventDestroy(ev);
            }
            eventsCreated = false;
        }
    };
    auto check = [&](cudaError_t status) {
        if (status != cudaSuccess) {
            release();
            checkCuda(status);
        }
    };

    for (cudaEvent_t &ev : marks) {
        check(cudaEventCreate(&ev));
    }
    eventsCreated = true;

    check(cudaEventRecord(marks[0]));
    check(cudaMalloc(&d_x, bytes));
    check(cudaMalloc(&d_y, bytes));
    check(cudaMalloc(&d_add, bytes));
    check(cudaMalloc(&d_saxpy, bytes));
    check(cudaMalloc(&d_relu, bytes));
    check(cudaEventRecord(marks[1]));
    check(cudaEventSynchronize(marks[1]));
    check(cudaEventElapsedTime(&timings.allocMs, marks[0], marks[1]));

    const int grid = static_cast<int>((static_cast<long long>(n) + block - 1) / block);
    const bool coalesced = mode == "coalesced";

    std::vector<float> h2dSamples;
    std::vector<float> addSamples;
    std::vector<float> saxpySamples;
    std::vector<float> reluSamples;
    std::vector<float> kernelSamples;
    std::vector<float> d2hSamples;
    std::vector<float> fullSamples;

    auto measureOnce = [&](bool record) {
        check(cudaEventRecord(marks[0]));
        check(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));
        check(cudaEventRecord(marks[1]));
        if (coalesced) {
            addCoalesced<<<grid, block>>>(d_x, d_y, d_add, n);
        } else {
            addNonCoalesced<<<grid, block>>>(d_x, d_y, d_add, n);
        }
        check(cudaEventRecord(marks[2]));
        if (coalesced) {
            saxpyCoalesced<<<grid, block>>>(d_x, d_y, d_saxpy, alpha, n);
        } else {
            saxpyNonCoalesced<<<grid, block>>>(d_x, d_y, d_saxpy, alpha, n);
        }
        check(cudaEventRecord(marks[3]));
        if (coalesced) {
            reluCoalesced<<<grid, block>>>(d_x, d_relu, n);
        } else {
            reluNonCoalesced<<<grid, block>>>(d_x, d_relu, n);
        }
        check(cudaEventRecord(marks[4]));
        check(cudaMemcpy(h_add.data(), d_add, bytes, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(h_saxpy.data(), d_saxpy, bytes, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(h_relu.data(), d_relu, bytes, cudaMemcpyDeviceToHost));
        check(cudaEventRecord(marks[5]));
        check(cudaEventSynchronize(marks[5]));
        check(cudaGetLastError());
        if (!record) {
            return;
        }

        float h2d = 0.f;
        float kernelAdd = 0.f;
        float kernelSaxpy = 0.f;
        float kernelRelu = 0.f;
        float d2h = 0.f;
        float full = 0.f;
        check(cudaEventElapsedTime(&h2d, marks[0], marks[1]));
        check(cudaEventElapsedTime(&kernelAdd, marks[1], marks[2]));
        check(cudaEventElapsedTime(&kernelSaxpy, marks[2], marks[3]));
        check(cudaEventElapsedTime(&kernelRelu, marks[3], marks[4]));
        check(cudaEventElapsedTime(&d2h, marks[4], marks[5]));
        check(cudaEventElapsedTime(&full, marks[0], marks[5]));
        h2dSamples.push_back(h2d);
        addSamples.push_back(kernelAdd);
        saxpySamples.push_back(kernelSaxpy);
        reluSamples.push_back(kernelRelu);
        kernelSamples.push_back(kernelAdd + kernelSaxpy + kernelRelu);
        d2hSamples.push_back(d2h);
        fullSamples.push_back(full);
    };

    measureOnce(false);
    for (int i = 0; i < repeats; ++i) {
        measureOnce(true);
    }

    timings.h2dMs = medianOf(h2dSamples);
    timings.kernelAddMs = medianOf(addSamples);
    timings.kernelSaxpyMs = medianOf(saxpySamples);
    timings.kernelReluMs = medianOf(reluSamples);
    timings.kernelMs = medianOf(kernelSamples);
    timings.d2hMs = medianOf(d2hSamples);
    timings.fullMs = medianOf(fullSamples);

    for (int i = 0; i < n; ++i) {
        const size_t index = static_cast<size_t>(i);
        const float refAdd = h_x[index] + h_y[index];
        const float refSaxpy = alpha * h_x[index] + h_y[index];
        const float refRelu = h_x[index] > 0.f ? h_x[index] : 0.f;
        timings.maxErrAdd = std::max(timings.maxErrAdd, std::fabs(refAdd - h_add[index]));
        timings.maxErrSaxpy = std::max(timings.maxErrSaxpy, std::fabs(refSaxpy - h_saxpy[index]));
        timings.maxErrRelu = std::max(timings.maxErrRelu, std::fabs(refRelu - h_relu[index]));
    }

    release();
}

void writeOutput(const std::string &path, const std::vector<float> &h_add,
                 const std::vector<float> &h_saxpy, const std::vector<float> &h_relu) {
    std::ofstream out(path);
    if (!out) {
        fail(1, "не удалось открыть файл результата: " + path);
    }

    out << std::setprecision(9);
    for (size_t i = 0; i < h_add.size(); ++i) {
        out << h_add[i] << ' ' << h_saxpy[i] << ' ' << h_relu[i] << '\n';
    }
}

void writeCsv(const std::string &path, const std::string &mode, int n, int block,
              const Timings &timings) {
    bool writeHeader = true;
    {
        std::ifstream existing(path);
        if (existing) {
            existing.seekg(0, std::ios::end);
            writeHeader = existing.tellg() == 0;
        }
    }

    std::ofstream out(path, std::ios::app);
    if (!out) {
        fail(1, "не удалось открыть CSV: " + path);
    }
    out << std::setprecision(9);
    if (writeHeader) {
        out << "mode,n,block,alloc_ms,h2d_ms,kernel_add_ms,kernel_saxpy_ms,kernel_relu_ms,"
               "kernel_ms,d2h_ms,full_ms,max_err_add,max_err_saxpy,max_err_relu\n";
    }
    out << mode << ',' << n << ',' << block << ',' << timings.allocMs << ',' << timings.h2dMs << ','
        << timings.kernelAddMs << ',' << timings.kernelSaxpyMs << ',' << timings.kernelReluMs << ','
        << timings.kernelMs << ',' << timings.d2hMs << ',' << timings.fullMs << ','
        << timings.maxErrAdd << ',' << timings.maxErrSaxpy << ',' << timings.maxErrRelu << '\n';
}

int main(int argc, char **argv) {
    Args args = parseArgs(argc, argv);

    int n = 0;
    float alpha = 0.f;
    std::vector<float> h_x;
    std::vector<float> h_y;
    std::vector<float> h_add;
    std::vector<float> h_saxpy;
    std::vector<float> h_relu;

    if (args.synthetic) {
        n = args.syntheticN;
        alpha = args.alpha;
        fillSynthetic(n, h_x, h_y);
    } else {
        readInput(args.input, n, alpha, h_x, h_y);
    }

    Timings timings;
    run(args.mode, h_x, h_y, alpha, args.block, args.repeats, h_add, h_saxpy, h_relu, timings);
    writeOutput(args.output, h_add, h_saxpy, h_relu);
    if (!args.csv.empty()) {
        writeCsv(args.csv, args.mode, n, args.block, timings);
    }
    return 0;
}
