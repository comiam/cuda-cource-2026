// Lab 1: CLI-обвес над ядрами из src/kernels.cu.
//
// Читает «N alpha», затем N пар «x y», гоняет на GPU add (x+y), SAXPY
// (alpha*x+y) и ReLU (max(0,x)) в режиме coalesced/noncoalesced, меряет
// CUDA events H2D / ядра / D2H / полный цикл, сверяет с CPU и пишет в
// --output N строк «x+y  alpha*x+y  max(0,x)» без заголовка.
// Коды возврата: 0 — успех, 1 — ошибка выполнения, 2 — неверные аргументы.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess) {                                                                 \
            std::cerr << "CUDA ошибка: " << cudaGetErrorString(err_) << " (строка " << __LINE__    \
                      << ")\n";                                                                    \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

// Ядра из kernels.cu.
extern __global__ void add_coalesced(const float *a, const float *b, float *out, int n);
extern __global__ void add_noncoalesced(const float *a, const float *b, float *out, int n);
extern __global__ void saxpy_coalesced(const float *x, const float *y, float *out, float alpha,
                                       int n);
extern __global__ void saxpy_noncoalesced(const float *x, const float *y, float *out, float alpha,
                                          int n);
extern __global__ void relu_coalesced(const float *x, float *out, int n);
extern __global__ void relu_noncoalesced(const float *x, float *out, int n);

namespace {

void print_help() {
    std::cout << "Lab 1: сложение векторов, SAXPY и ReLU на GPU.\n"
                 "Использование: ./submission --input <путь> --output <путь> "
                 "[--mode coalesced|noncoalesced] [--alpha f] [--block 1..1024] [--repeat n]\n"
                 "Вход: строка «N alpha», далее N строк «x y» (float). "
                 "Выход: N строк «x+y alpha*x+y max(0,x)», без заголовка.\n"
                 "Режимы доступа: coalesced — поток берёт элемент i = blockIdx*blockDim+threadIdx; "
                 "noncoalesced — j = threadIdx*gridDim+blockIdx (шаг потоков warp равен grid).\n"
                 "По умолчанию: coalesced, block=256, repeat=10; alpha берётся из входа, "
                 "--alpha переопределяет его.\n"
                 "Коды возврата: 0 — успех, 1 — ошибка выполнения, 2 — неверные аргументы.\n";
}

bool read_input(const std::string &path, std::vector<float> &a, std::vector<float> &b,
                float &alpha) {
    std::ifstream in(path);
    long n = 0;
    if (!(in >> n >> alpha) || n < 0 || n > INT32_MAX) {
        std::cerr << "Некорректный входной файл: " << path << " (первая строка: «N alpha»)\n";
        return false;
    }
    a.resize(n);
    b.resize(n);
    for (long i = 0; i < n; ++i) {
        if (!(in >> a[i] >> b[i])) {
            std::cerr << path << ": ожидалось 2n=" << 2 * n << " чисел, обрыв на " << i << "\n";
            return false;
        }
    }
    return true;
}

float median_of(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return v.empty() ? 0.0f : (v.size() % 2 ? v[m] : (v[m - 1] + v[m]) * 0.5f);
}

} // namespace

int main(int argc, char **argv) {
    std::string in_path, out_path, mode = "coalesced";
    float alpha = 2.0f;
    bool alpha_override = false;
    int block = 256, repeat = 10;

    auto val = [&](int &i) -> std::string {
        if (i + 1 >= argc) {
            std::cerr << argv[i] << ": требуется значение\n";
            std::exit(2);
        }
        return argv[++i];
    };
    auto num = [](const std::string &s) {
        char *e = nullptr;
        std::strtod(s.c_str(), &e);
        return e != s.c_str() && *e == '\0';
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--input")
            in_path = val(i);
        else if (a == "--output")
            out_path = val(i);
        else if (a == "--mode")
            mode = val(i);
        else if (a == "--alpha" || a == "--block" || a == "--repeat") {
            std::string v = val(i);
            if (!num(v)) {
                std::cerr << a << ": требуется число\n";
                return 2;
            }
            if (a == "--alpha") {
                alpha = std::strtof(v.c_str(), nullptr);
                alpha_override = true;
            } else if (a == "--block")
                block = std::atoi(v.c_str());
            else
                repeat = std::atoi(v.c_str());
        } else if (a == "--help" || a == "-h") {
            print_help();
            return 0;
        } else {
            std::cerr << "Неизвестный аргумент: " << a << " (см. --help)\n";
            return 2;
        }
    }
    if (mode != "coalesced" && mode != "noncoalesced") {
        std::cerr << "--mode: ожидается coalesced|noncoalesced\n";
        return 2;
    }
    if (in_path.empty() || out_path.empty()) {
        std::cerr << "Требуется --input <путь> --output <путь> (см. --help)\n";
        return 2;
    }
    if (block < 1 || block > 1024 || repeat < 1) {
        std::cerr << "--block: 1..1024, --repeat: >= 1\n";
        return 2;
    }

    std::vector<float> h_a, h_b;
    float file_alpha = 0.0f;
    if (!read_input(in_path, h_a, h_b, file_alpha))
        return 2;
    if (!alpha_override)
        alpha = file_alpha;
    const int n = static_cast<int>(h_a.size());

    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "Не удалось открыть выходной файл: " << out_path << "\n";
        return 1;
    }
    std::vector<float> h_add(n), h_saxpy(n), h_relu(n);
    auto write_results = [&]() {
        out << std::setprecision(9);
        for (int i = 0; i < n; ++i) {
            out << h_add[i] << ' ' << h_saxpy[i] << ' ' << h_relu[i] << '\n';
        }
    };
    if (n == 0) { // Пустой вход: ядра не запускаем.
        write_results();
        std::cout << "n=0: ядра не запускались.\n";
        return 0;
    }

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    // Три буфера GPU: a, b и общий выход d_out — ядра идут последовательно,
    // после каждого результата копируем в свой вектор.
    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t ev[6]; // 0:старт 1:H2D 2:add 3:saxpy 4:relu 5:D2H
    for (cudaEvent_t &e : ev)
        CUDA_CHECK(cudaEventCreate(&e));

    const int grid = (n + block - 1) / block;
    const bool coal = (mode == "coalesced");
    auto launch_add = [&]() -> cudaError_t {
        if (coal)
            add_coalesced<<<grid, block, 0, stream>>>(d_a, d_b, d_out, n);
        else
            add_noncoalesced<<<grid, block, 0, stream>>>(d_a, d_b, d_out, n);
        return cudaGetLastError();
    };
    auto launch_saxpy = [&]() -> cudaError_t {
        if (coal)
            saxpy_coalesced<<<grid, block, 0, stream>>>(d_a, d_b, d_out, alpha, n);
        else
            saxpy_noncoalesced<<<grid, block, 0, stream>>>(d_a, d_b, d_out, alpha, n);
        return cudaGetLastError();
    };
    auto launch_relu = [&]() -> cudaError_t {
        if (coal)
            relu_coalesced<<<grid, block, 0, stream>>>(d_a, d_out, n);
        else
            relu_noncoalesced<<<grid, block, 0, stream>>>(d_a, d_out, n);
        return cudaGetLastError();
    };

    // 1 прогрев + repeat замеров. Интервалы: H2D, add(ядро), saxpy(ядро+коп add),
    // relu(ядро+коп saxpy), D2H(коп relu), полный цикл ev0->ev5.
    std::vector<float> t[6];
    for (int r = 0; r <= repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(ev[0], stream));
        CUDA_CHECK(cudaMemcpyAsync(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(ev[1], stream));
        CUDA_CHECK(launch_add());
        CUDA_CHECK(cudaEventRecord(ev[2], stream));
        CUDA_CHECK(cudaMemcpyAsync(h_add.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(launch_saxpy());
        CUDA_CHECK(cudaEventRecord(ev[3], stream));
        CUDA_CHECK(cudaMemcpyAsync(h_saxpy.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(launch_relu());
        CUDA_CHECK(cudaEventRecord(ev[4], stream));
        CUDA_CHECK(cudaMemcpyAsync(h_relu.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(ev[5], stream));
        CUDA_CHECK(cudaEventSynchronize(ev[5]));
        if (r > 0) {
            for (int k = 0; k < 5; ++k) {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev[k], ev[k + 1]);
                t[k].push_back(ms);
            }
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev[0], ev[5]);
            t[5].push_back(ms);
        }
    }
    for (cudaEvent_t &e : ev)
        CUDA_CHECK(cudaEventDestroy(e));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    // CPU-сверка (эталон: a+b, alpha*a+b, max(a,0)). Допуск atol + rtol*|ref|:
    // GPU-сакс пи FMA, расхождение пропорционально величине.
    double max_err[3] = {0.0, 0.0, 0.0};
    const double tol = 1e-5, rtol = 1e-5;
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        const double ref[3] = {static_cast<double>(h_a[i]) + h_b[i],
                               static_cast<double>(alpha) * h_a[i] + h_b[i],
                               std::max(static_cast<double>(h_a[i]), 0.0)};
        const float *got[3] = {h_add.data(), h_saxpy.data(), h_relu.data()};
        for (int k = 0; k < 3; ++k) {
            double d = std::fabs(got[k][i] - ref[k]);
            max_err[k] = std::max(max_err[k], d);
            if (d > tol + rtol * std::fabs(ref[k]))
                ok = false;
        }
    }

    write_results();
    std::cout << std::fixed << std::setprecision(4) << "mode=" << mode << " n=" << n
              << " alpha=" << alpha << " block=" << block << " grid=" << grid << "\n"
              << "медианы, мс: H2D=" << median_of(t[0]) << " add(ядро)=" << median_of(t[1])
              << " saxpy(ядро+коп)=" << median_of(t[2]) << " relu(ядро+коп)=" << median_of(t[3])
              << " D2H(коп relu)=" << median_of(t[4]) << " полный цикл=" << median_of(t[5]) << "\n"
              << std::scientific << "max|ошибка| add=" << max_err[0] << " saxpy=" << max_err[1]
              << " relu=" << max_err[2] << " (atol 1e-05 + rtol 1e-05*|ref|) — "
              << (ok ? "OK" : "РАСХОЖДЕНИЕ") << "\n"
              << "выделение памяти (3 буфера по " << bytes << " Б) в полный цикл не входит\n";
    return ok ? 0 : 1;
}
