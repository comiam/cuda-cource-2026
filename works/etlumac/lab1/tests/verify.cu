// Проверка корректности: наши ядра (оба режима) сравниваются с готовыми функциями.
//   add:   thrust::transform + thrust::plus
//   saxpy: cublasSaxpy
//   relu:  thrust::transform + thrust::maximum (максимум из x и 0)
#include "kernels.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// Заполняет x и y числами одного из трёх видов.
// kind 0: случайные числа от -10 до 10; kind 1: нули (+0 и -0); kind 2: чередование знаков.
void fill_data(int kind, int n, std::vector<float> &x, std::vector<float> &y) {
    std::mt19937 gen(12345 + n); // seed генератора случайных чисел
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    x.resize(n);
    y.resize(n);
    for (int i = 0; i < n; ++i) {
        if (kind == 0) {
            x[i] = dist(gen);
            y[i] = dist(gen);
        } else if (kind == 1) {
            x[i] = (i % 2 == 0) ? 0.0f : -0.0f;
            y[i] = (i % 3 == 0) ? 0.0f : -0.0f;
        } else {
            x[i] = (i % 2 == 0) ? static_cast<float>(i % 100) : -static_cast<float>(i % 100);
            y[i] = (i % 3 == 0) ? static_cast<float>(i % 50) : -static_cast<float>(i % 50);
        }
    }
}

// Один тест: считает нашими ядрами и эталоном, обновляет максимальные ошибки.
// Для add и relu результат должен совпасть точно (+0 равен -0).
// Для saxpy допускаем разницу в округлении (FMA или отдельные умножение и сложение):
//   |наш - эталон| <= 4 * FLT_EPSILON * (|alpha * x| + |y|)
bool run_test(cublasHandle_t handle, int kind, int n, float alpha, bool strided, double &err_add,
              double &err_saxpy, double &err_relu) {
    std::vector<float> x, y;
    fill_data(kind, n, x, y);

    // Наши ядра
    std::vector<float> my_add, my_saxpy, my_relu;
    compute_on_gpu(alpha, x, y, strided, my_add, my_saxpy, my_relu);

    // Эталон на GPU
    thrust::device_vector<float> d_x(x.begin(), x.end());
    thrust::device_vector<float> d_y(y.begin(), y.end());
    thrust::device_vector<float> d_add(n), d_relu(n);
    thrust::device_vector<float> d_saxpy(y.begin(), y.end()); // копия исходного y: saxpy его меняет

    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_add.begin(), thrust::plus<float>());
    thrust::transform(d_x.begin(), d_x.end(), thrust::constant_iterator<float>(0.0f),
                      d_relu.begin(), thrust::maximum<float>());
    cublasStatus_t status = cublasSaxpy(handle, n, &alpha, thrust::raw_pointer_cast(d_x.data()), 1,
                                        thrust::raw_pointer_cast(d_saxpy.data()), 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasSaxpy failed, status %d\n", static_cast<int>(status));
        return false;
    }

    std::vector<float> ref_add(n), ref_saxpy(n), ref_relu(n);
    thrust::copy(d_add.begin(), d_add.end(), ref_add.begin());
    thrust::copy(d_saxpy.begin(), d_saxpy.end(), ref_saxpy.begin());
    thrust::copy(d_relu.begin(), d_relu.end(), ref_relu.begin());

    // Сравнение
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        double e_add = std::fabs(static_cast<double>(my_add[i]) - ref_add[i]);
        double e_saxpy = std::fabs(static_cast<double>(my_saxpy[i]) - ref_saxpy[i]);
        double e_relu = std::fabs(static_cast<double>(my_relu[i]) - ref_relu[i]);
        double tolerance = 4.0 * FLT_EPSILON * (std::fabs(alpha * x[i]) + std::fabs(y[i]));
        if (e_add > err_add)
            err_add = e_add;
        if (e_saxpy > err_saxpy)
            err_saxpy = e_saxpy;
        if (e_relu > err_relu)
            err_relu = e_relu;
        if (e_add != 0.0 || e_relu != 0.0 || e_saxpy > tolerance) {
            ok = false;
        }
    }
    return ok;
}

int main() {
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }

    const int sizes[] = {1, 2, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4097, 100000, 1000003};
    const float alpha = 0.75f;
    bool all_ok = true;

    std::printf("alpha = %.2f, tolerance for saxpy: 4 * FLT_EPSILON * (|alpha*x| + |y|)\n", alpha);
    std::printf("%-8s %-13s %-12s %-12s %-12s %s\n", "N", "mode", "max_err_add", "max_err_saxpy",
                "max_err_relu", "result");
    for (int n : sizes) {
        for (int strided = 0; strided < 2; ++strided) {
            double err_add = 0.0, err_saxpy = 0.0, err_relu = 0.0;
            bool ok = true;
            for (int kind = 0; kind < 3; ++kind) { // три вида данных
                if (!run_test(handle, kind, n, alpha, strided == 1, err_add, err_saxpy, err_relu)) {
                    ok = false;
                }
            }
            std::printf("%-8d %-13s %-12.3g %-12.3g %-12.3g %s\n", n,
                        strided ? "noncoalesced" : "coalesced", err_add, err_saxpy, err_relu,
                        ok ? "OK" : "FAIL");
            all_ok = all_ok && ok;
        }
    }
    cublasDestroy(handle);
    std::printf(all_ok ? "ALL OK\n" : "SOME TESTS FAILED\n");
    return all_ok ? 0 : 1;
}