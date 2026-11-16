// CUDA-ядра Lab 1: сложение векторов, SAXPY и ReLU.
//
// Каждый поток обрабатывает ровно один элемент массива (паттерн Map).
// Для каждой операции есть два варианта обращения к памяти:
// - coalesced:   соседние потоки одного warp читают и пишут соседние элементы;
// - noncoalesced: та же логическая операция, но соседние потоки обращаются
//   к элементам с шагом gridDim.x (транспонированная схема индексации).
// Оба варианта дают идентичный результат и обрабатывают одинаковое число
// элементов, различается только организация обращений к global memory.

// Коалесцированный вариант: i = blockIdx.x * blockDim.x + threadIdx.x.
__global__ void add_coalesced(const float *__restrict__ a, const float *__restrict__ b,
                              float *__restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// Некоалесцированный вариант: поток (b, t) обрабатывает элемент t * grid + b,
// поэтому потоки одного warp (соседние t) обращаются к элементам на расстоянии
// gridDim.x друг от друга. Схема биективна: все индексы 0..n-1 покрываются
// ровно по одному разу (лишние индексы отбрасываются проверкой j < n).
__global__ void add_noncoalesced(const float *__restrict__ a, const float *__restrict__ b,
                                 float *__restrict__ out, int n) {
    int j = threadIdx.x * gridDim.x + blockIdx.x;
    if (j < n) {
        out[j] = a[j] + b[j];
    }
}

// SAXPY: out[i] = alpha * x[i] + y[i].
// Отдельный выходной буфер вместо in-place: многократные прогоны должны
// каждый раз получать одинаковые входы x и y.
__global__ void saxpy_coalesced(const float *__restrict__ x, const float *__restrict__ y,
                                float *__restrict__ out, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = alpha * x[i] + y[i];
    }
}

__global__ void saxpy_noncoalesced(const float *__restrict__ x, const float *__restrict__ y,
                                   float *__restrict__ out, float alpha, int n) {
    int j = threadIdx.x * gridDim.x + blockIdx.x;
    if (j < n) {
        out[j] = alpha * x[j] + y[j];
    }
}

// ReLU: out[i] = max(0, x[i]).
__global__ void relu_coalesced(const float *__restrict__ x, float *__restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fmaxf(x[i], 0.0f);
    }
}

__global__ void relu_noncoalesced(const float *__restrict__ x, float *__restrict__ out, int n) {
    int j = threadIdx.x * gridDim.x + blockIdx.x;
    if (j < n) {
        out[j] = fmaxf(x[j], 0.0f);
    }
}
