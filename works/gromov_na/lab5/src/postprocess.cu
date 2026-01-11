#include "postprocess.h"

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <vector>

#define CUDA_CHECK(call)                                                                                  \
    do                                                                                                    \
    {                                                                                                     \
        cudaError_t error = call;                                                                         \
        if (error != cudaSuccess)                                                                         \
        {                                                                                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            throw std::runtime_error(cudaGetErrorString(error));                                          \
        }                                                                                                 \
    } while (0)

struct Anchor
{
    float x1, y1, x2, y2;
};

// cортировка
struct CandidateScoreGreater
{
    __host__ __device__ bool operator()(const Candidate &a, const Candidate &b) const
    {
        return a.score > b.score;
    }
};

// cигмоида
__device__ __forceinline__ float sigmoidf_dev(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

// вычисление IoU
__device__ __forceinline__ float iou_dev(const Candidate &a, const Candidate &b)
{
    float xx1 = fmaxf(a.x1, b.x1);
    float yy1 = fmaxf(a.y1, b.y1);
    float xx2 = fminf(a.x2, b.x2);
    float yy2 = fminf(a.y2, b.y2);

    float w = fmaxf(0.0f, xx2 - xx1);
    float h = fmaxf(0.0f, yy2 - yy1);
    float inter = w * h;

    float areaA = fmaxf(0.0f, a.x2 - a.x1) * fmaxf(0.0f, a.y2 - a.y1);
    float areaB = fmaxf(0.0f, b.x2 - b.x1) * fmaxf(0.0f, b.y2 - b.y1);
    float union_area = areaA + areaB - inter;

    return union_area > 0.0f ? (inter / union_area) : 0.0f;
}

// константные данные для анкоров
__constant__ int c_level_offsets[6]; // префиксные суммы в анкорах. offsets[0]=0, offsets[5]=totalAnchors
__constant__ int c_featW[5];
__constant__ int c_featH[5];
__constant__ int c_stride[5];
__constant__ float c_baseSize[5];

// обновление метаданных анкоров в зависимости от размера входа
static void update_anchor_meta_if_needed(int inputW, int inputH)
{
    // кэшируем последний обработанный размер, чтобы избежать повторных копирований
    static int lastW = -1;
    static int lastH = -1;
    if (inputW == lastW && inputH == lastH)
        return;

    const int stride_h[5] = {8, 16, 32, 64, 128};
    const float baseSize_h[5] = {32.f, 64.f, 128.f, 256.f, 512.f};

    int featW_h[5];
    int featH_h[5];
    int offsets_h[6];
    offsets_h[0] = 0;

    for (int l = 0; l < 5; ++l)
    {
        int s = stride_h[l];
        featW_h[l] = (inputW + s - 1) / s; // ceil
        featH_h[l] = (inputH + s - 1) / s; // ceil
        int anchorsL = featW_h[l] * featH_h[l] * 9;
        offsets_h[l + 1] = offsets_h[l] + anchorsL;
    }

    // копирование
    CUDA_CHECK(cudaMemcpyToSymbol(c_stride, stride_h, sizeof(c_stride)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_baseSize, baseSize_h, sizeof(c_baseSize)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_featW, featW_h, sizeof(c_featW)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_featH, featH_h, sizeof(c_featH)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_level_offsets, offsets_h, sizeof(c_level_offsets)));

    lastW = inputW;
    lastH = inputH;
}

// вычисление анкора по индексу
__device__ __forceinline__ Anchor anchor_from_index_dynamic(int i)
{
    // поиск уровня по смещениям
    int level = 0;
#pragma unroll
    for (int l = 0; l < 5; ++l)
    {
        if (i < c_level_offsets[l + 1])
        {
            level = l;
            break;
        }
    }

    int local = i - c_level_offsets[level];
    int fw = c_featW[level];
    int stride = c_stride[level];
    float size = c_baseSize[level];

    // cellIdx * 9 + (scaleIdx*3 + ratioIdx)
    int cell = local / 9;
    int rem = local - cell * 9;
    int scaleIdx = rem / 3;
    int ratioIdx = rem - scaleIdx * 3;

    int cy = cell / fw;
    int cx = cell - cy * fw;

    float ctr_x = (cx + 0.5f) * stride;
    float ctr_y = (cy + 0.5f) * stride;

    const float scales[3] = {1.0f, 1.2599210499f, 1.5874010520f};
    const float ratios[3] = {0.5f, 1.0f, 2.0f};

    float base = size * scales[scaleIdx];
    float area = base * base;
    float ar = ratios[ratioIdx];
    float aw = sqrtf(area / ar);
    float ah = aw * ar;

    Anchor a;
    a.x1 = ctr_x - 0.5f * aw;
    a.y1 = ctr_y - 0.5f * ah;
    a.x2 = ctr_x + 0.5f * aw;
    a.y2 = ctr_y + 0.5f * ah;
    return a;
}

// cuda ядро для декодирования и фильтрации
__global__ void k_decode_and_filter(
    const float *cls_logits,  // [A*C]
    const float *bbox_deltas, // [A*4]
    int numAnchors,
    int numClasses,
    float confTh,
    float scaleX,
    float scaleY,
    int origW,
    int origH,
    Candidate *out,
    int maxOut,
    int *outCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAnchors)
        return;

    // находим класс с большей уверенностью
    float bestS = 0.0f;
    int bestC = -1;
    const float *logits = cls_logits + i * numClasses;
#pragma unroll 4
    for (int c = 0; c < numClasses; ++c)
    {
        float s = sigmoidf_dev(logits[c]);
        if (s > bestS)
        {
            bestS = s;
            bestC = c;
        }
    }
    if (bestS < confTh)
        return;

    // декодируем
    Anchor a = anchor_from_index_dynamic(i);
    float ax = 0.5f * (a.x1 + a.x2);
    float ay = 0.5f * (a.y1 + a.y2);
    float aw = (a.x2 - a.x1);
    float ah = (a.y2 - a.y1);

    float dx = bbox_deltas[i * 4 + 0];
    float dy = bbox_deltas[i * 4 + 1];
    float dw = bbox_deltas[i * 4 + 2];
    float dh = bbox_deltas[i * 4 + 3];

    float px = dx * aw + ax;
    float py = dy * ah + ay;
    float pw = __expf(dw) * aw;
    float ph = __expf(dh) * ah;

    float x1 = (px - 0.5f * pw) * scaleX;
    float y1 = (py - 0.5f * ph) * scaleY;
    float x2 = (px + 0.5f * pw) * scaleX;
    float y2 = (py + 0.5f * ph) * scaleY;

    // обрезка
    x1 = fminf(fmaxf(x1, 0.0f), (float)(origW - 1));
    y1 = fminf(fmaxf(y1, 0.0f), (float)(origH - 1));
    x2 = fminf(fmaxf(x2, 0.0f), (float)(origW - 1));
    y2 = fminf(fmaxf(y2, 0.0f), (float)(origH - 1));

    if ((x2 - x1) <= 1.0f || (y2 - y1) <= 1.0f)
        return;

    int idx = atomicAdd(outCount, 1);
    if (idx >= maxOut)
    {
        // если лимит превышен, уменьшаем счетчик обратно
        atomicAdd(outCount, -1);
        return;
    }
    out[idx] = Candidate{x1, y1, x2, y2, bestS, bestC};
}

// инициализация массива
__global__ void k_init_int(int *arr, int n, int val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        arr[i] = val;
}

// ядро для подавления nms
__global__ void k_nms_suppress(
    const Candidate *cand,
    int n,
    float iouTh,
    int *suppressed)
{
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (j >= n)
        return;
    if (j <= i)
        return;
    if (suppressed[i])
        return;
    if (suppressed[j])
        return;
    float iou = iou_dev(cand[i], cand[j]);
    if (iou > iouTh)
    {
        // cand отсортирован по убыванию уверенности, поэтому i имеет >= оценку, чем j
        atomicExch(&suppressed[j], 1);
    }
}

// глобальный поток для асинхронных операций
static cudaStream_t g_stream = nullptr;

// инициализация глобального потока
static void init_global_stream()
{
    if (g_stream == nullptr)
    {
        CUDA_CHECK(cudaStreamCreate(&g_stream));
    }
}

// основная функция постобработки
std::vector<Detection> retinanet_postprocess_gpu(
    const float *d_cls_logits,
    const float *d_bbox_deltas,
    int numAnchors,
    int numClasses,
    int inputW,
    int inputH,
    int origW,
    int origH,
    float confThreshold,
    float nmsThreshold,
    int maxCandidates,
    int topK,
    cudaStream_t stream)
{
    // назначаем поток
    cudaStream_t use_stream = (stream != nullptr) ? stream : g_stream;
    if (use_stream == nullptr)
    {
        init_global_stream();
        use_stream = g_stream;
    }

    update_anchor_meta_if_needed(inputW, inputH);

    // проверяем количество анкоров для текущего размера входа
    const int strides_h[5] = {8, 16, 32, 64, 128};
    int expected = 0;
    for (int l = 0; l < 5; ++l)
    {
        int s = strides_h[l];
        int fw = (inputW + s - 1) / s;
        int fh = (inputH + s - 1) / s;
        expected += fw * fh * 9;
    }
    if (numAnchors != expected)
    {
        // фолбэк
        return {};
    }

    float scaleX = (float)origW / (float)inputW;
    float scaleY = (float)origH / (float)inputH;

    Candidate *d_cand = nullptr;
    int *d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cand, sizeof(Candidate) * maxCandidates));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), use_stream));

    int block = 256;
    int grid = (numAnchors + block - 1) / block;
    k_decode_and_filter<<<grid, block, 0, use_stream>>>(
        d_cls_logits, d_bbox_deltas,
        numAnchors, numClasses,
        confThreshold,
        scaleX, scaleY,
        origW, origH,
        d_cand, maxCandidates, d_count);

    // синхронизируем поток перед чтением
    CUDA_CHECK(cudaStreamSynchronize(use_stream));

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_count <= 0)
    {
        cudaFree(d_cand);
        cudaFree(d_count);
        return {};
    }
    if (h_count > maxCandidates)
        h_count = maxCandidates;

    // сортируем кандидатов по убыванию уверенности на gpu
    thrust::device_ptr<Candidate> cand_begin(d_cand);
    thrust::device_ptr<Candidate> cand_end(d_cand + h_count);
    thrust::sort(thrust::cuda::par.on(use_stream), cand_begin, cand_end, CandidateScoreGreater{});

    // оставляем только topK для nms, чтобы сохранить O(N^2) разумным
    int n = h_count;
    if (topK > 0 && n > topK)
        n = topK;

    int *d_supp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_supp, sizeof(int) * n));
    k_init_int<<<(n + 255) / 256, 256, 0, use_stream>>>(d_supp, n, 0);

    dim3 grid2(n, (n + 255) / 256);
    k_nms_suppress<<<grid2, 256, 0, use_stream>>>(d_cand, n, nmsThreshold, d_supp);

    // копируем обратно topK кандидатов + флаги подавления
    std::vector<Candidate> h_cand(n);
    std::vector<int> h_supp(n);
    CUDA_CHECK(cudaMemcpyAsync(h_cand.data(), d_cand, sizeof(Candidate) * n, cudaMemcpyDeviceToHost, use_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_supp.data(), d_supp, sizeof(int) * n, cudaMemcpyDeviceToHost, use_stream));

    // синхронизируем перед использованием данных на хосте
    CUDA_CHECK(cudaStreamSynchronize(use_stream));

    std::vector<Detection> out;
    out.reserve(256);
    for (int i = 0; i < n; ++i)
    {
        if (h_supp[i])
            continue;
        Detection d{};
        d.bbox[0] = h_cand[i].x1;
        d.bbox[1] = h_cand[i].y1;
        d.bbox[2] = h_cand[i].x2;
        d.bbox[3] = h_cand[i].y2;
        d.confidence = h_cand[i].score;
        d.class_id = h_cand[i].classId;
        out.push_back(d);
    }

    cudaFree(d_supp);
    cudaFree(d_cand);
    cudaFree(d_count);

    return out;
}

// функция для освобождения глобального потока
extern "C" void cleanup_postprocess_stream()
{
    if (g_stream != nullptr)
    {
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
    }
}