#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <filesystem>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "structures.h"
#include "postprocess.h"

#define CUDA_CHECK(call)                                                                                  \
    do                                                                                                    \
    {                                                                                                     \
        cudaError_t error = call;                                                                         \
        if (error != cudaSuccess)                                                                         \
        {                                                                                                 \
            fprintf(stderr, "cuda error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            throw std::runtime_error(cudaGetErrorString(error));                                          \
        }                                                                                                 \
    } while (0)

using namespace nvinfer1;
namespace fs = std::filesystem;

// вспомогательная функция для получения размера элемента по типу данных tensorrt
static size_t getElementSize(DataType dtype)
{
    switch (dtype)
    {
    case DataType::kFLOAT:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT8:
        return 1;
    case DataType::kINT32:
        return 4;
    case DataType::kBOOL:
        return 1;
    case DataType::kUINT8:
        return 1;
    case DataType::kFP8:
        return 1;
    case DataType::kBF16:
        return 2;
    case DataType::kINT64:
        return 8;
    case DataType::kINT4:
        return 1; // упрощенно
    default:
        return 4; // fallback to float
    }
}

static const char *dataTypeToString(DataType dtype)
{
    switch (dtype)
    {
    case DataType::kFLOAT:
        return "FP32";
    case DataType::kHALF:
        return "FP16";
    case DataType::kINT8:
        return "INT8";
    case DataType::kINT32:
        return "INT32";
    case DataType::kBOOL:
        return "BOOL";
    case DataType::kUINT8:
        return "UINT8";
    case DataType::kFP8:
        return "FP8";
    case DataType::kBF16:
        return "BF16";
    case DataType::kINT64:
        return "INT64";
    case DataType::kINT4:
        return "INT4";
    default:
        return "UNKNOWN";
    }
}

class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

// вспомогательная функция для очистки и загрузки меток
static inline std::string trim_copy(const std::string &s)
{
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b]))
        ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1]))
        --e;
    return s.substr(b, e - b);
}

static std::vector<std::string> load_labels_txt(const std::string &path)
{
    std::ifstream f(path);
    if (!f.good())
        return {};
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line))
    {
        line = trim_copy(line);
        if (line.empty())
            continue;
        if (line[0] == '#')
            continue;
        out.push_back(line);
    }
    return out;
}

static std::string format_label(int classId, int labelOffset, float conf, const std::vector<std::string> &classNames)
{
    int idx = classId + labelOffset;
    std::string name;
    if (idx >= 0 && idx < (int)classNames.size())
    {
        name = classNames[idx];
    }
    else
    {
        name = std::to_string(classId);
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s %.2f", name.c_str(), conf);
    return std::string(buf);
}

// функция для отрисовки детекций на изображении
static void draw_detections(cv::Mat &img, const std::vector<Detection> &dets, const std::vector<std::string> &classNames, int labelOffset)
{
    // палитра (bgr). первый цвет — синий, остальные дают разный цвет по классу
    const std::vector<cv::Scalar> palette = {
        cv::Scalar(255, 0, 0),   // blue
        cv::Scalar(255, 128, 0), // light blue
        cv::Scalar(255, 0, 128), // purple-ish
        cv::Scalar(255, 255, 0), // cyan
        cv::Scalar(0, 128, 255), // orange
        cv::Scalar(0, 255, 255), // yellow
        cv::Scalar(0, 255, 0),   // green
        cv::Scalar(0, 0, 255),   // red
        cv::Scalar(128, 0, 255), // magenta
        cv::Scalar(255, 0, 255), // pink
    };
    for (const auto &d : dets)
    {
        int x1 = (int)std::round(d.bbox[0]);
        int y1 = (int)std::round(d.bbox[1]);
        int x2 = (int)std::round(d.bbox[2]);
        int y2 = (int)std::round(d.bbox[3]);
        x1 = std::max(0, std::min(x1, img.cols - 1));
        y1 = std::max(0, std::min(y1, img.rows - 1));
        x2 = std::max(0, std::min(x2, img.cols - 1));
        y2 = std::max(0, std::min(y2, img.rows - 1));
        // детерминированный выбор цвета по отображаемому индексу (с учетом labeloffset)
        int cid = d.class_id + labelOffset;
        if (cid < 0)
            cid = 0;
        int idx = cid % (int)palette.size();
        cv::Scalar color = palette[idx];
        const int thickness = 3;
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);
        std::string label = format_label(d.class_id, labelOffset, d.confidence, classNames);
        int baseLine = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
        int tx = x1;
        int ty = std::max(0, y1 - textSize.height - baseLine - 3);
        cv::Rect bgRect(tx, ty, textSize.width + 6, textSize.height + baseLine + 6);
        bgRect.width = std::min(bgRect.width, img.cols - bgRect.x);
        bgRect.height = std::min(bgRect.height, img.rows - bgRect.y);
        cv::rectangle(img, bgRect, color, cv::FILLED);
        cv::putText(img, label, cv::Point(tx + 3, ty + textSize.height + 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
}

// класс детектора
class RetinaNetDetector
{
public:
    // конструктор
    RetinaNetDetector(const std::string &enginePath) : stream(nullptr)
    {
        initLibNvInferPlugins(static_cast<void *>(&gLogger), "");
        loadEngine(enginePath);
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    // деструктор
    ~RetinaNetDetector()
    {
        if (stream)
            cudaStreamDestroy(stream);
        for (void *buf : buffers)
        {
            if (buf)
                cudaFree(buf);
        }
    }

    std::vector<Detection> detect(const cv::Mat &img, float confThreshold = 0.5f)
    {
        // предобработка
        float *hostInputBuffer = nullptr;
        preprocess(img, hostInputBuffer);
        // инференс
        if (inputDtype == DataType::kFLOAT)
        {
            // fp32: прямое копирование
            CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], hostInputBuffer, inputSize, cudaMemcpyHostToDevice, stream));
        }
        else if (inputDtype == DataType::kHALF)
        {
            // fp16: конвертируем на cpu, затем копируем
            std::vector<__half> fp16Buffer(inputNumElements);
            for (size_t i = 0; i < inputNumElements; ++i)
            {
                fp16Buffer[i] = __float2half(hostInputBuffer[i]);
            }
            CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], fp16Buffer.data(), inputSize, cudaMemcpyHostToDevice, stream));
            cudaStreamSynchronize(stream);
        }
        else if (inputDtype == DataType::kINT8)
        {
            std::vector<int8_t> int8Buffer(inputNumElements);
            for (size_t i = 0; i < inputNumElements; ++i)
            {
                float val = hostInputBuffer[i] * 127.0f;
                val = std::max(-127.0f, std::min(127.0f, val));
                int8Buffer[i] = static_cast<int8_t>(val);
            }
            CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], int8Buffer.data(), inputSize, cudaMemcpyHostToDevice, stream));
            cudaStreamSynchronize(stream);
            std::cerr << "[warn] int8 input: using naive quantization (scale=127). "
                      << "for best results, use engine with fp32 input." << std::endl;
        }
        else
        {
            // fallback
            std::cerr << "[warn] unsupported input dtype, copying as fp32" << std::endl;
            CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], hostInputBuffer, inputNumElements * sizeof(float), cudaMemcpyHostToDevice, stream));
        }
        context->enqueueV3(stream);
        // постобработка
        std::vector<Detection> detections;
        // если движок предоставляет выходы nms плагина, numdetsindex будет установлен.
        // в противном случае, мы ожидаем необработанные выходы: cls_logits + bbox_regression.
        if (numDetsIndex >= 0)
        {
            // синхронизация перед чтением результатов
            cudaStreamSynchronize(stream);
            int numDetections = 0;
            CUDA_CHECK(cudaMemcpyAsync(&numDetections, buffers[numDetsIndex], sizeof(int), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream); // нужно знать numdetections для выделения векторов
            std::vector<float> boxes(numDetections * 4);
            std::vector<float> scores(numDetections);
            std::vector<float> classes(numDetections);
            CUDA_CHECK(cudaMemcpyAsync(boxes.data(), buffers[boxesIndex], numDetections * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(scores.data(), buffers[scoresIndex], numDetections * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(classes.data(), buffers[classesIndex], numDetections * sizeof(float), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);
            for (int i = 0; i < numDetections; ++i)
            {
                if (scores[i] >= confThreshold)
                {
                    Detection d;
                    d.bbox[0] = boxes[i * 4 + 0];
                    d.bbox[1] = boxes[i * 4 + 1];
                    d.bbox[2] = boxes[i * 4 + 2];
                    d.bbox[3] = boxes[i * 4 + 3];
                    d.confidence = scores[i];
                    d.class_id = static_cast<int>(classes[i]);
                    detections.push_back(d);
                }
            }
        }
        else
        {
            // здесь предполагаем boxesindex->cls_logits, scoresindex->bbox_regression
            // получаем numanchors/numclasses из размеров выходных тензоров
            if (clsTensorName.empty() || bboxTensorName.empty())
            {
                throw std::runtime_error("raw outputs not found (expected tensors: cls_logits and bbox_regression)");
            }
            auto clsDims = engine->getTensorShape(clsTensorName.c_str());
            auto boxDims = engine->getTensorShape(bboxTensorName.c_str());
            auto parseAnchorsClasses = [](const nvinfer1::Dims &d, int &outAnchors, int &outClasses)
            {
                if (d.nbDims == 3)
                { // [n, a, c]
                    outAnchors = d.d[1];
                    outClasses = d.d[2];
                }
                else if (d.nbDims == 2)
                { // [a, c]
                    outAnchors = d.d[0];
                    outClasses = d.d[1];
                }
                else
                {
                    outAnchors = -1;
                    outClasses = -1;
                }
            };
            int numAnchors = -1;
            int numClasses = -1;
            parseAnchorsClasses(clsDims, numAnchors, numClasses);
            if (numAnchors <= 0 || numClasses <= 0)
            {
                throw std::runtime_error("unexpected cls_logits dims (expected [1,a,c] or [a,c])");
            }

            // gpu постпроцесс: decode + фильтр + сортировка + nms (topk)
            // используем стандартную пост-обработку для обеих моделей
            const int maxCandidates = 6000;
            const int topK = 1500;

            detections = retinanet_postprocess_gpu(
                (const float *)buffers[boxesIndex],
                (const float *)buffers[scoresIndex],
                numAnchors, numClasses,
                inputW, inputH,
                img.cols, img.rows,
                confThreshold,
                0.5f,
                maxCandidates,
                topK,
                stream); // передаем поток детектора для синхронизации

            // если нет детекций, просто продолжаем с пустым вектором
            if (detections.empty())
            {
                std::cout << "[info] no detections found for current frame with confidence threshold " << confThreshold << std::endl;
            }
        }
        delete[] hostInputBuffer;
        return detections;
    }

    cv::Size getInputSize() const { return cv::Size(inputW, inputH); }
    int getRawNumClasses() const { return rawNumClasses; }

private:
    std::shared_ptr<ICudaEngine> engine;
    std::shared_ptr<IExecutionContext> context;
    std::vector<void *> buffers;
    cudaStream_t stream; // повторно используемый cuda поток для инференса
    int inputIndex, numDetsIndex, boxesIndex, scoresIndex, classesIndex;
    size_t inputSize;        // размер входного буфера
    size_t inputNumElements; // количество элементов входа (c*h*w)
    int inputH, inputW, inputC;
    DataType inputDtype = DataType::kFLOAT;
    std::string clsTensorName;
    std::string bboxTensorName;
    int rawNumClasses = -1;

    void loadEngine(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.good())
        {
            throw std::runtime_error("error reading engine file: " + path);
        }
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> trtModelStream(size);
        file.read(trtModelStream.data(), size);
        file.close();
        std::unique_ptr<IRuntime, void (*)(IRuntime *)> runtime{createInferRuntime(gLogger), [](IRuntime *r)
                                                                { delete r; }};
        // trt 10: deserializecudaengine возвращает icudaengine* который мы удаляем вручную
        engine = std::shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream.data(), size), [](ICudaEngine *e)
                                              { delete e; });
        if (!engine)
            throw std::runtime_error("failed to deserialize engine");
        // trt 10: createexecutioncontext возвращает iexecutioncontext*
        context = std::shared_ptr<IExecutionContext>(engine->createExecutionContext(), [](IExecutionContext *c)
                                                     { delete c; });
        // настройка буферов с использованием имен тензоров
        int numTensors = engine->getNbIOTensors();
        buffers.resize(numTensors);
        inputIndex = -1;
        numDetsIndex = -1;
        boxesIndex = -1;
        scoresIndex = -1;
        classesIndex = -1;
        for (int i = 0; i < numTensors; ++i)
        {
            const char *name = engine->getIOTensorName(i);
            TensorIOMode mode = engine->getTensorIOMode(name);
            if (mode == TensorIOMode::kINPUT)
            {
                inputIndex = i;
                auto dims = engine->getTensorShape(name);
                // nchw
                inputH = dims.d[2];
                inputW = dims.d[3];
                inputC = dims.d[1];
                inputNumElements = (size_t)inputC * inputH * inputW;
                // учитываем реальный тип данных тензора
                inputDtype = engine->getTensorDataType(name);
                size_t elemSize = getElementSize(inputDtype);
                inputSize = inputNumElements * elemSize;
                std::cout << "[trt] input tensor '" << name << "' dtype: " << dataTypeToString(inputDtype)
                          << ", shape: [1," << inputC << "," << inputH << "," << inputW << "]" << std::endl;
                cudaMalloc(&buffers[i], inputSize);
                context->setTensorAddress(name, buffers[i]);
            }
            else
            {
                // маппинг выходов
                std::string sName(name);
                if (sName.find("num_detections") != std::string::npos)
                    numDetsIndex = i;
                else if (sName.find("nmsed_boxes") != std::string::npos)
                    boxesIndex = i;
                else if (sName.find("nmsed_scores") != std::string::npos)
                    scoresIndex = i;
                else if (sName.find("nmsed_classes") != std::string::npos)
                    classesIndex = i;
                else if (sName == "cls_logits")
                { // raw head output
                    boxesIndex = i;
                    clsTensorName = name;
                    // пытаемся получить raw numclasses из статической формы тензора
                    auto d = engine->getTensorShape(name);
                    if (d.nbDims == 3)
                        rawNumClasses = d.d[2]; // [n,a,c]
                    else if (d.nbDims == 2)
                        rawNumClasses = d.d[1]; // [a,c]
                }
                else if (sName == "bbox_regression")
                { // raw head output
                    scoresIndex = i;
                    bboxTensorName = name;
                }
                // выделение памяти в зависимости от формы и реального типа данных
                // для динамических размеров используем максимальные из профиля оптимизации
                auto dims = engine->getTensorShape(name);
                DataType dtype = engine->getTensorDataType(name);
                size_t elemSize = getElementSize(dtype);
                size_t vol = 1;
                bool hasDynamic = false;
                for (int d = 0; d < dims.nbDims; ++d)
                {
                    if (dims.d[d] == -1)
                    {
                        hasDynamic = true;
                        break;
                    }
                }
                if (hasDynamic)
                {
                    // получаем максимальные размеры из профиля оптимизации
                    int profileIdx = context->getOptimizationProfile();
                    auto maxDims = engine->getProfileShape(name, profileIdx, OptProfileSelector::kMAX);
                    for (int d = 0; d < maxDims.nbDims; ++d)
                    {
                        vol *= (maxDims.d[d] > 0) ? maxDims.d[d] : 1;
                    }
                }
                else
                {
                    for (int d = 0; d < dims.nbDims; ++d)
                    {
                        vol *= dims.d[d];
                    }
                }
                cudaMalloc(&buffers[i], vol * elemSize);
                context->setTensorAddress(name, buffers[i]);
                std::cout << "[trt] output tensor '" << name << "' dtype: " << dataTypeToString(dtype) << std::endl;
            }
        }
    }

    void preprocess(const cv::Mat &img, float *&hostBuffer)
    {
        // изменение размера
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(inputW, inputH));
        // приведение к float и нормализация как в torchvision detection models:
        // x = (x/255 - mean) / std, в порядке rgb
        cv::Mat floatImg;
        resized.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);
        // opencv загружает bgr; torchvision ожидает rgb
        cv::cvtColor(floatImg, floatImg, cv::COLOR_BGR2RGB);
        const cv::Scalar mean(0.485, 0.456, 0.406);
        const cv::Scalar stdv(0.229, 0.224, 0.225);
        cv::subtract(floatImg, mean, floatImg);
        cv::divide(floatImg, stdv, floatImg);
        // hwc -> chw
        hostBuffer = new float[inputSize / sizeof(float)];
        // извлечение каналов
        std::vector<cv::Mat> channels(3);
        for (int i = 0; i < 3; ++i)
        {
            channels[i] = cv::Mat(inputH, inputW, CV_32FC1, hostBuffer + i * inputH * inputW);
        }
        cv::split(floatImg, channels);
    }
};

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " <engine_file> <video_or_image_file> [output_path] [confidence_threshold]" << std::endl;
        std::cerr << "example: " << argv[0] << " models/engine.trt input.mp4 output.mp4 0.5" << std::endl;
        return 1;
    }
    std::string enginePath = argv[1];
    std::string inputPath = argv[2];
    std::string outputPath = (argc >= 4) ? argv[3] : "";
    float confThreshold = (argc >= 5) ? std::stof(argv[4]) : 0.1f; // по умолчанию
    try
    {
        RetinaNetDetector detector(enginePath);

        std::vector<std::string> labels;
        int labelOffset = 0; // 0: classid==index; -1: classid в [1..n] (фон на 0)
        if (const char *env = std::getenv("RETINANET_LABELS"))
        {
            auto tmp = load_labels_txt(env);
            if (!tmp.empty())
            {
                labels = std::move(tmp);
                std::cout << "loaded labels from retinanet_labels: " << env << std::endl;
            }
            else
            {
                throw std::runtime_error(std::string("retinanet_labels is set but file could not be read or is empty: ") + env);
            }
        }
        else
        {
            std::filesystem::path p(enginePath);
            std::filesystem::path candidate = p.has_parent_path() ? (p.parent_path() / "labels.txt") : std::filesystem::path("labels.txt");
            if (std::filesystem::exists(candidate))
            {
                auto tmp = load_labels_txt(candidate.string());
                if (!tmp.empty())
                {
                    labels = std::move(tmp);
                    std::cout << "loaded labels from: " << candidate.string() << std::endl;
                }
            }
        }
        if (labels.empty())
        {
            throw std::runtime_error(
                "labels.txt is required. put labels.txt next to the .engine file "
                "or set retinanet_labels to a valid labels.txt path.");
        }
        int nc = detector.getRawNumClasses();
        if (nc > 0)
        {
            if (nc == (int)labels.size())
            {
                labelOffset = 0;
            }
            else if (nc == (int)labels.size() + 1)
            {
                labelOffset = -1;
                std::cout << "[info] using labeloffset=-1 (skip background): numclasses=" << nc
                          << ", labels=" << labels.size() << std::endl;
            }
            else
            {
                throw std::runtime_error(
                    "labels mismatch: numclasses(" + std::to_string(nc) +
                    ") must equal labels.size(" + std::to_string(labels.size()) +
                    ") or labels.size()+1 (background).");
            }
        }
        else
        {
            throw std::runtime_error("could not read raw numclasses from engine at load time (cls_logits shape unknown).");
        }
        cv::Mat image = cv::imread(inputPath);
        if (!image.empty())
        {
            if (outputPath.empty())
                outputPath = "output.png";
            auto start = std::chrono::high_resolution_clock::now();
            auto dets = detector.detect(image, confThreshold);
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            draw_detections(image, dets, labels, labelOffset);
            cv::putText(image, "inference: " + std::to_string(ms) + " ms",
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            if (!cv::imwrite(outputPath, image))
            {
                std::cerr << "failed to write output image: " << outputPath << std::endl;
                return 1;
            }
            std::cout << "done! result saved to " << outputPath << std::endl;
            cleanup_postprocess_stream();
            return 0;
        }
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened())
        {
            std::cerr << "error opening input file (not an image/video?): " << inputPath << std::endl;
            return 1;
        }
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
        if (fps <= 0)
            fps = 30;
        if (outputPath.empty())
            outputPath = "output.mp4";
        cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
        cv::Mat frame;
        int frameCount = 0;
        // переменные для метрик
        double totalInferenceTime = 0.0;
        long totalDetections = 0;
        std::vector<Detection> lastDetections;
        auto totalStart = std::chrono::high_resolution_clock::now();
        while (cap.read(frame))
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Detection> dets = detector.detect(frame, confThreshold);
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            // накопление метрик
            totalInferenceTime += ms;
            totalDetections += dets.size();
            lastDetections = dets;
            draw_detections(frame, dets, labels, labelOffset);
            cv::putText(frame, "inference: " + std::to_string(ms) + " ms",
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            writer.write(frame);
            if (++frameCount % 10 == 0)
            {
                std::cout << "processed frame " << frameCount << std::endl;
            }
        }
        auto totalEnd = std::chrono::high_resolution_clock::now();
        double totalSeconds = std::chrono::duration<double>(totalEnd - totalStart).count();
        cap.release();
        writer.release();
        std::cout << "video: " << inputPath << " (" << width << "x" << height << ", " << fps << " fps)" << std::endl;
        std::cout << "total processing time: " << totalSeconds << " sec" << std::endl;
        std::cout << "average inference time: " << totalInferenceTime / frameCount << " ms/frame" << std::endl;
        std::cout << "detected objects: " << totalDetections << std::endl;
        std::cout << "result: " << outputPath << std::endl;
        std::cout << "\nexample detections (from last frame):" << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, lastDetections.size()); ++i)
        {
            const auto &d = lastDetections[i];
            std::string className;
            int idx = d.class_id + labelOffset;
            if (idx >= 0 && idx < (int)labels.size())
            {
                className = labels[idx];
            }
            else
            {
                className = std::to_string(d.class_id);
            }
            std::cout << "- " << className << " (conf: " << d.confidence << ") ["
                      << (int)d.bbox[0] << ", " << (int)d.bbox[1] << ", "
                      << (int)d.bbox[2] << ", " << (int)d.bbox[3] << "]" << std::endl;
        }
        cleanup_postprocess_stream();
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << std::endl;
        cleanup_postprocess_stream();
        return 1;
    }
    cleanup_postprocess_stream();
    return 0;
}