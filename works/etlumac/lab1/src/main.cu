#include "kernels.h"

#include <cstdio>
#include <string>
#include <vector>

// Читает файл: "N alpha", затем N пар "x y". Возвращает false, если файл не открылся
// или формат неверный.
bool read_input(const std::string &path, float &alpha, std::vector<float> &x,
                std::vector<float> &y) {
    FILE *f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        std::fprintf(stderr, "Cannot open input file: %s\n", path.c_str());
        return false;
    }
    int n = 0;
    if (std::fscanf(f, "%d %f", &n, &alpha) != 2 || n < 0) {
        std::fprintf(stderr, "Bad input: the first line must be 'N alpha'\n");
        std::fclose(f);
        return false;
    }
    for (int i = 0; i < n; ++i) {
        float a = 0.0f, b = 0.0f;
        if (std::fscanf(f, "%f %f", &a, &b) != 2) {
            std::fprintf(stderr, "Bad input: expected %d pairs 'x y', got %d\n", n, i);
            std::fclose(f);
            return false;
        }
        x.push_back(a);
        y.push_back(b);
    }
    std::fclose(f);
    return true;
}

// Пишет по строке на каждую пару: "x+y alpha*x+y max(0,x)". Девять значащих цифр.
bool write_output(const std::string &path, const std::vector<float> &h_add,
                  const std::vector<float> &h_saxpy, const std::vector<float> &h_relu) {
    FILE *f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        std::fprintf(stderr, "Cannot open output file: %s\n", path.c_str());
        return false;
    }
    for (size_t i = 0; i < h_add.size(); ++i) {
        std::fprintf(f, "%.9g %.9g %.9g\n", h_add[i], h_saxpy[i], h_relu[i]);
    }
    if (std::fclose(f) != 0) {
        std::fprintf(stderr, "Cannot write output file: %s\n", path.c_str());
        return false;
    }
    return true;
}

void print_help() {
    std::printf("Lab 1: vector add, SAXPY and ReLU on GPU\n"
                "\n"
                "Usage:\n"
                "  submission --help\n"
                "  submission --input <file> --output <file> [--mode <mode>]\n"
                "\n"
                "Arguments:\n"
                "  --input <file>   input file: first 'N alpha', then N lines 'x y'\n"
                "  --output <file>  output file: N lines 'x+y alpha*x+y max(0,x)'\n"
                "  --mode <mode>    coalesced (default) or noncoalesced\n"
                "\n"
                "Exit codes: 0 - success, 1 - runtime or write error,\n"
                "2 - bad arguments or bad input.\n"
                "For N = 0 an empty output file is written and no kernel is launched.\n");
}

int main(int argc, char **argv) {
    std::string input_path, output_path, mode = "coalesced";

    // Разбор аргументов командной строки
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_help();
            return 0;
        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (input_path.empty() || output_path.empty()) {
        std::fprintf(stderr, "Both --input and --output are required (see --help)\n");
        return 2;
    }
    if (mode != "coalesced" && mode != "noncoalesced") {
        std::fprintf(stderr, "Unknown mode: %s (use coalesced or noncoalesced)\n", mode.c_str());
        return 2;
    }

    // Чтение входа
    float alpha = 0.0f;
    std::vector<float> h_x, h_y;
    if (!read_input(input_path, alpha, h_x, h_y)) {
        return 2;
    }

    // Вычисление на GPU (при N = 0 ядро не запускаем)
    std::vector<float> h_add, h_saxpy, h_relu;
    if (!h_x.empty()) {
        compute_on_gpu(alpha, h_x, h_y, mode == "noncoalesced", h_add, h_saxpy, h_relu);
    }

    // Запись результата
    if (!write_output(output_path, h_add, h_saxpy, h_relu)) {
        return 1;
    }
    std::printf("Done: N=%zu, mode=%s\n", h_x.size(), mode.c_str());
    return 0;
}