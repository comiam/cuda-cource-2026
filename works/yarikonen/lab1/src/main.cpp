#include "gpu.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <string>
#include <string_view>

namespace {

struct Options {
    std::string input;
    std::string output;
    bool noncoalesced = false;
    unsigned block = 256;
    unsigned repeat = 0;
};

bool read_positive(std::string_view arg, unsigned &value) {
    const auto end = arg.data() + arg.size();
    const auto result = std::from_chars(arg.data(), end, value);
    return result.ec == std::errc{} && result.ptr == end && value > 0;
}

bool parse_options(int argc, char **argv, Options &options) {
    if (argc % 2 == 0) {
        return false;
    }
    for (int i = 1; i < argc; i += 2) {
        const std::string_view key = argv[i];
        const std::string_view value = argv[i + 1];
        if (key == "--input") {
            options.input = value;
        } else if (key == "--output") {
            options.output = value;
        } else if (key == "--mode" && (value == "coalesced" || value == "noncoalesced")) {
            options.noncoalesced = value == "noncoalesced";
        } else if (key == "--block-size") {
            if (!read_positive(value, options.block) || options.block > 1024) {
                return false;
            }
        } else if (key == "--repeat") {
            if (!read_positive(value, options.repeat)) {
                return false;
            }
        } else {
            return false;
        }
    }
    return !options.input.empty() && !options.output.empty();
}

bool read_float(std::istream &file, std::string &token, float &value) {
    if (!(file >> token)) {
        return false;
    }
    char *end = nullptr;
    value = std::strtof(token.c_str(), &end);
    return end == token.c_str() + token.size() && std::isfinite(value);
}

bool read_input(const std::string &path, Input &input) {
    std::ifstream file(path);
    file.imbue(std::locale::classic());
    std::string count;
    if (!(file >> count)) {
        return false;
    }
    auto digits = std::string_view(count);
    if (digits.front() == '+') {
        digits.remove_prefix(1);
    }
    int n = 0;
    const auto end = digits.data() + digits.size();
    const auto parsed = std::from_chars(digits.data(), end, n);
    std::string token;
    if (parsed.ec != std::errc{} || parsed.ptr != end || n < 0 ||
        !read_float(file, token, input.alpha)) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        float x, y;
        if (!read_float(file, token, x) || !read_float(file, token, y)) {
            return false;
        }
        input.hX.push_back(x);
        input.hY.push_back(y);
    }
    file >> std::ws;
    return !file.bad() && file.eof();
}

bool write_output(const std::string &path, const Output &out) {
    std::ofstream file(path);
    file.imbue(std::locale::classic());
    file << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i = 0; i < out[0].size() && file; ++i) {
        file << out[0][i] << ' ' << out[1][i] << ' ' << out[2][i] << '\n';
    }
    file.close();
    return bool(file);
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout
            << "Usage: submission --input PATH --output PATH [--mode MODE] "
               "[--block-size N] [--repeat N]\n"
               "Modes: coalesced (default), noncoalesced. Block size: 1..1024 (default 256).\n"
               "--repeat N: one warmup, N timed cycles, and median timings on stdout.\n"
               "Input: N alpha followed by N pairs x y. Output: N rows of add SAXPY ReLU.\n"
               "N=0 writes an empty output file.\n";
        return 0;
    }
    try {
        Options options;
        if (!parse_options(argc, argv, options)) {
            std::cerr << "Invalid arguments\n";
            return 2;
        }
        Input input{};
        if (!read_input(options.input, input)) {
            std::cerr << "Invalid input: " << options.input << '\n';
            return 2;
        }
        const auto out = compute(input, options.noncoalesced, options.block, options.repeat);
        if (!write_output(options.output, out)) {
            std::cerr << "Cannot write output: " << options.output << '\n';
            return 1;
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
