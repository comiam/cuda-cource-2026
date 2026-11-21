/**
 * @file config.cpp
 * @brief Implementation of command-line parsing and config file loading.
 */

#include "config.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace lab1 {

namespace {

std::string trim(const std::string &s) {
    size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

AccessMode parseMode(const std::string &s) {
    if (s == "coalesced") {
        return AccessMode::Coalesced;
    }
    if (s == "noncoalesced") {
        return AccessMode::Noncoalesced;
    }
    std::cerr << "Unknown mode: " << s << "\n";
    std::exit(EXIT_FAILURE);
}

} // namespace

void printUsage(const char *program) {
    std::cout << "Usage: " << program << " --input <config.txt> --output <dir> [--mode <mode>]\n"
              << "\n"
              << "Required:\n"
              << "  --input <path>   Text file with key=value settings\n"
              << "  --output <dir>   Directory for the report\n"
              << "\n"
              << "Optional:\n"
              << "  --mode <mode>    coalesced (default) or noncoalesced\n"
              << "\n"
              << "Config keys (one per line):\n"
              << "  n=<size>         Number of elements for the benchmark\n"
              << "  alpha=<float>    SAXPY coefficient\n"
              << "  seed=<uint>      RNG seed for reproducible inputs\n"
              << "  block_size=<int> Threads per block (0 sweeps 64..1024)\n"
              << "  repeats=<int>    Repetitions after warmup\n";
}

Config parseArgs(int argc, char **argv) {
    Config cfg;
    bool hasInput = false;
    bool hasOutput = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }

        if (arg == "--input") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --input\n";
                std::exit(EXIT_FAILURE);
            }
            cfg.inputPath = argv[++i];
            hasInput = true;
            continue;
        }

        if (arg == "--output") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --output\n";
                std::exit(EXIT_FAILURE);
            }
            cfg.outputPath = argv[++i];
            hasOutput = true;
            continue;
        }

        if (arg == "--mode") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --mode\n";
                std::exit(EXIT_FAILURE);
            }
            cfg.mode = parseMode(argv[++i]);
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        printUsage(argv[0]);
        std::exit(EXIT_FAILURE);
    }

    if (!hasInput || !hasOutput) {
        std::cerr << "Both --input and --output are required.\n";
        printUsage(argv[0]);
        std::exit(EXIT_FAILURE);
    }

    return cfg;
}

void readConfigFile(const std::string &path, Config &cfg) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Cannot open input config: " << path << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    int lineno = 0;

    while (std::getline(file, line)) {
        ++lineno;
        line = trim(line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            std::cerr << "Invalid config line " << lineno << ": " << line << "\n";
            std::exit(EXIT_FAILURE);
        }

        std::string key = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));

        try {
            if (key == "n") {
                cfg.n = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "alpha") {
                cfg.alpha = std::stof(value);
            } else if (key == "seed") {
                cfg.seed = static_cast<unsigned>(std::stoul(value));
            } else if (key == "block_size") {
                cfg.blockSize = std::stoi(value);
            } else if (key == "repeats") {
                cfg.repeats = std::stoi(value);
            } else {
                std::cerr << "Unknown config key: " << key << "\n";
                std::exit(EXIT_FAILURE);
            }
        } catch (const std::exception &e) {
            std::cerr << "Bad value for " << key << " at line " << lineno << ": " << e.what()
                      << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    if (cfg.repeats < 1) {
        cfg.repeats = 1;
    }
}

} // namespace lab1
