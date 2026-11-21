/**
 * @file config.h
 * @brief CLI argument parsing and text configuration loading.
 */

#ifndef LAB1_CONFIG_H
#define LAB1_CONFIG_H

#include <string>

#include "kernels.h"

namespace lab1 {

/** Program settings read from the command line and the input file. */
struct Config {
    std::string inputPath;
    std::string outputPath;
    AccessMode mode = AccessMode::Coalesced;

    std::size_t n = 1u << 20; // benchmark size
    float alpha = 2.0f;
    unsigned seed = 2026u;
    int blockSize = 256; // 0 means sweep several block sizes
    int repeats = 20;
};

/** Print usage help to stdout. */
void printUsage(const char *program);

/** Parse command-line arguments into a Config. */
Config parseArgs(int argc, char **argv);

/** Read key=value settings from the input file into a Config. */
void readConfigFile(const std::string &path, Config &cfg);

} // namespace lab1

#endif // LAB1_CONFIG_H
