/**
 * @file report.cpp
 * @brief Report generation implementation.
 */

#include "report.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace lab1 {

namespace {

std::string modeString(AccessMode mode) {
    return mode == AccessMode::Coalesced ? "coalesced" : "noncoalesced";
}

} // namespace

void printValidationTable(std::ostream &os, const std::vector<ValidationResult> &results) {
    os << "\n### Validation (small sizes)\n\n";
    os << "| Operation | Size | Max abs error | Mismatches (>1e-5) |\n";
    os << "|-----------|------|---------------|--------------------|\n";

    for (const auto &r : results) {
        os << "| " << std::setw(9) << std::left << r.opName << " | " << std::setw(4) << r.n << " | "
           << std::scientific << std::setprecision(2) << r.maxErr << " | " << r.mismatches
           << " |\n";
    }

    os << std::defaultfloat << std::setprecision(6);
}

void printBenchmarkTable(std::ostream &os, const std::vector<BenchmarkResult> &results,
                         AccessMode mode) {
    os << "\n### Benchmark: " << modeString(mode) << " access\n\n";
    os << "| Operation | Block | H2D, ms | Kernel, ms | D2H, ms | Full cycle, "
          "ms | Alloc, ms |\n";
    os << "|-----------|-------|---------|------------|---------|-------------"
          "--|-----------|\n";

    for (const auto &r : results) {
        os << "| " << std::setw(9) << std::left << r.opName << " | " << std::setw(5) << r.blockSize
           << " | " << std::fixed << std::setprecision(3) << r.median.h2dMs << " | "
           << r.median.kernelMs << " | " << r.median.d2hMs << " | " << r.median.fullMs << " | "
           << r.allocMs << " |\n";
    }

    os << std::defaultfloat << std::setprecision(6);
}

void writeReport(const Config &cfg, const std::vector<ValidationResult> &validation,
                 const std::vector<BenchmarkResult> &benchmarks) {
    if (std::system(("mkdir -p '" + cfg.outputPath + "'").c_str()) != 0) {
        std::cerr << "Failed to create output directory: " << cfg.outputPath << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::string reportPath = cfg.outputPath + "/report.txt";
    std::ofstream report(reportPath);
    if (!report) {
        std::cerr << "Failed to open report file: " << reportPath << "\n";
        std::exit(EXIT_FAILURE);
    }

    report << "Lab 1: Vector Operations and Activations\n";
    report << "Mode: " << modeString(cfg.mode) << "\n";
    report << "Benchmark size: " << cfg.n << " elements ("
           << (cfg.n * sizeof(float) / (1024.0 * 1024.0)) << " MiB)\n";
    report << "SAXPY alpha: " << cfg.alpha << "\n";
    report << "RNG seed: " << cfg.seed << "\n";
    report << "Repeats (after warmup): " << cfg.repeats << "\n";
    report << "Timers: CUDA events, median reported\n";

    printValidationTable(report, validation);
    printBenchmarkTable(report, benchmarks, cfg.mode);

    report.close();

    std::cout << "Report written to: " << reportPath << "\n";
    std::ifstream echo(reportPath);
    std::cout << echo.rdbuf();
}

} // namespace lab1
