/**
 * @file report.h
 * @brief Result structures and report generation for the benchmark.
 */

#ifndef LAB1_REPORT_H
#define LAB1_REPORT_H

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "config.h"
#include "kernels.h"

namespace lab1 {

/** Result of validating one operation on a small size. */
struct ValidationResult {
    std::string opName;
    std::size_t n;
    float maxErr;
    std::size_t mismatches;
};

/** Measured times for one benchmark iteration. */
struct Timing {
    float h2dMs = 0.0f;
    float kernelMs = 0.0f;
    float d2hMs = 0.0f;
    float fullMs = 0.0f;
};

/** Aggregated result of benchmarking one operation with one block size. */
struct BenchmarkResult {
    std::string opName;
    int blockSize;
    Timing median;
    float allocMs = 0.0f;
};

/** Print the validation table to the given stream. */
void printValidationTable(std::ostream &os, const std::vector<ValidationResult> &results);

/** Print the benchmark timing table to the given stream. */
void printBenchmarkTable(std::ostream &os, const std::vector<BenchmarkResult> &results,
                         AccessMode mode);

/** Write the full report to the configured output directory. */
void writeReport(const Config &cfg, const std::vector<ValidationResult> &validation,
                 const std::vector<BenchmarkResult> &benchmarks);

} // namespace lab1

#endif // LAB1_REPORT_H
