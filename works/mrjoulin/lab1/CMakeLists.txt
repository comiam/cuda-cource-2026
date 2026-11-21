cmake_minimum_required(VERSION 3.20)
project(lab1 LANGUAGES CXX CUDA)

add_executable(submission src/main.cu src/kernels.cu src/config.cpp src/report.cpp)

target_include_directories(submission PRIVATE include)

set_target_properties(
  submission
  PROPERTIES CXX_STANDARD 17
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             CXX_STANDARD_REQUIRED ON)

# Target Turing (Quadro RTX 6000 / compute capability 7.5) by default. Adjust if the target GPU
# differs.
set_property(TARGET submission PROPERTY CUDA_ARCHITECTURES 75)

# Reasonable release flags; keep debug info line-table-only for profiling.
target_compile_options(submission PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                                          $<$<CONFIG:Release>:-O3> $<$<CONFIG:Debug>:-G>)
