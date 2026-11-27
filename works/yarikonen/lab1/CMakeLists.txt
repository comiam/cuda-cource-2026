cmake_minimum_required(VERSION 3.18)
project(lab1 LANGUAGES CXX CUDA)

add_executable(submission src/main.cpp src/gpu.cu src/kernels.cu)
target_compile_features(submission PRIVATE cxx_std_17)
set_target_properties(
  submission
  PROPERTIES CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>")
