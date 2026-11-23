cmake_minimum_required(VERSION 3.20)
project(lab1 LANGUAGES CXX CUDA)

add_executable(submission src/main.cu src/kernels.cu)
target_include_directories(submission PRIVATE include)
set_target_properties(submission PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
