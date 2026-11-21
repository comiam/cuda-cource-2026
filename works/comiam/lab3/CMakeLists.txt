cmake_minimum_required(VERSION 3.18)
project(lab3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the upstream codec unmodified; restore its original filenames at build time.
configure_file(vendor/lodepng.header "${CMAKE_CURRENT_BINARY_DIR}/lodepng.h" COPYONLY)
configure_file(vendor/lodepng.source "${CMAKE_CURRENT_BINARY_DIR}/lodepng.cpp" COPYONLY)
add_executable(submission src/main.cu "${CMAKE_CURRENT_BINARY_DIR}/lodepng.cpp")
target_include_directories(submission PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
