cmake_minimum_required(VERSION 3.18)
project(lab2 LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_executable(submission src/main.cu)

option(LAB2_CUBLAS_REFERENCE "Build the separate cuBLAS reference executable" OFF)
if(LAB2_CUBLAS_REFERENCE)
  find_package(CUDAToolkit REQUIRED)
  add_executable(cublas_reference src/main.cu)
  target_compile_definitions(cublas_reference PRIVATE LAB2_CUBLAS_REFERENCE)
  target_link_libraries(cublas_reference PRIVATE CUDA::cublas)
endif()
