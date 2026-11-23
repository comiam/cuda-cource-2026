cmake_minimum_required(VERSION 3.20)
project(lab1 LANGUAGES CXX CUDA)

add_executable(submission src/main.cu src/kernels.cu)
target_include_directories(submission PRIVATE include)
set_target_properties(submission PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)

# Замеры времени (tests/bench.cu). Для сдачи нужна только цель submission.
add_executable(bench tests/bench.cu src/kernels.cu)
target_include_directories(bench PRIVATE include)
set_target_properties(bench PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)

# Программа проверки корректности (tests/verify.cu). Для сдачи нужна только цель submission.
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  add_executable(verify tests/verify.cu src/kernels.cu)
  target_include_directories(verify PRIVATE include)
  target_link_libraries(verify PRIVATE CUDA::cublas)
  set_target_properties(verify PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
endif()
