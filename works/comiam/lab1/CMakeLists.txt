cmake_minimum_required(VERSION 3.20)
project(submission_smoke LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(submission src/main.cu)
