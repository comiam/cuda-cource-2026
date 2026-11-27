cmake_minimum_required(VERSION 3.20)

project(lab1 LANGUAGES CXX CUDA)

add_executable(submission src/main.cu)

set_property(TARGET submission PROPERTY CXX_STANDARD 17)
set_property(TARGET submission PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET submission PROPERTY CUDA_STANDARD 17)
set_property(TARGET submission PROPERTY CUDA_STANDARD_REQUIRED ON)
