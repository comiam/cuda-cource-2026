cmake_minimum_required(VERSION 3.20)
project(course_ci_smoke LANGUAGES CXX CUDA)
add_executable(submission src/main.cu)
set_target_properties(submission PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
