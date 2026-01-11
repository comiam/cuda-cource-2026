cmake_minimum_required(VERSION 3.18)
project(retinanet_tensorrt LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PLUGINS nvinfer_plugin
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY_INFER OR NOT TENSORRT_LIBRARY_PLUGINS)
    message(FATAL_ERROR "Cannot find TensorRT libraries")
endif()

include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/postprocess.cu
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75)

target_link_libraries(${PROJECT_NAME} 
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_PLUGINS}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    cudart
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -O3>
)