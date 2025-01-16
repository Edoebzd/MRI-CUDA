cmake_minimum_required(VERSION 3.29)
#project(MRI-CUDA LANGUAGES C CXX CUDA)
project(MRI-CUDA LANGUAGES C CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 14)

include_directories(src)
include_directories(src/fft)

add_executable(MRI-CUDA
        src/main.cpp
        src/utils.cpp
        src/utils.h
        src/stb_image_write.h
        src/fft/fft-cuda.cu
        src/fft/fft-cuda.cuh
)

set_target_properties(MRI-CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

#find_package(CUDAToolkit REQUIRED)
#include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(ISMRMRD REQUIRED)
link_directories(${ISMRMRD_LIBRARY_DIRS})
include_directories(${ISMRMRD_INCLUDE_DIRS})
target_link_libraries(MRI-CUDA ${ISMRMRD_LIBRARIES})