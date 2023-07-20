cmake_minimum_required(VERSION 2.8)

project (GPUgmatch)

set(CUDA_SEPARABLE_COMPILATION OFF)

find_package(CUDA REQUIRED)
find_package(OpenMP)

cuda_add_library(kernel kernel.cuh)

set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g -Wall -march=native -pthread")

set(CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75 -Xcompiler -fopenmp -lgomp)

get_filename_component(CUDA_LIB_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY)

find_library(CUDA_cudadevrt_LIBRARY cudadevrt PATHS ${CUDA_LIB_PATH})

cuda_add_executable(GPUgmatch main.cu)

target_link_libraries(GPUgmatch kernel ${CUDA_cudadevrt_LIBRARY})

if(OpenMP_CXX_FOUND)
    target_link_libraries(GPUgmatch OpenMP::OpenMP_CXX)
endif()
