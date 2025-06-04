#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/core/hal/interface.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " - " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define BLOCK_SIZE 16
#define MAX_KERNEL_WIDTH 64 // Maximum kernel width for __constant__ memory allocation
#define MAX_KERNEL_AREA (MAX_KERNEL_WIDTH * MAX_KERNEL_WIDTH)


__device__ __forceinline__ int clamp(int value, int minVal, int maxVal) {
    return (value < minVal) ? minVal : (value > maxVal) ? maxVal : value;
}


#endif

