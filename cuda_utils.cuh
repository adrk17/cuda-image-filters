#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <opencv2/core/hal/interface.h>

/**
 * @brief Check for CUDA errors and print an error message if one occurs.
 * @param call The CUDA function call to check.
 */
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


 /**
  * @brief Clamp a value to a specified range.
  * @param value The value to clamp.
  * @param minVal The minimum value of the range.
  * @param maxVal The maximum value of the range.
  * @return The clamped value.
  */
__device__ __forceinline__ int clamp(int value, int minVal, int maxVal) {
    return (value < minVal) ? minVal : (value > maxVal) ? maxVal : value;
}


#endif

