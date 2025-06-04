#ifndef GAUSSIAN_CUH
#define GAUSSIAN_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/hal/interface.h>

#include "cuda_utils.cuh"


__global__ void gaussianBlurXKernel(const uchar* input, float* temp, int rows, int cols, const float* gaussianKernel, int kWidth);
__global__ void gaussianBlurYKernel(const float* temp, uchar* output, int rows, int cols, const float* gaussianKernel, int kWidth);

#endif