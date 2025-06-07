#ifndef KERNEL_LAUNCHER_H
#define KERNEL_LAUNCHER_H

#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#include "filter_types.h"
#include "kernel_launcher_utils.h"

cudaError_t launchGaussianBlur(const uchar* d_input, uchar* d_output, int rows, int cols, int kernelWidth, float sigma, dim3 grid, dim3 block, float* elapsedMs);
cudaError_t launchErosion(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs);
cudaError_t launchDilation(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs);
cudaError_t launchOpening(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs);
cudaError_t launchClosing(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs);

#endif
