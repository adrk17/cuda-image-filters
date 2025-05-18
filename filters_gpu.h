#ifndef FILTERS_GPU_H
#define FILTERS_GPU_H


#include <opencv2/core.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t launchGaussianBlur(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, double sigma, dim3 grid, dim3 block);
cudaError_t launchErosion(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block);
cudaError_t launchDilation(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block);
cudaError_t launchOpening(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block);
cudaError_t launchClosing(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block);

#endif
