#ifndef FILTERS_GPU_H
#define FILTERS_GPU_H


#include <opencv2/core.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void launchGaussianBlur(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, double sigma, dim3 grid, dim3 block);
void launchErosion(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block);
void launchDilation(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block);
void launchOpening(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block);
void launchClosing(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block);

#endif
