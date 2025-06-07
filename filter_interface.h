#ifndef FILTER_INTERFACE_H
#define FILTER_INTERFACE_H

// OpenCV - cpu
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// OpenCV - gpu
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#include "filter_types.h"
#include "kernel_launcher.h"
#include "kernel_launcher_utils.h"

#include "cpu_timer.h"
#include "cuda_timer.h"

cv::Mat applyFilterOpenCvCpu(const cv::Mat& input, FilterType type, const FilterParams& params, bool verbose = true);
cv::Mat applyFilterGpu(const cv::Mat& input, FilterType type, const FilterParams& params, bool verbose = true);
cv::Mat applyFilterOpenCvGpu(const cv::Mat& input, FilterType type, const FilterParams& params, bool verbose = true);

cv::Mat applyFilterOpenCvCpuIterations(const cv::Mat& input, FilterType type, const FilterParams& params, int iterations, bool verbose = true, float* avgMs = nullptr);
cv::Mat applyFilterGpuIterations(const cv::Mat& input, FilterType type, const FilterParams& params, int iterations, bool verbose = true, float* avgMs = nullptr);
cv::Mat applyFilterOpenCvGpuIterations(const cv::Mat& input, FilterType type, const FilterParams& params, int iterations, bool verbose = true, float* avgMs = nullptr);

#endif
