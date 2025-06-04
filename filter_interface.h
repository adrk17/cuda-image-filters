#ifndef FILTER_INTERFACE_H
#define FILTER_INTERFACE_H

#include <opencv2/core.hpp>

#include "filter_types.h"
#include "kernel_launcher.h"
#include "kernel_launcher_utils.h"

#include "cpu_timer.h"
#include "cuda_timer.h"

cv::Mat applyFilterCpu(const cv::Mat& input, FilterType type, const FilterParams& params);
cv::Mat applyFilterGpu(const cv::Mat& input, FilterType type, const FilterParams& params);

#endif
