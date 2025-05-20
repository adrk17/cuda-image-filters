#ifndef FILTER_INTERFACE_H
#define FILTER_INTERFACE_H

#include "filter_common.h"
#include <opencv2/core.hpp>

cv::Mat applyFilterCpu(const cv::Mat& input, FilterType type, const FilterParams& params);
cv::Mat applyFilterGpu(const cv::Mat& input, FilterType type, const FilterParams& params);

#endif
