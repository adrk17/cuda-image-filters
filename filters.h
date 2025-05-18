#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

enum class FilterType {
    GAUSSIAN_BLUR,
    EROSION,
    DILATION,
    OPENING,
    CLOSING
};

struct FilterParams {
    // GaussianBlur
    cv::Size kernelSize = { 5, 5 };
    double sigma = 1.5;

	// Morphology - Erosion, Dilation, Opening, Closing
    int morphShape = cv::MORPH_RECT;
    cv::Size morphKernelSize = { 3, 3 };
};

cv::Mat applyFilterCpu(const cv::Mat& input, FilterType type, const FilterParams& params);
cv::Mat applyFilterGpu(const cv::Mat& input, FilterType type, const FilterParams& params);

#endif