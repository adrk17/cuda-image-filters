#ifndef FILTER_COMMON_H
#define FILTER_COMMON_H

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
    int kernelWidth = 5;
    float sigma = 1.5;

	// Morphology - Erosion, Dilation, Opening, Closing
    int morphShape = cv::MORPH_RECT;
    cv::Size morphKernelSize = { 3, 3 };
};


#endif