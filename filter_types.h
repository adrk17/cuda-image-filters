#ifndef FILTER_TYPES_H
#define FILTER_TYPES_H

#include <opencv2/opencv.hpp>

/**
 * Enum class representing different filter types for image processing.
 */
enum class FilterType {
    GAUSSIAN_BLUR,
    EROSION,
    DILATION,
    OPENING,
    CLOSING
};

/**
 * Struct containing parameters for various filters.
 * These parameters can be adjusted based on the filter type.
 */
struct FilterParams {
    // GaussianBlur
    int kernelWidth = 5;
    float sigma = 1.5;

	// Morphology - Erosion, Dilation, Opening, Closing
    int morphShape = cv::MORPH_RECT;
    cv::Size morphKernelSize = { 3, 3 };
};


#endif