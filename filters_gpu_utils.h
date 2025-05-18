#ifndef FILTERS_GPU_UTILS_H
#define FILTERS_GPU_UTILS_H

#include <opencv2/core.hpp>

#include "cuda_utils.h"
#include "filters.h"

uchar* prepareMorphMask(const FilterParams& params) {
    cv::Mat mask = cv::getStructuringElement(params.morphShape, params.morphKernelSize);
    size_t maskBytes = mask.total();

    uchar* d_mask = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mask, maskBytes));
    CUDA_CHECK(cudaMemcpy(d_mask, mask.data, maskBytes, cudaMemcpyHostToDevice));

    return d_mask;
}

bool isMorphologicalFilter(FilterType type) {
    return type == FilterType::EROSION ||
        type == FilterType::DILATION ||
        type == FilterType::OPENING ||
        type == FilterType::CLOSING;
}

#endif