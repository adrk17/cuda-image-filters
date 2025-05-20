#ifndef FILTER_CUDA_UTILS_H
#define FILTER_CUDA_UTILS_H

#include <opencv2/core.hpp>
#include "cuda_utils.h"
#include "filter_common.h"

uchar* prepareMorphMask(const FilterParams& params);
float* generateGaussianKernel1D(int size, float sigma);
bool isMorphologicalFilter(FilterType type);

#endif