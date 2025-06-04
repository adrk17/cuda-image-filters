#ifndef KERNEL_LAUNCHER_UTILS_H
#define KERNEL_LAUNCHER_UTILS_H

#include <opencv2/core.hpp>
#include "cuda_utils.cuh"
#include "filter_types.h"

uchar* prepareMorphMask(const FilterParams& params);
float* generateGaussianKernel1D(int size, float sigma);
bool isMorphologicalFilter(FilterType type);

#endif