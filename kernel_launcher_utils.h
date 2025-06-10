#ifndef KERNEL_LAUNCHER_UTILS_H
#define KERNEL_LAUNCHER_UTILS_H

#include <opencv2/core.hpp>
#include "cuda_utils.cuh"
#include "filter_types.h"

/**
 * @brief Generates a 1D Gaussian kernel.
 * This function creates a 1D Gaussian kernel of the specified size and standard deviation.
 * @param size Size of the kernel (must be odd).
 * @param sigma Standard deviation of the Gaussian distribution.
 * @return Pointer to the generated Gaussian kernel.
 */
float* generateGaussianKernel1D(int size, float sigma);

/**
 * Checks if the filter type is a morphological filter.
 */
bool isMorphologicalFilter(FilterType type);

#endif