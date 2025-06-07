#include "kernel_launcher_utils.h"

bool isMorphologicalFilter(FilterType type) {
    return type == FilterType::EROSION ||
        type == FilterType::DILATION ||
        type == FilterType::OPENING ||
        type == FilterType::CLOSING;
}

float* generateGaussianKernel1D(int size, float sigma) {
    if (size < 3 || size % 2 == 0) {
        throw std::invalid_argument("Gaussian kernel size must be an odd number >= 3");
    }

    int radius = size / 2;
    float* kernel = (float*)malloc(size * sizeof(float));
    if (!kernel) {
        throw std::runtime_error("Failed to allocate memory for Gaussian kernel");
    }

    float sum = 0.0f;
    float sigma2 = 2.0f * sigma * sigma;

    for (int i = -radius; i <= radius; ++i) {
        float value = std::exp(-(i * i) / sigma2);
        kernel[i + radius] = value;
        sum += value;
    }

    for (int i = 0; i < size; ++i) {
        kernel[i] /= sum;
    }

    return kernel;
}