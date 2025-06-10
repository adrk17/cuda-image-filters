#ifndef FILTER_INTERFACE_H
#define FILTER_INTERFACE_H

// OpenCV - cpu
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// OpenCV - gpu
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#include "filter_types.h"
#include "kernel_launcher.h"
#include "kernel_launcher_utils.h"

#include "cpu_timer.h"
#include "cuda_timer.h"
/**
 * Applies the selected filter to a grayscale image using custom CUDA kernels on the GPU.
 * This function allocates device memory, copies the input image to the GPU, launches the appropriate CUDA kernel
 * (such as Gaussian blur, erosion, dilation, opening, or closing), and copies the result back to the host.
 * It can print kernel execution time if verbose is enabled.
 *
 * @param input      Grayscale input image (CV_8UC1).
 * @param type       FilterType enum specifying which filter to apply.
 * @param params     FilterParams struct containing filter-specific parameters.
 * @param verbose    If true, prints kernel execution time and error messages.
 * @return           Output image after filtering, or a clone of the input on error.
 */
cv::Mat applyFilterGpu(const cv::Mat& input, FilterType type, const FilterParams& params, bool verbose = true);
/**
 * Applies the selected filter to a grayscale image using OpenCV's CPU functions.
 * The function selects and applies the appropriate OpenCV filter (e.g., GaussianBlur, erode, dilate, morphologyEx)
 * based on the filter type and parameters. Prints timing information if verbose is enabled.
 *
 * @param input      Grayscale input image (CV_8UC1).
 * @param type       FilterType enum specifying which filter to apply.
 * @param params     FilterParams struct containing filter-specific parameters.
 * @param verbose    If true, prints filter execution time and error messages.
 * @return           Output image after filtering, or a clone of the input on error.
 */
cv::Mat applyFilterOpenCvCpu(const cv::Mat& input, FilterType type, const FilterParams& params, bool verbose = true);
/**
 * Applies the selected filter to a grayscale image using OpenCV's CUDA-accelerated functions.
 * The image is uploaded to GPU memory, the appropriate OpenCV CUDA filter is created and applied,
 * and the result is downloaded back to the host. Prints timing information if verbose is enabled.
 *
 * @param input      Grayscale input image (CV_8UC1).
 * @param type       FilterType enum specifying which filter to apply.
 * @param params     FilterParams struct containing filter-specific parameters.
 * @param verbose    If true, prints filter execution time and error messages.
 * @return           Output image after filtering, or the input image if the filter type is unsupported.
 */
cv::Mat applyFilterOpenCvGpu(const cv::Mat& input, FilterType type, const FilterParams& params, bool verbose = true);

/**
 * Applies the selected filter using OpenCV's CPU functions for a specified number of iterations.
 * The filter is applied repeatedly, and the average execution time can be measured and returned.
 * Useful for benchmarking or repeated processing.
 *
 * @param input      Grayscale input image (CV_8UC1).
 * @param type       FilterType enum specifying which filter to apply.
 * @param params     FilterParams struct containing filter-specific parameters.
 * @param iterations Number of times to apply the filter.
 * @param verbose    If true, prints timing information for each iteration and the average.
 * @param avgMs      Pointer to store average execution time in milliseconds (can be nullptr).
 * @return           Output image after the last iteration.
 */
cv::Mat applyFilterOpenCvCpuIterations(const cv::Mat& input, FilterType type, const FilterParams& params, int iterations, bool verbose = true, float* avgMs = nullptr);
/**
 * Applies the selected filter using custom CUDA kernels for a specified number of iterations.
 * The filter is applied repeatedly on the GPU, and the average execution time can be measured and returned.
 * Useful for benchmarking or repeated processing.
 *
 * @param input      Grayscale input image (CV_8UC1).
 * @param type       FilterType enum specifying which filter to apply.
 * @param params     FilterParams struct containing filter-specific parameters.
 * @param iterations Number of times to apply the filter.
 * @param verbose    If true, prints timing information for each iteration and the average.
 * @param avgMs      Pointer to store average execution time in milliseconds (can be nullptr).
 * @return           Output image after the last iteration.
 */
cv::Mat applyFilterGpuIterations(const cv::Mat& input, FilterType type, const FilterParams& params, int iterations, bool verbose = true, float* avgMs = nullptr);
/**
 * Applies the selected filter using OpenCV's CUDA-accelerated functions for a specified number of iterations.
 * The filter is applied repeatedly on the GPU, and the average execution time can be measured and returned.
 * Useful for benchmarking or repeated processing.
 *
 * @param input      Grayscale input image (CV_8UC1).
 * @param type       FilterType enum specifying which filter to apply.
 * @param params     FilterParams struct containing filter-specific parameters.
 * @param iterations Number of times to apply the filter.
 * @param verbose    If true, prints timing information for each iteration and the average.
 * @param avgMs      Pointer to store average execution time in milliseconds (can be nullptr).
 * @return           Output image after the last iteration.
 */
cv::Mat applyFilterOpenCvGpuIterations(const cv::Mat& input, FilterType type, const FilterParams& params, int iterations, bool verbose = true, float* avgMs = nullptr);

#endif
