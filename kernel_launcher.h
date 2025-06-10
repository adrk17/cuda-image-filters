#ifndef KERNEL_LAUNCHER_H
#define KERNEL_LAUNCHER_H

#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#include "filter_types.h"
#include "kernel_launcher_utils.h"

/**
 * Launches a custom CUDA Gaussian blur operation on a grayscale image.
 * This function uploads a 1D Gaussian kernel to constant memory, allocates temporary device memory,
 * and runs two CUDA kernels to perform separable convolution in the X and Y directions.
 * The result is written to the output device buffer. Optionally measures and returns execution time.
 *
 * @param d_input    Device input image (grayscale).
 * @param d_output   Device output image.
 * @param rows       Number of rows in the image.
 * @param cols       Number of columns in the image.
 * @param kernelWidth Width of the Gaussian kernel (must be odd).
 * @param sigma      Standard deviation of the Gaussian.
 * @param grid       CUDA grid dimensions.
 * @param block      CUDA block dimensions.
 * @param elapsedMs  Optional pointer to store elapsed time in ms.
 * @return           CUDA error code.
 */
cudaError_t launchGaussianBlur(const uchar* d_input, uchar* d_output, int rows, int cols, int kernelWidth, float sigma, dim3 grid, dim3 block, float* elapsedMs);

/**
 * Launches a custom CUDA erosion operation on a grayscale image.
 * This function uploads the morphological structuring element to constant memory,
 * allocates shared memory for each block, and runs a CUDA kernel that computes the minimum value
 * under the structuring element for each pixel. Optionally measures and returns execution time.
 *
 * @param d_input    Device input image (grayscale).
 * @param d_output   Device output image.
 * @param rows       Number of rows in the image.
 * @param cols       Number of columns in the image.
 * @param kernelSize Size of the morphological kernel (e.g., cv::Size(3, 3)).
 * @param morphShape Shape of the morphological operation (e.g., cv::MORPH_RECT).
 * @param grid       CUDA grid dimensions.
 * @param block      CUDA block dimensions.
 * @param elapsedMs  Optional pointer to store elapsed time in ms.
 * @return           CUDA error code.
 */
cudaError_t launchErosion(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs);

/**
 * Launches a custom CUDA dilation operation on a grayscale image.
 * This function uploads the morphological structuring element to constant memory,
 * allocates shared memory for each block, and runs a CUDA kernel that computes the maximum value
 * under the structuring element for each pixel. Optionally measures and returns execution time.
 *
 * @param d_input    Device input image (grayscale).
 * @param d_output   Device output image.
 * @param rows       Number of rows in the image.
 * @param cols       Number of columns in the image.
 * @param kernelSize Size of the morphological kernel (e.g., cv::Size(3, 3)).
 * @param morphShape Shape of the morphological operation (e.g., cv::MORPH_RECT).
 * @param grid       CUDA grid dimensions.
 * @param block      CUDA block dimensions.
 * @param elapsedMs  Optional pointer to store elapsed time in ms.
 * @return           CUDA error code.
 */
cudaError_t launchDilation(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs);

/**
 * Launches a custom CUDA opening operation (erosion followed by dilation) on a grayscale image.
 * This function uploads the structuring element to constant memory, allocates temporary device memory,
 * and sequentially runs the erosion and dilation CUDA kernels. Optionally measures and returns execution time.
 *
 * @param d_input    Device input image (grayscale).
 * @param d_output   Device output image.
 * @param rows       Number of rows in the image.
 * @param cols       Number of columns in the image.
 * @param kernelSize Size of the morphological kernel (e.g., cv::Size(3, 3)).
 * @param morphShape Shape of the morphological operation (e.g., cv::MORPH_RECT).
 * @param grid       CUDA grid dimensions.
 * @param block      CUDA block dimensions.
 * @param elapsedMs  Optional pointer to store elapsed time in ms.
 * @return           CUDA error code.
 */
cudaError_t launchOpening(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs);

/**
 * Launches a custom CUDA closing operation (dilation followed by erosion) on a grayscale image.
 * This function uploads the structuring element to constant memory, allocates temporary device memory,
 * and sequentially runs the dilation and erosion CUDA kernels. Optionally measures and returns execution time.
 *
 * @param d_input    Device input image (grayscale).
 * @param d_output   Device output image.
 * @param rows       Number of rows in the image.
 * @param cols       Number of columns in the image.
 * @param kernelSize Size of the morphological kernel (e.g., cv::Size(3, 3)).
 * @param morphShape Shape of the morphological operation (e.g., cv::MORPH_RECT).
 * @param grid       CUDA grid dimensions.
 * @param block      CUDA block dimensions.
 * @param elapsedMs  Optional pointer to store elapsed time in ms.
 * @return           CUDA error code.
 */
cudaError_t launchClosing(const uchar* d_input, uchar* d_output, int rows, int cols, cv::Size kernelSize, int morphShape, dim3 grid, dim3 block, float* elapsedMs);

#endif
