#include "filter_cuda_kernels.h"
#include <device_launch_parameters.h>
#include "filter_cuda_utils.h"


////// GAUSSIAN BLUR //////
__device__ int clamp(int value, int minVal, int maxVal) {
	return (value < minVal) ? minVal : (value > maxVal) ? maxVal : value;
}

__global__ void gaussianBlurKernel(const uchar* input, uchar* output, int rows, int cols, const float* gaussianKernel, int kWidth) {
	/// Define dynamic shared memory
	extern __shared__ float shared[];

	float* kernel1D = shared; // kernel1D is the first part of shared memory
	int radius = kWidth / 2;
	const int tileSize = BLOCK_SIZE + 2 * radius;

	float* tile = (float*)&kernel1D[kWidth];

	int lx = threadIdx.x;
	int ly = threadIdx.y;
	int x = blockIdx.x * BLOCK_SIZE + lx;
	int y = blockIdx.y * BLOCK_SIZE + ly;
	int globalIdx = y * cols + x;

	int tx = lx + radius;
	int ty = ly + radius;

	/// Load gaussian kernel into shared memory
	int threadId = ly * BLOCK_SIZE + lx;
	int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE;

	for (int i = threadId; i < kWidth; i += threadsPerBlock) { // for loop needed if there are more values in the mask than threads in block
		kernel1D[i] = gaussianKernel[i];
	}

	/// Load image data into shared memory.
	/// Threads on the edges load 4 or 2 pixels while the rest load 1 pixel 
	for (int dy = ly; dy < tileSize; dy += BLOCK_SIZE)
	{
		for (int dx = lx; dx < tileSize; dx += BLOCK_SIZE)
		{
			int imgX = blockIdx.x * BLOCK_SIZE + dx - radius;
			int imgY = blockIdx.y * BLOCK_SIZE + dy - radius;

			imgX = clamp(imgX, 0, cols - 1); 
			imgY = clamp(imgY, 0, rows - 1);

			tile[dy * tileSize + dx] = static_cast<float>(input[imgY * cols + imgX]);
		}
	}

	__syncthreads(); // Sync threads to load data into shared memory before proceeding

	// Gaussian blur in X direction
	float sum = 0.0f;
	for (int k = -radius; k <= radius; ++k) {
		sum += kernel1D[k + radius] * tile[ty * tileSize + tx + k];
	}
	__syncthreads(); // Sync threads to ensure all threads have completed the X blur before overwriting the tile
	tile[ty * tileSize + tx] = sum; // Store the result in the tile

	__syncthreads();

	// Gaussian blur in Y direction on the partially blurred tile

	sum = 0.0f;
	for (int k = -radius; k <= radius; ++k) {
		sum += kernel1D[k + radius] * tile[(ty + k) * tileSize + tx];
	}

	if (x < cols && y < rows) {
		// Write the result to the output image
		output[globalIdx] = static_cast<uchar>(sum);
	}
}

float* allocateGaussianKernelGpu(int kWidth, float sigma) {
	float* d_kernel = nullptr;
	size_t size = kWidth * sizeof(float);

	float* h_kernel = generateGaussianKernel1D(kWidth, sigma);

	CUDA_CHECK(cudaMalloc(&d_kernel, size));
	CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, size, cudaMemcpyHostToDevice));
	free(h_kernel);

	return d_kernel;
}

cudaError_t launchGaussianBlur(const uchar* d_input, uchar* d_output, int rows, int cols, int kernelWidth, float sigma, dim3 grid, dim3 block)
{
	float* d_kernel = allocateGaussianKernelGpu(kernelWidth, sigma);

	int radius = kernelWidth / 2;

	size_t sharedMemBytes =
		kernelWidth * sizeof(float) // kernel1D
		+ (BLOCK_SIZE + 2 * radius) * (BLOCK_SIZE + 2 * radius) * sizeof(float); // tile

	gaussianBlurKernel <<<grid, block, sharedMemBytes>>>(d_input, d_output, rows, cols, d_kernel, kernelWidth);
	return cudaSuccess;
}


////// EROSION //////

__global__ void erosionKernel(const uchar* input, uchar* output, int rows, int cols, const uchar* mask, int kWidth, int kHeight) {
	// PLACEHOLDER
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < cols && y < rows) {
		int idx = y * cols + x;
		output[idx] = input[idx];
	}
}

cudaError_t launchErosion(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block)
{
	erosionKernel<<<grid, block >>>(d_input, d_output, rows, cols, d_mask, kernelSize.width, kernelSize.height);
	return cudaSuccess;
}


////// DILATION //////

__global__ void dilationKernel(const uchar* input, uchar* output, int rows, int cols, const uchar* mask, int kWidth, int kHeight) {
	// PLACEHOLDER
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < cols && y < rows) {
		int idx = y * cols + x;
		output[idx] = input[idx];
	}
}

cudaError_t launchDilation(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block)
{
	dilationKernel<<<grid, block>>>(d_input, d_output, rows, cols, d_mask, kernelSize.width, kernelSize.height);
	return cudaSuccess;
}

////// OPENING //////

__global__ void openingKernel(const uchar* input, uchar* output, int rows, int cols, const uchar* mask, int kWidth, int kHeight) {
	// PLACEHOLDER
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < cols && y < rows) {
		int idx = y * cols + x;
		output[idx] = input[idx];
	}
}

cudaError_t launchOpening(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block)
{
	openingKernel<<<grid, block>>>(d_input, d_output, rows, cols, d_mask, kernelSize.width, kernelSize.height);
	return cudaSuccess;
}

////// CLOSING //////

__global__ void closingKernel(const uchar* input, uchar* output, int rows, int cols, const uchar* mask, int kWidth, int kHeight) {
	// PLACEHOLDER
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < cols && y < rows) {
		int idx = y * cols + x;
		output[idx] = input[idx];
	}
}

cudaError_t launchClosing(const uchar* d_input, uchar* d_output, int rows, int cols, const uchar* d_mask, cv::Size kernelSize, dim3 grid, dim3 block)
{
	closingKernel<<<grid, block >>>(d_input, d_output, rows, cols, d_mask, kernelSize.width, kernelSize.height);
	return cudaSuccess;
}